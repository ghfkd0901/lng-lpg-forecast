"""JCC 자동 계산 — 일본 재무성 무역통계(e-Stat API)의 원유 수입 금액·수량으로 산출

JCC(Japan Crude Cocktail)는 일본 원유 수입 평균 통관가격(CIF)입니다.
  · e-Stat 「普通貿易統計 貿易統計_全国分 概況品別国別表 輸入」의 概況品 30301 原油及び粗油
  · 국가별 수입 금액(千円)·수량(KL)을 합산 → 엔/KL → 엔/배럴 → 월평균 엔/달러 환율로 $/배럴
재무성이 다음 달 하순쯤 발표하므로 보통 1~2개월 전 달까지 나옵니다.
"""
import re

import numpy as np
import pandas as pd
import requests
import yfinance as yf

API = "https://api.e-stat.go.jp/rest/3.0/app/json"
STATS_CODE = "00350300"       # 政府統計コード: 普通貿易統計
CRUDE_CODE = "30301"          # 概況品: 原油及び粗油 (API 코드는 30301000)
BBL_PER_KL = 1 / 0.158987294928


class EStatError(RuntimeError):
    pass


def _as_list(x):
    return [] if x is None else x if isinstance(x, list) else [x]


def _text(x) -> str:
    return x.get("$", "") if isinstance(x, dict) else str(x or "")


def _call(app_id: str, method: str, **params) -> dict:
    r = requests.get(f"{API}/{method}", params={"appId": app_id, "lang": "J", **params}, timeout=60)
    r.raise_for_status()
    root = next(iter(r.json().values()))          # GET_STATS_LIST / GET_META_INFO / GET_STATS_DATA
    result = root.get("RESULT", {})
    if int(result.get("STATUS", 0)) >= 100:       # 0 정상, 1 데이터 없음, 2 일부 파라미터 무시 / 100 이상 오류
        raise EStatError(f"e-Stat 오류 {result.get('STATUS')}: {result.get('ERROR_MSG')}")
    return root


def find_tables(app_id: str) -> list[dict]:
    """전국 概況品別国別表 輸入 통계표 목록 (공개일 오래된 순 → 나중 표가 같은 달을 덮어씀)"""
    root = _call(app_id, "getStatsList", statsCode=STATS_CODE, searchWord="概況品別国別表 AND 輸入")
    tables = []
    for t in _as_list(root.get("DATALIST_INF", {}).get("TABLE_INF")):
        title = _text(t.get("TITLE"))
        stat = _text(t.get("STATISTICS_NAME"))
        if "概況品別国別表" in title and "輸入" in title and "税関別" not in title + stat:
            years = [int(y) for y in re.findall(r"(?<!\d)(20\d\d|19\d\d)(?!\d)", title)]
            tables.append({"id": t["@id"], "title": title, "open": str(t.get("OPEN_DATE", "")),
                           "years": (min(years), max(years)) if years else None})
    if not tables:
        raise EStatError("e-Stat에서 概況品別国別表 輸入 통계표를 찾지 못했습니다.")
    return sorted(tables, key=lambda t: t["open"])


def _classes(app_id: str, table_id: str) -> dict:
    meta = _call(app_id, "getMetaInfo", statsDataId=table_id)
    return {c["@id"]: {"name": c.get("@name", ""), "items": _as_list(c.get("CLASS"))}
            for c in _as_list(meta["METADATA_INF"]["CLASS_INF"]["CLASS_OBJ"])}


def _crude_code(classes: dict) -> tuple[str, str]:
    """(概況品 분류 id, 원유 코드)"""
    for cid, c in classes.items():
        if "概況品" in c["name"] and "数量" not in c["name"]:
            for it in c["items"]:
                if it["@code"] == CRUDE_CODE or it.get("@name", "").startswith(f"{CRUDE_CODE}_"):
                    return cid, it["@code"]
    raise EStatError("통계표에서 '30301 原油及び粗油' 항목을 찾지 못했습니다.")


def _measure_class(classes: dict) -> str:
    """월별 수량·금액 분류 id (예: cat02 '概況品目表の数量・金額')"""
    for cid, c in classes.items():
        if "数量" in c["name"] and "金額" in c["name"]:
            return cid
    return "tab"


def _tab_map(classes: dict, mid: str) -> dict:
    """수량·금액 항목 코드 → (월, '数量'|'金額', 단위) · 합계·단위 항목은 제외"""
    out = {}
    for it in classes.get(mid, {}).get("items", []):
        name = it.get("@name", "")
        m = re.search(r"(\d{1,2})月", name)
        kind = "金額" if "金額" in name else "数量" if "数量" in name else None
        if m and kind and not re.search(r"合計|累計|計_", name):
            if kind == "数量" and re.search(r"数量2", name):      # 제2수량 단위는 사용 안 함
                continue
            out[it["@code"]] = (int(m.group(1)), kind, it.get("@unit", ""))
    if not out:
        names = [it.get("@name", "") for it in classes.get(mid, {}).get("items", [])][:8]
        raise EStatError(f"월별 수량·금액 항목을 해석하지 못했습니다. (항목 예: {names})")
    return out


def _year_map(classes: dict) -> dict:
    out = {}
    for it in classes.get("time", {}).get("items", []):
        m = re.search(r"(\d{4})", it.get("@name", "") or it["@code"])
        if m:
            out[it["@code"]] = int(m.group(1))
    return out


def _num(v) -> float:
    try:
        return float(str(v).replace(",", ""))
    except ValueError:
        return np.nan          # '-', '***', 'X' 등


def fetch_crude_imports(app_id: str, start: pd.Period) -> pd.DataFrame:
    """월별 원유 수입 금액(千円)·수량(KL) — start 연도 이후를 담은 통계표만 조회"""
    frames = []
    for t in find_tables(app_id):
        if t["years"] and t["years"][1] < start.year:
            continue
        classes = _classes(app_id, t["id"])
        cid, code = _crude_code(classes)
        mid = _measure_class(classes)
        tabs, years = _tab_map(classes, mid), _year_map(classes)
        single_year = t["years"][0] if t["years"] and t["years"][0] == t["years"][1] else None
        root = _call(app_id, "getStatsData", statsDataId=t["id"],
                     **{f"cd{cid[0].upper()}{cid[1:]}": code}, metaGetFlg="N", cntGetFlg="N")
        values = _as_list(root.get("STATISTICAL_DATA", {}).get("DATA_INF", {}).get("VALUE"))
        rows = []
        for v in values:
            tab = tabs.get(v.get(f"@{mid}"))
            year = years.get(v.get("@time")) or single_year
            if tab and year:
                month, kind, unit = tab
                rows.append({"month": pd.Period(year=year, month=month, freq="M"), "kind": kind,
                             "unit": v.get("@unit") or unit, "value": _num(v["$"])})
        if rows:
            df = pd.DataFrame(rows)
            # 千KL 단위면 KL로
            df.loc[(df["kind"] == "数量") & df["unit"].str.contains("千", na=False), "value"] *= 1000
            frames.append(df.groupby(["month", "kind"])["value"].sum(min_count=1).unstack())
    if not frames:
        raise EStatError("원유 수입 데이터를 받지 못했습니다.")
    out = frames[0]
    for f in frames[1:]:
        out = f.combine_first(out)                 # 나중에 공개된 표(확정·확보) 우선
    out = out.rename(columns={"金額": "value_kjpy", "数量": "qty_kl"}).dropna()
    out = out[(out["qty_kl"] > 0) & (out.index >= start)].sort_index()
    return out


def fetch_usdjpy(start: pd.Period) -> pd.Series:
    daily = yf.download("JPY=X", start=start.to_timestamp().strftime("%Y-%m-%d"), progress=False,
                        auto_adjust=False)["Close"]
    if isinstance(daily, pd.DataFrame):
        daily = daily.iloc[:, 0]
    monthly = daily.resample("MS").mean()
    monthly.index = monthly.index.to_period("M")
    return monthly.rename("USD_JPY")


def compute_jcc(app_id: str, start: pd.Period) -> pd.DataFrame:
    """월별 JCC 계산표: 금액(千円), 수량(KL), 엔/KL, 엔/달러, JCC($/배럴)"""
    imp = fetch_crude_imports(app_id, start)
    fx = fetch_usdjpy(imp.index.min())
    df = imp.join(fx, how="left")
    df["JPY_per_KL"] = df["value_kjpy"] * 1000 / df["qty_kl"]
    df["JCC"] = df["JPY_per_KL"] / BBL_PER_KL / df["USD_JPY"]
    df = df.dropna(subset=["JCC"])
    bad = df[(df["JCC"] < 10) | (df["JCC"] > 300)]
    if len(bad):
        raise EStatError(f"계산된 JCC가 비정상입니다 (예: {bad.index[0]} {bad['JCC'].iloc[0]:.1f}$). 단위를 확인하세요.")
    return df
