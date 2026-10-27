"""데이터 업데이트 페이지 — Yahoo Finance 시장 지표 · e-Stat JCC 수집, JCC 수동 입력 → Master_Data 시트 반영"""
from datetime import date

import gspread
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from google.oauth2 import service_account

import jcc_estat

# Yahoo Finance 티커 → Master_Data 열 (일별 종가의 월평균)
TICKERS = {
    "CL=F":  "WTI",
    "BZ=F":  "Brent",
    "NG=F":  "HenryHub",
    "JKM=F": "JKM",
    "TTF=F": "TTF",
    "KRW=X": "USD_KRW",
}
COLUMN_NAMES = {  # 화면 표시명
    "WTI": "WTI", "Brent": "브렌트유", "HenryHub": "헨리허브", "JKM": "JKM",
    "TTF": "TTF", "USD_KRW": "환율", "Dubai": "두바이유", "JCC": "JCC",
}
USED = ["JCC", "Brent", "JKM", "USD_KRW"]   # 요금 전망에 쓰는 지표
SHEET = "Master_Data"


@st.cache_resource
def _write_client():
    creds = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    return gspread.authorize(creds)


def _worksheet(sheet_id: str):
    return _write_client().open_by_key(sheet_id).worksheet(SHEET)


def read_master(ws) -> pd.DataFrame:
    """시트 그대로 (월 인덱스, 열 순서 유지)"""
    raw = ws.get_all_values()
    df = pd.DataFrame(raw[1:], columns=raw[0]).replace("", np.nan).dropna(how="all")
    df.index = pd.to_datetime(df.pop(raw[0][0])).dt.to_period("M")
    df = df.apply(lambda s: pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce"))
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.attrs["sheet_rows"] = len(raw)
    return df


def write_master(ws, df: pd.DataFrame):
    """시트 전체를 한 번에 덮어쓰기 (clear 없이 A1부터 → 쓰는 도중 실패해도 기존 데이터가 지워지지 않음)"""
    rows = [["Date", *df.columns]]
    for p, row in df.iterrows():
        rows.append([f"{p}-01", *("" if pd.isna(v) else float(v) for v in row)])
    ws.update(values=rows, range_name="A1", value_input_option="USER_ENTERED")   # 날짜 칸은 날짜 형식 유지
    old_rows = df.attrs.get("sheet_rows", 0)
    if old_rows > len(rows):   # 중복 행 정리 등으로 줄었으면 남는 아래 행 비우기
        last_cell = gspread.utils.rowcol_to_a1(old_rows, len(rows[0]))
        ws.batch_clear([f"A{len(rows) + 1}:{last_cell}"])


def fetch_yahoo(start: pd.Period) -> pd.DataFrame:
    """start 월부터 오늘까지 일별 종가 → 월평균 (이번 달은 오늘까지의 평균)"""
    daily = yf.download(list(TICKERS), start=start.to_timestamp().strftime("%Y-%m-%d"),
                        end=(pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                        interval="1d", progress=False, auto_adjust=False)
    if daily.empty:
        raise RuntimeError("Yahoo Finance에서 데이터를 받지 못했습니다.")
    monthly = daily["Close"].resample("MS").mean().rename(columns=TICKERS)
    monthly.index = monthly.index.to_period("M")
    return monthly.dropna(how="all")


def changes(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """새 값이 기존과 다른 칸 목록 (월, 지표, 기존, 새 값)"""
    out = []
    for p, row in new.iterrows():
        for c, v in row.items():
            if pd.isna(v):
                continue
            before = old.at[p, c] if p in old.index and c in old.columns else np.nan
            if pd.isna(before) or not np.isclose(before, v, rtol=1e-6):
                out.append({"월": str(p), "지표": COLUMN_NAMES.get(c, c), "기존": before, "새 값": v})
    return pd.DataFrame(out, columns=["월", "지표", "기존", "새 값"])


def _merge(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """새 값 우선, 새 값이 비어 있으면 기존 유지 · 기존 열 순서 유지"""
    merged = new.combine_first(old).sort_index()
    merged = merged[[*old.columns, *(c for c in merged.columns if c not in old.columns)]]
    merged.attrs = old.attrs
    return merged


def _apply(ws, merged: pd.DataFrame, message: str):
    write_master(ws, merged)
    st.cache_data.clear()   # 전망 페이지가 새 데이터로 다시 계산되도록
    st.session_state.pop("yahoo_preview", None)
    st.session_state.pop("jcc_preview", None)
    st.session_state["update_done"] = message


def render_status(master: pd.DataFrame):
    rows = []
    for c in master.columns:
        s = master[c].dropna()
        rows.append({
            "지표": COLUMN_NAMES.get(c, c),
            "최근 월": str(s.index[-1]) if len(s) else "-",
            "최근 값": float(s.iloc[-1]) if len(s) else np.nan,
            "수집": "Yahoo Finance" if c in TICKERS.values() else "e-Stat (일본 무역통계)" if c == "JCC" else "수동 입력",
            "전망 사용": "✅" if c in USED else "",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                 column_config={"최근 값": st.column_config.NumberColumn(format="%.2f")})


def _unlocked() -> bool:
    """secrets에 update_password가 있으면 비밀번호 확인, 없으면 누구나 업데이트 가능"""
    password = st.secrets.get("update_password")
    if not password:
        st.caption("⚠️ `update_password`가 secrets에 없어 링크가 있는 누구나 시트를 업데이트할 수 있습니다.")
        return True
    if st.session_state.get("update_unlocked"):
        return True
    entered = st.text_input("업데이트 비밀번호", type="password")
    if entered:
        if entered == password:
            st.session_state["update_unlocked"] = True
            st.rerun()
        st.error("비밀번호가 맞지 않습니다.")
    return False


def render_jcc_auto(ws, master: pd.DataFrame, jcc: pd.Series, months: list):
    app_id = st.secrets.get("ESTAT_APP_ID")
    if not app_id:
        st.info("secrets에 `ESTAT_APP_ID`(e-Stat API 애플리케이션 ID)를 넣으면 사용할 수 있습니다.")
        return

    default_start = (jcc.index[-1] - 6) if len(jcc) else pd.Period("2021-01", "M")
    c1, c2 = st.columns([2, 1])
    start = c1.selectbox("계산 시작 월", months, index=months.index(max(min(default_start, months[-1]), months[0])),
                         format_func=str, key="jcc_start",
                         help="기존 시트 값과 비교·검증하려면 몇 년 전부터 가져와 보세요 (e-Stat DB는 1988년부터 있음).")
    if c2.button("📥 계산하기", use_container_width=True, key="jcc_fetch"):
        with st.spinner("e-Stat에서 원유 수입 통계를 가져오는 중..."):
            try:
                st.session_state["jcc_preview"] = jcc_estat.compute_jcc(app_id, start)
            except Exception as e:
                st.error(f"JCC 계산에 실패했습니다. ({e})")

    calc = st.session_state.get("jcc_preview")
    if calc is None or calc.empty:
        return

    table = pd.DataFrame({
        "월": calc.index.astype(str),
        "수입량(천KL)": calc["qty_kl"].values / 1000,
        "수입금액(억엔)": calc["value_kjpy"].values / 1e5,
        "엔/달러": calc["USD_JPY"].values,
        "계산 JCC": calc["JCC"].values,
        "시트 JCC": jcc.reindex(calc.index).values,
    })
    table["차이"] = table["계산 JCC"] - table["시트 JCC"]
    both = table.dropna(subset=["차이"])
    if len(both):
        mae = both["차이"].abs().mean()
        mape = (both["차이"].abs() / both["시트 JCC"]).mean() * 100
        st.markdown(f"**시트 값과 비교 ({len(both)}개월):** 평균 차이 {both['차이'].mean():+.2f}$ · "
                    f"평균 절대오차 {mae:.2f}$ ({mape:.1f}%)")
    fmt = st.column_config.NumberColumn
    st.dataframe(table, hide_index=True, use_container_width=True, column_config={
        "수입량(천KL)": fmt(format="%.0f"), "수입금액(억엔)": fmt(format="%.0f"), "엔/달러": fmt(format="%.2f"),
        "계산 JCC": fmt(format="%.2f"), "시트 JCC": fmt(format="%.2f"), "차이": fmt(format="%+.2f")})

    missing = calc.index.difference(jcc.index)
    mode = st.radio("반영 방식", ["시트에 없는 달만 채우기", "가져온 모든 달 덮어쓰기"], horizontal=True, key="jcc_mode")
    target = calc.loc[missing if mode.startswith("시트에 없는") else calc.index, ["JCC"]].round(2)
    if target.empty:
        st.info("시트에 없는 달이 없습니다. 반영할 내용이 없습니다.")
        return
    if st.button(f"✅ JCC {len(target)}개월 시트에 반영 ({target.index[0]} ~ {target.index[-1]})",
                 type="primary", use_container_width=True, key="jcc_apply"):
        with st.spinner("시트에 쓰는 중..."):
            _apply(ws, _merge(master, target), f"JCC {len(target)}개월을 반영했습니다 ({target.index[0]} ~ {target.index[-1]}).")
        st.rerun()


def render(sheet_id: str):
    st.title("🔄 데이터 업데이트")
    st.caption(f"[에너지 지표 시트](https://docs.google.com/spreadsheets/d/{sheet_id})의 `{SHEET}` 탭을 갱신합니다. "
               "반영하면 요금 전망 페이지가 새 데이터로 다시 계산됩니다.")

    if msg := st.session_state.pop("update_done", None):
        st.success(msg)

    try:
        ws = _worksheet(sheet_id)
        master = read_master(ws)
    except Exception as e:
        st.error(f"시트를 불러오지 못했습니다. ({e})")
        st.stop()

    st.subheader("현재 데이터")
    render_status(master)

    if not _unlocked():
        st.stop()

    # ── 1. Yahoo Finance 자동 수집
    st.subheader("1. 시장 지표 자동 수집")
    st.markdown("WTI · 브렌트유 · 헨리허브 · JKM · TTF · 환율을 Yahoo Finance에서 가져와 **일별 종가의 월평균**으로 넣습니다.")
    last_auto = max(master[c].dropna().index[-1] for c in TICKERS.values() if c in master.columns)
    months = list(pd.period_range("2015-01", pd.Timestamp.today().to_period("M"), freq="M"))
    default_start = last_auto - 2   # 마지막 달은 월중에 수집된 부분 평균일 수 있어 몇 달 앞부터 다시
    start = st.selectbox("수집 시작 월", months, index=months.index(max(default_start, months[0])),
                         format_func=str, help="이 달부터 오늘까지 다시 가져와 시트 값을 덮어씁니다.")
    st.caption("이번 달은 오늘까지의 평균이라 월말까지 값이 바뀝니다. 다음에 다시 수집하면 갱신됩니다.")

    if st.button("📥 가져오기", use_container_width=True):
        with st.spinner("Yahoo Finance에서 가져오는 중..."):
            try:
                st.session_state["yahoo_preview"] = (start, fetch_yahoo(start))
            except Exception as e:
                st.error(f"수집에 실패했습니다. ({e})")

    if preview := st.session_state.get("yahoo_preview"):
        p_start, new = preview
        diff = changes(master, new)
        if diff.empty:
            st.info(f"{p_start} 이후 값이 시트와 같습니다. 반영할 내용이 없습니다.")
        else:
            st.markdown(f"**바뀌는 값 {len(diff)}개** ({new.index[0]} ~ {new.index[-1]})")
            st.dataframe(diff, hide_index=True, use_container_width=True,
                         column_config={"기존": st.column_config.NumberColumn(format="%.2f"),
                                        "새 값": st.column_config.NumberColumn(format="%.2f")})
            if st.button("✅ 시트에 반영", type="primary", use_container_width=True):
                with st.spinner("시트에 쓰는 중..."):
                    _apply(ws, _merge(master, new), f"시장 지표 {len(diff)}개 값을 반영했습니다.")
                st.rerun()

    jcc = master["JCC"].dropna() if "JCC" in master.columns else pd.Series(dtype=float)

    # ── 2. JCC 자동 수집 (e-Stat)
    st.subheader("2. JCC 자동 수집")
    st.markdown("일본 재무성 무역통계(e-Stat)의 **원유 수입 금액 ÷ 수입량**을 월평균 엔/달러 환율로 나눠 "
                "JCC($/배럴)를 계산합니다. 재무성 발표가 다음 달 하순이라 보통 1~2개월 전 달까지 나옵니다.")
    render_jcc_auto(ws, master, jcc, months)

    # ── 3. JCC 수동 입력
    st.subheader("3. JCC 수동 입력")
    st.markdown("자동 수집이 안 되거나 발표값으로 직접 고칠 때 씁니다.")
    next_month = jcc.index[-1] + 1 if len(jcc) else months[-1]
    with st.form("jcc"):
        c1, c2 = st.columns(2)
        jcc_months = [m for m in months if m <= months[-1]]
        month = c1.selectbox("월", jcc_months, index=jcc_months.index(min(next_month, months[-1])), format_func=str)
        value = c2.number_input("JCC ($/배럴)", 10.0, 300.0, float(jcc.iloc[-1]) if len(jcc) else 70.0,
                                step=0.01, format="%.2f")
        if month in jcc.index:
            st.caption(f"{month}에는 이미 {jcc[month]:.2f}가 있습니다. 저장하면 덮어씁니다.")
        if st.form_submit_button("💾 저장", type="primary", use_container_width=True):
            new = pd.DataFrame({"JCC": [value]}, index=pd.PeriodIndex([month], freq="M"))
            with st.spinner("시트에 쓰는 중..."):
                _apply(ws, _merge(master, new), f"{month} JCC {value:.2f}를 저장했습니다.")
            st.rerun()
    st.caption(f"최근 JCC: {jcc.index[-1]} {jcc.iloc[-1]:.2f} $/배럴" if len(jcc) else "")
