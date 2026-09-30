import warnings
warnings.filterwarnings("ignore")

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn
import xgboost
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import gspread
from google.oauth2 import service_account
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import Holt, SimpleExpSmoothing
from xgboost import XGBRegressor

st.set_page_config(page_title="LNG·LPG 요금 전망", page_icon="📊", layout="centered")

# ───────────────────────────────
# 상수
# ───────────────────────────────
SHEET_ID     = "1Y5hxWDA_SRXyXhZF7bZLmI8MFQg9r78HLiHoSFVeneE"
# 요금비교 시트 (링크 공개): 원료비·공급비용, 산업용 요금, LPG 가격
TARIFF_CSV   = (
    "https://docs.google.com/spreadsheets/d/"
    "12RGk0NyM24_zxLIJXNAobcinZ714kdDKeeoDSt9Hb9c/export?format=csv&gid=0"
)
TRAIN_START  = pd.Period("2015-01", "M")
# 코로나 유가 급락·회복 구간 (학습 제외 옵션)
COVID_PERIOD = (pd.Period("2020-03", "M"), pd.Period("2021-12", "M"))
CHART_START  = pd.Period("2020-01", "M")
MAX_HORIZON  = 24
MODEL_DIR    = Path(__file__).parent / "models"   # 학습한 모델 저장 위치

# ㎥ 환산 — 도시가스는 요금비교 시트의 최신 열량(MJ/㎥), LPG는 프로판 기준
LPG_MJ_PER_KG = 50.2                  # 프로판 총발열량 (시트의 원/kg ↔ 원/MJ 환산과 일치)
LPG_KG_PER_M3 = 44.097 / 22.414       # 프로판 기체 밀도 약 1.967 kg/㎥ (0℃, 1기압)

# 연료별 입력 지표와 요금 반영 시차(개월) — 환율(USD_KRW)은 당월 값을 공통으로 사용
FUELS = {
    "LNG": {"label": "LNG 도매요금", "inputs": {"JCC": 4, "JKM": 2}, "digits": 4,
            "color": "#2a78d6", "band": "rgba(42, 120, 214, 0.15)"},
    "LPG": {"label": "LPG 요금",    "inputs": {"Brent": 1}, "digits": 2,
            "color": "#e07b39", "band": "rgba(224, 123, 57, 0.15)"},
}
INDICATORS = {  # Master_Data 열 → (표시명, 단위, 입력 범위)
    "JCC":   ("JCC",     "$/배럴", (10.0, 300.0)),
    "Brent": ("브렌트유", "$/배럴", (10.0, 300.0)),
    "JKM":   ("JKM",     "$/MMBtu", (1.0, 100.0)),
}
# 발표가 늦은 지표는 미발표 달을 선행 지표로 추정 — JCC ≈ 1개월 전 브렌트유 (상관계수 0.98)
PROXIES = {"JCC": ("Brent", 1)}

COLOR_ACTUAL = "#52514e"


# ───────────────────────────────
# 데이터 로드 (Google Sheets, 1시간 캐시)
# ───────────────────────────────
@st.cache_resource
def get_gspread_client():
    creds = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"],
    )
    return gspread.authorize(creds)


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce")


def _sheet_to_df(ws) -> pd.DataFrame:
    raw = ws.get_all_values()
    df = pd.DataFrame(raw[1:], columns=raw[0]).replace("", np.nan)
    df = df.dropna(how="all").dropna(axis=1, how="all")

    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    df = df.apply(_to_num)

    df.index = df.index.to_period("M")
    df = df[~df.index.duplicated(keep="last")]
    # 월 단위로 빈 달 없이 정렬 (shift = 정확히 N개월)
    return df.reindex(pd.period_range(df.index.min(), df.index.max(), freq="M"))


@st.cache_data(ttl=3600, show_spinner=False)
def load_data() -> dict:
    sh = get_gspread_client().open_by_key(SHEET_ID)
    master = _sheet_to_df(sh.worksheet("Master_Data"))   # 열 선택은 캐시 밖에서 (지표 추가 시 캐시 꼬임 방지)
    gas = _sheet_to_df(sh.worksheet("gas_price"))
    wholesale = gas.iloc[:, 0].rename("Wholesale_Price").dropna()

    tariff = pd.read_csv(TARIFF_CSV)
    tariff.columns = tariff.columns.str.replace("\n", " ").str.strip()
    tariff.index = pd.to_datetime(tariff.pop("Date")).dt.to_period("M")
    tariff = tariff[~tariff.index.duplicated(keep="last")].sort_index().apply(_to_num)

    # gas_price 탭 이후 달은 원료비 + 가스공사 공급비용(= 도매요금)으로 이어 붙임
    recent = (tariff["원료비"] + tariff["가스공사 공급비용"]).dropna()
    wholesale = pd.concat([wholesale, recent[recent.index > wholesale.index[-1]]])

    return {
        "master":      master,
        "LNG":         wholesale,
        "LPG":         tariff["LPG_SK가스 가정상업용 (mj, VAT별도)"].dropna(),  # 원/MJ, VAT 별도
        "lng_retail":  tariff["산업용_원/MJ"].dropna(),                          # 산업용 요금 실적
        "retail_cost": float(tariff["대성에너지 공급비용"].dropna().iloc[-1]),   # 도매 → 산업용 가산분
        "heat":        float(tariff["열량"].dropna().iloc[-1]),                 # 도시가스 열량 (MJ/㎥)
    }


# ───────────────────────────────
# 모델 학습 (데이터가 바뀔 때만 재학습)
# ───────────────────────────────
def feature_name(col: str, lag: int) -> str:
    return f"{col}_Lag{lag}"


def training_xy(master: pd.DataFrame, target: pd.Series, inputs: dict, exclude_covid: bool = False):
    feats = master.ffill()
    X = pd.DataFrame({feature_name(c, lag): feats[c].shift(lag) for c, lag in inputs.items()})
    X["USD_KRW"] = feats["USD_KRW"]
    df = X.join(target.rename("y"), how="inner").dropna()
    df = df[df.index >= TRAIN_START]
    if exclude_covid:
        df = df[(df.index < COVID_PERIOD[0]) | (df.index > COVID_PERIOD[1])]
    return df[X.columns], df["y"]


MODEL_SPECS = {
    "Linear Regression": lambda: LinearRegression(),
    "Random Forest":     lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": lambda: GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost":           lambda: XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
}


def _model_path(folder: Path, name: str) -> Path:
    return folder / f"{name.lower().replace(' ', '_')}.joblib"


def _lib_versions() -> dict:
    return {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__}


def _load_saved(folder: Path, data_key: str) -> dict | None:
    """저장된 모델이 같은 학습 데이터·같은 라이브러리 버전으로 만들어졌으면 불러오기"""
    try:
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        if meta["data_key"] != data_key or meta.get("versions") != _lib_versions():
            return None
        return {name: joblib.load(_model_path(folder, name)) for name in MODEL_SPECS}
    except Exception:
        return None


@st.cache_resource(show_spinner=False, max_entries=4)
def train_models(name: str, data_key: str, _X: pd.DataFrame, _y: pd.Series) -> dict:
    """저장된 모델 우선 사용, 학습 데이터가 바뀌었을 때만 재학습 후 파일로 저장"""
    folder = MODEL_DIR / name
    models = _load_saved(folder, data_key)
    if models is not None:
        return models

    models = {m: make().fit(_X, _y) for m, make in MODEL_SPECS.items()}
    try:
        folder.mkdir(parents=True, exist_ok=True)
        for m, model in models.items():
            joblib.dump(model, _model_path(folder, m))
        meta = {"data_key": data_key, "trained_at": datetime.now().isoformat(timespec="seconds"),
                "features": list(_X.columns), "rows": len(_X),
                "train_range": f"{_X.index[0]} ~ {_X.index[-1]}", "versions": _lib_versions()}
        (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass  # 저장 실패(읽기 전용 환경 등)해도 학습한 모델로 계속 진행
    return models


def predict_all(models: dict, X: pd.DataFrame) -> np.ndarray:
    """(행 수, 모델 수) 예측 행렬"""
    return np.column_stack([m.predict(X) for m in models.values()])


def scenario_X(inputs: dict, values: dict, fx: float) -> pd.DataFrame:
    X = pd.DataFrame({feature_name(c, lag): [values[c]] for c, lag in inputs.items()})
    X["USD_KRW"] = fx
    return X


# ───────────────────────────────
# 미래 지표 (실제값 우선, 없으면 Holt 추세 예측)
# ───────────────────────────────
def holt_forecast(values: np.ndarray, n: int) -> np.ndarray:
    try:
        fc = Holt(values, exponential=False).fit(optimized=True).forecast(n)
        return np.clip(fc, values.min() * 0.5, values.max() * 2.0)
    except Exception:
        return SimpleExpSmoothing(values).fit(optimized=True).forecast(n)


def project(series: pd.Series, periods: list) -> tuple[list, list]:
    s = series.dropna()
    last = s.index[-1]
    n_ahead = max(0, max((p - last).n for p in periods))
    fc = holt_forecast(s.to_numpy(dtype=float), n_ahead) if n_ahead else []

    values, is_actual = [], []
    for p in periods:
        if p <= last:
            values.append(float(s[s.index <= p].iloc[-1]))
            is_actual.append(True)
        else:
            values.append(float(fc[(p - last).n - 1]))
            is_actual.append(False)
    return values, is_actual


def proxy_fit(master: pd.DataFrame, col: str) -> tuple[float, float]:
    """col ≈ intercept + slope × 선행 지표(시차 적용) 선형회귀 계수"""
    src, lag = PROXIES[col]
    fit = pd.concat([master[col], master[src].shift(lag)], axis=1).dropna()
    slope, intercept = np.polyfit(fit.iloc[:, 1], fit.iloc[:, 0], 1)
    return slope, intercept


def project_indicator(master: pd.DataFrame, col: str, periods: list) -> tuple[list, list]:
    """PROXIES에 있는 지표는 미발표 달을 선행 지표의 선형회귀로 추정, 나머지는 Holt 추세"""
    if col not in PROXIES:
        return project(master[col], periods)

    src, lag = PROXIES[col]
    s = master[col].dropna()
    slope, intercept = proxy_fit(master, col)
    src_values, _ = project(master[src], [p - lag for p in periods])

    values, is_actual = [], []
    for p, x in zip(periods, src_values):
        if p <= s.index[-1]:
            values.append(float(s[s.index <= p].iloc[-1]))
            is_actual.append(True)
        else:
            values.append(intercept + slope * x)
            is_actual.append(False)
    return values, is_actual


@st.cache_data(show_spinner=False)
def build_forecast(data_key: str, _master: pd.DataFrame, _models: dict, start: pd.Period, inputs: dict) -> pd.DataFrame:
    periods = [start + i for i in range(MAX_HORIZON)]
    X, all_actual = pd.DataFrame(index=range(MAX_HORIZON)), np.ones(MAX_HORIZON, dtype=bool)
    for c, lag in inputs.items():
        values, actual = project_indicator(_master, c, [p - lag for p in periods])
        X[feature_name(c, lag)] = values
        all_actual &= np.array(actual)
    X["USD_KRW"], _ = project(_master["USD_KRW"], periods)

    preds = predict_all(_models, X)

    fc = pd.DataFrame({
        "월":       [str(p) for p in periods],
        "예상요금":  preds.mean(axis=1),
        "최저":      preds.min(axis=1),
        "최고":      preds.max(axis=1),
    })
    for c, lag in inputs.items():
        fc[c] = X[feature_name(c, lag)]
    fc["환율"] = X["USD_KRW"]
    # 입력 지표가 모두 발표된 실적이면 확정, 하나라도 추세 예측값이면 추정
    fc["지표 구분"] = np.where(all_actual, "확정 지표", "추정 지표")
    fc.index = pd.PeriodIndex(periods, freq="M")
    return fc


def feature_label(col: str, lag: int) -> str:
    return f"{INDICATORS[col][0]}({lag}개월전)"


FEATURE_ORDER = ["JCC", "JKM", "USD_KRW", "Brent"]   # 표의 입력 지표 열 순서


def ordered_features(inputs: dict) -> list[tuple[str, str, int]]:
    """(지표, 열 이름, 시차) 목록 — 환율(당월) 포함, FEATURE_ORDER 순"""
    items = {c: feature_label(c, lag) for c, lag in inputs.items()}
    items["USD_KRW"] = "환율(당월)"
    lags = {**inputs, "USD_KRW": 0}
    return [(c, items[c], lags[c]) for c in FEATURE_ORDER if c in items]


def feature_frame(master: pd.DataFrame, periods) -> pd.DataFrame:
    """각 월의 요금에 반영되는 입력 지표 — 시차만큼 이전 달 값을 같은 행에"""
    all_inputs = {c: lag for cfg in FUELS.values() for c, lag in cfg["inputs"].items()}
    return pd.DataFrame({
        label: project_indicator(master, c, [p - lag for p in periods])[0]
        for c, label, lag in ordered_features(all_inputs)
    })


def compare_frame(data: dict, fcs: dict, start: pd.Period, n: int) -> pd.DataFrame:
    """LNG 산업용(도매 전망 + 공급비용) vs LPG, 실적이 있는 달은 실적 사용"""
    periods = pd.period_range(start, periods=n, freq="M")

    def path(actual: pd.Series, fc: pd.Series) -> pd.Series:
        return pd.concat([actual, fc[fc.index > actual.index[-1]]]).reindex(periods)

    lng = path(data["lng_retail"], fcs["LNG"]["예상요금"] + data["retail_cost"])
    lpg = path(data["LPG"], fcs["LPG"]["예상요금"])
    is_actual = (periods <= data["lng_retail"].index[-1]) & (periods <= data["LPG"].index[-1])

    return pd.DataFrame({
        "월":         [str(p) for p in periods],
        "LNG":        lng.values,
        "LPG":        lpg.values,
        "차이":       (lpg - lng).values,
        "LNG 절감률": ((lpg - lng) / lpg * 100).values,
        "구분":       np.where(is_actual, "실적", "전망"),
    })


def to_view(data: dict, fcs: dict, factors: dict) -> tuple[dict, dict]:
    """화면 표시 단위로 환산한 복사본 (factors: 연료별 원/MJ → 표시 단위 배수)"""
    f_lng, f_lpg = factors["LNG"], factors["LPG"]
    view_data = {**data, "LNG": data["LNG"] * f_lng, "lng_retail": data["lng_retail"] * f_lng,
                 "retail_cost": data["retail_cost"] * f_lng, "LPG": data["LPG"] * f_lpg}
    view_fcs = {}
    for fuel, fc in fcs.items():
        fc = fc.copy()
        fc[["예상요금", "최저", "최고"]] *= factors[fuel]
        view_fcs[fuel] = fc
    return view_data, view_fcs


# ───────────────────────────────
# 차트
# ───────────────────────────────
def _add_fuel_traces(fig: go.Figure, actual: pd.Series, fc: pd.DataFrame, name: str,
                     color: str, band: str, actual_color: str, digits: int, unit: str, offset: float = 0.0,
                     yaxis: str = "y"):
    hist = actual[actual.index >= CHART_START]
    hist_x = hist.index.to_timestamp()
    fc = fc[fc.index > actual.index[-1]]
    fc_x = fc.index.to_timestamp()
    mean, lo, hi = fc["예상요금"] + offset, fc["최저"] + offset, fc["최고"] + offset
    fmt = f"%{{y:,.{digits}f}} {unit}"

    fig.add_trace(go.Scatter(
        x=[*fc_x, *fc_x[::-1]], y=[*hi, *lo[::-1]],
        fill="toself", fillcolor=band, line=dict(width=0),
        name=f"{name} 모델 간 범위", hoverinfo="skip", showlegend=False, yaxis=yaxis,
    ))
    fig.add_trace(go.Scatter(
        x=hist_x, y=hist.values, name=f"{name} 실적",
        line=dict(color=actual_color, width=2),
        hovertemplate=f"{name} 실적 {fmt}<extra></extra>", yaxis=yaxis,
    ))
    # 전망선을 마지막 실적과 이어서 표시
    fig.add_trace(go.Scatter(
        x=[hist_x[-1], *fc_x], y=[hist.iloc[-1], *mean], name=f"{name} 전망",
        line=dict(color=color, width=2, dash="dot"),
        hovertemplate=f"{name} 전망 {fmt}<extra></extra>", yaxis=yaxis,
    ))


def _layout(fig: go.Figure, unit: str) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=420, hovermode="x unified",
        margin=dict(t=40, b=20, l=10, r=10),
        yaxis=dict(title=unit, gridcolor="#ecebe8"),
        xaxis=dict(showgrid=False, hoverformat="%Y-%m"),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0),
    )
    return fig


def plot_forecast(actual: pd.Series, fc: pd.DataFrame, fuel: str, unit: str, digits: int) -> go.Figure:
    cfg = FUELS[fuel]
    fig = go.Figure()
    _add_fuel_traces(fig, actual, fc, fuel, cfg["color"], cfg["band"], COLOR_ACTUAL, digits, unit)

    estimated = fc[fc["지표 구분"] == "추정 지표"]
    if not estimated.empty:
        boundary = estimated.index[0].to_timestamp()
        fig.add_vline(x=boundary, line=dict(color="#a3a29d", width=1, dash="dash"))
        fig.add_annotation(
            x=boundary, y=1, yref="paper", yanchor="bottom", xanchor="left",
            text=" 이후 추정 지표 반영", showarrow=False, font=dict(size=11, color="#52514e"),
        )
    return _layout(fig, unit)


def plot_compare(data: dict, fcs: dict, end: pd.Period, unit: str, digits: int, dual_axis: bool) -> go.Figure:
    """dual_axis: 두 연료의 값 크기가 크게 다를 때(㎥) LNG는 왼쪽, LPG는 오른쪽 축"""
    fig = go.Figure()
    for fuel, actual, offset, yaxis in (("LNG", data["lng_retail"], data["retail_cost"], "y"),
                                        ("LPG", data["LPG"], 0.0, "y2" if dual_axis else "y")):
        cfg = FUELS[fuel]
        fc = fcs[fuel][fcs[fuel].index <= end]
        _add_fuel_traces(fig, actual, fc, fuel, cfg["color"], cfg["band"], cfg["color"], digits, unit, offset, yaxis)
    _layout(fig, unit)
    if dual_axis:
        fig.update_layout(
            yaxis=dict(title=dict(text=f"LNG ({unit})", font=dict(color=FUELS["LNG"]["color"])),
                       tickfont=dict(color=FUELS["LNG"]["color"]), automargin=True),
            yaxis2=dict(title=dict(text=f"LPG ({unit})", font=dict(color=FUELS["LPG"]["color"])),
                        tickfont=dict(color=FUELS["LPG"]["color"]),
                        overlaying="y", side="right", showgrid=False, automargin=True),
        )
    return fig


# ───────────────────────────────
# 화면 구성
# ───────────────────────────────
def render_fuel_tab(fuel: str, actual: pd.Series, fc: pd.DataFrame, source_note: str, unit: str, d: int):
    cfg = FUELS[fuel]
    last_period, last_price = actual.index[-1], float(actual.iloc[-1])

    k1, k2, k3 = st.columns(3)
    k1.metric(f"최근 실적 ({last_period})", f"{last_price:,.{d}f}")
    for col, i in ((k2, 0), (k3, 5)):
        row = fc.iloc[i]
        col.metric(
            f"{'다음 달' if i == 0 else f'{i + 1}개월 후'} 전망 ({row['월']})",
            f"{row['예상요금']:,.{d}f}",
            delta=f"{row['예상요금'] - last_price:+,.{d}f}",
            delta_color="inverse",
        )
    st.caption(f"단위: {unit} · {source_note}")

    horizon = st.radio(
        "전망 기간", [6, 12, 24], index=1, horizontal=True,
        format_func=lambda n: f"{n}개월", key=f"horizon_{fuel}",
    )
    view = fc.head(horizon)

    st.plotly_chart(plot_forecast(actual, view, fuel, unit, d), use_container_width=True)

    # 입력 지표는 시차만큼 이전 달 값이 같은 행에 들어 있음 → 열 이름에 시차 표시
    table = view.rename(columns={
        **{c: feature_label(c, lag) for c, lag in cfg["inputs"].items()}, "환율": "환율(당월)",
    })
    feature_cols = [label for _, label, _ in ordered_features(cfg["inputs"])]
    table = table[["월", *feature_cols, "예상요금", "최저", "최고", "지표 구분"]]
    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        column_config={
            "예상요금": st.column_config.NumberColumn(f"예상요금 ({unit})", format=f"%.{d}f"),
            "최저":     st.column_config.NumberColumn("모델 최저", format=f"%.{d}f"),
            "최고":     st.column_config.NumberColumn("모델 최고", format=f"%.{d}f"),
            **feature_column_config(),
        },
    )
    st.caption(
        f"‘확정 지표’는 {describe_inputs(cfg['inputs'])} 실적이 모두 발표된 달, "
        "‘추정 지표’는 추세 예측값을 사용한 달로 불확실성이 더 큽니다."
    )
    st.download_button(
        "⬇️ 전망표 다운로드 (CSV)",
        table.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{fuel}_전망_{last_period}.csv",
        mime="text/csv",
        key=f"download_{fuel}",
    )


def feature_column_config() -> dict:
    """입력 지표 열 표시 형식 (단위는 도움말에)"""
    config = {"환율(당월)": st.column_config.NumberColumn(format="%.0f", help="원/$ · 해당 월")}
    for cfg in FUELS.values():
        for c, lag in cfg["inputs"].items():
            config[feature_label(c, lag)] = st.column_config.NumberColumn(
                format="%.2f", help=f"{INDICATORS[c][1]} · 해당 월보다 {lag}개월 전 값",
            )
    return config


def describe_inputs(inputs: dict) -> str:
    return " · ".join(f"{INDICATORS[c][0]} {lag}개월" for c, lag in inputs.items())


def render_model_guide(master: pd.DataFrame, data: dict, train_info: dict, exclude_covid: bool):
    last = {c: master[c].dropna().index[-1] for c in master.columns}
    lng_in, lpg_in = FUELS["LNG"]["inputs"], FUELS["LPG"]["inputs"]
    proxy_lines = []
    for col, (src, lag) in PROXIES.items():
        slope, intercept = proxy_fit(master, col)
        proxy_lines.append(
            f"- **{INDICATORS[col][0]}**: {last[col]}까지 실적, 이후는 "
            f"`{INDICATORS[col][0]} = {intercept:.2f} + {slope:.3f} × {lag}개월 전 {INDICATORS[src][0]}` 회귀식으로 추정"
        )
    holt_cols = [c for c in [*INDICATORS, "USD_KRW"] if c not in PROXIES]
    holt_lines = [
        f"- **{INDICATORS[c][0] if c in INDICATORS else '환율'}**: {last[c]}까지 실적, 이후는 Holt 추세 예측"
        for c in holt_cols
    ]
    covid_from, covid_to = COVID_PERIOD
    train_lines = [
        f"    - {fuel}: {start} ~ {end} 중 요금 실적이 있는 {rows}개월"
        + (f" (코로나 기간 {covid_from} ~ {covid_to} 제외)" if exclude_covid else "")
        for fuel, (rows, start, end) in train_info.items()
    ]

    st.markdown(f"""
**1. 무엇을 예측하나요**
- **LNG**: 산업용 천연가스 도매요금(원료비 + 가스공사 공급비용)을 예측한 뒤, 대성에너지 공급비용
  ({data['retail_cost']:.4f} 원/MJ)을 더해 산업용 요금으로 LPG와 비교합니다.
- **LPG**: SK가스 가정·상업용 공급가격(원/MJ, VAT 별도)을 예측합니다.

**2. 입력 변수와 반영 시차**

| 연료 | 입력 변수 | 의미 |
|---|---|---|
| LNG | {describe_inputs(lng_in)} · 환율 당월 | 가스공사 도입 계약이 JCC(일본 원유 수입가격)에 연동되어 몇 달 뒤 요금에 반영되고, 현물 LNG 가격(JKM)이 단기 비용에 영향 |
| LPG | {describe_inputs(lpg_in)} · 환율 당월 | 사우디 CP(국제 LPG 가격)가 국제유가를 따라가며 다음 달 공급가격에 반영 |

시차는 과거 데이터로 여러 시차를 시험해(시계열 교차검증) 오차가 가장 작은 값을 골랐습니다.

**3. 예측 모델**
- 선형회귀 · 랜덤포레스트 · 그래디언트 부스팅 · XGBoost **4개 모델의 평균**을 예상요금으로 씁니다.
  그래프의 음영과 표의 최저·최고는 4개 모델 중 가장 낮은 값과 높은 값입니다.
- 학습 기간:
{chr(10).join(train_lines)}
- 학습한 모델은 파일로 저장해 두고, 시트 데이터가 바뀔 때만 다시 학습합니다.

**4. 앞으로의 입력값은 어떻게 넣나요**

요금이 반영되는 달의 입력값이 이미 발표됐으면 **실적**을, 아직 없으면 **추정값**을 씁니다.
{chr(10).join(proxy_lines)}
{chr(10).join(holt_lines)}

Holt 추세 예측은 최근 수준과 기울기를 이어가는 지수평활 방식이며, 과거 최저값의 절반 ~ 최고값의 2배 범위로 제한합니다.
JCC는 발표가 한 달가량 늦어서, JCC와 움직임이 거의 같은(상관계수 약 0.98) 1개월 전 브렌트유로 추정합니다.
표의 **지표 구분**은 입력값이 모두 실적이면 ‘확정 지표’, 하나라도 추정값이면 ‘추정 지표’입니다.

**5. 단위와 비교 기준**
- 기본 단위는 원/MJ(열량 기준)이고, VAT는 제외했습니다.
- ㎥ 환산: 도시가스 {data['heat']:.3f} MJ/㎥(최신 열량), LPG {LPG_MJ_PER_KG * LPG_KG_PER_M3:.1f} MJ/㎥
  (프로판 {LPG_MJ_PER_KG} MJ/kg × {LPG_KG_PER_M3:.3f} kg/㎥, 0℃·1기압).
- **LNG 절감률**은 단위와 관계없이 같은 열량 기준으로 계산합니다.

**6. 유의사항**
- 국제 가격과 환율의 과거 관계를 바탕으로 한 통계적 추정입니다. 정부의 요금 동결·조정, 미수금 정산,
  공급비용 변경 같은 정책 요인은 반영하지 않습니다.
- ‘추정 지표’ 구간은 입력값 자체가 추정이라 불확실성이 더 큽니다.
""")


def saving_text(lng: float, lpg: float) -> str:
    pct = (lpg - lng) / lpg * 100
    return f"LNG가 LPG보다 {pct:.1f}% 저렴" if pct >= 0 else f"LNG가 LPG보다 {-pct:.1f}% 비쌈"


# ═══════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════
def main():
    st.title("📊 LNG · LPG 요금 전망")
    covid_from, covid_to = COVID_PERIOD
    covid_label = f"코로나 기간 제외 ({covid_from.year}.{covid_from.month} ~ {covid_to.year}.{covid_to.month})"
    exclude_covid = st.radio(
        "학습 데이터", ["전체 기간", covid_label], horizontal=True,
        help="코로나 기간 제외: 유가가 급락했다가 회복하던 비정상 구간을 빼고 모델을 학습합니다.",
    ) == covid_label
    per_m3 = st.toggle("㎥ 단위로 보기", help="끄면 원/MJ(열량 기준), 켜면 원/㎥(부피 기준)로 표시합니다.")

    with st.spinner("최신 데이터를 불러오는 중..."):
        try:
            data = load_data()
        except Exception as e:
            st.error(f"데이터를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요. ({e})")
            st.stop()

        missing = [c for c in [*INDICATORS, "USD_KRW"] if c not in data["master"].columns]
        if missing:
            st.error(f"Master_Data 시트에 {', '.join(missing)} 열이 없습니다. 데이터 수집 후 다시 시도해 주세요.")
            st.stop()
        master = data["master"][[*INDICATORS, "USD_KRW"]]
        models, fcs, train_info = {}, {}, {}
        for fuel, cfg in FUELS.items():
            X, y = training_xy(master, data[fuel], cfg["inputs"], exclude_covid)
            train_info[fuel] = (len(X), X.index[0], X.index[-1])
            data_key = (f"{fuel}_{'_'.join(X.columns)}_{len(X)}_{X.index[-1]}_{y.sum():.6f}_"
                        f"{X.to_numpy().sum():.6f}_{master.index[-1]}")
            models[fuel] = train_models(f"{fuel}_{'covid_excluded' if exclude_covid else 'all'}", data_key, X, y)
            fcs[fuel] = build_forecast(data_key, master, models[fuel], data[fuel].index[-1] + 1, cfg["inputs"])

    lng_last, lpg_last = data["lng_retail"].index[-1], data["LPG"].index[-1]
    cmp_start = min(lng_last, lpg_last) + 1

    # 표시 단위: 원/MJ 그대로 또는 원/㎥ (도시가스 열량 · LPG 프로판 열량×밀도)
    lpg_mj_per_m3 = LPG_MJ_PER_KG * LPG_KG_PER_M3
    factors = {"LNG": data["heat"], "LPG": lpg_mj_per_m3} if per_m3 else {"LNG": 1.0, "LPG": 1.0}
    unit = "원/㎥" if per_m3 else "원/MJ"
    d_cmp = 1 if per_m3 else 2
    view_data, view_fcs = to_view(data, fcs, factors)
    retail_cost = view_data["retail_cost"]

    # 절감률은 단위와 무관하게 같은 열량(MJ) 기준으로 계산
    saving = compare_frame(data, fcs, cmp_start, MAX_HORIZON)["LNG 절감률"]
    cmp_all = compare_frame(view_data, view_fcs, cmp_start, MAX_HORIZON)
    cmp_all["LNG 절감률"] = saving
    features = feature_frame(master, pd.period_range(cmp_start, periods=MAX_HORIZON, freq="M"))
    cmp_all = pd.concat([cmp_all, features], axis=1)[
        ["월", *features.columns, "LNG", "LPG", "차이", "LNG 절감률", "구분"]
    ]

    st.caption(
        f"실적 기준: LNG **{lng_last.year}년 {lng_last.month}월** · LPG **{lpg_last.year}년 {lpg_last.month}월** · "
        f"LNG는 {describe_inputs(FUELS['LNG']['inputs'])}, LPG는 {describe_inputs(FUELS['LPG']['inputs'])} 시차의 "
        "국제 에너지 가격과 원/달러 환율을 바탕으로 AI 모델 4종의 평균으로 산출한 참고용 전망입니다."
    )

    # ── 핵심 지표: 다음 달 LNG vs LPG
    nxt = cmp_all.iloc[0]
    k1, k2, k3 = st.columns(3)
    k1.metric(
        f"LNG 산업용 ({nxt['월']})", f"{nxt['LNG']:,.{d_cmp}f}",
        delta=f"{nxt['LNG'] - view_data['lng_retail'].iloc[-1]:+,.{d_cmp}f}", delta_color="inverse",
    )
    k2.metric(
        f"LPG ({nxt['월']})", f"{nxt['LPG']:,.{d_cmp}f}",
        delta=f"{nxt['LPG'] - view_data['LPG'].iloc[-1]:+,.{d_cmp}f}", delta_color="inverse",
    )
    k3.metric("LPG 대비 LNG (같은 열량 기준)", f"{nxt['LNG 절감률']:.1f}% 저렴" if nxt["LNG 절감률"] >= 0
              else f"{-nxt['LNG 절감률']:.1f}% 비쌈")
    st.caption(f"단위: {unit} (VAT 별도) · 전월 실적 대비 증감")
    if per_m3:
        st.caption(
            f"㎥ 환산: 도시가스 {data['heat']:.3f} MJ/㎥(최신 열량), "
            f"LPG {lpg_mj_per_m3:.1f} MJ/㎥(프로판 {LPG_MJ_PER_KG} MJ/kg × {LPG_KG_PER_M3:.3f} kg/㎥, 0℃·1기압). "
            f"LPG 1㎥의 열량이 도시가스의 약 {lpg_mj_per_m3 / data['heat']:.1f}배라 ㎥당 가격을 그대로 비교할 수 없으며, "
            "절감률은 같은 열량 기준입니다."
        )

    tab_cmp, tab_lng, tab_lpg, tab_calc = st.tabs(
        ["⚖️ LNG·LPG 비교", "🔵 LNG 도매요금", "🟠 LPG 요금", "🧮 시나리오 계산기"]
    )

    # ── LNG vs LPG 비교
    with tab_cmp:
        horizon = st.radio(
            "전망 기간", [6, 12, 24], index=1, horizontal=True,
            format_func=lambda n: f"{n}개월", key="horizon_cmp",
        )
        view = cmp_all.head(horizon)

        st.plotly_chart(
            plot_compare(view_data, view_fcs, cmp_start + horizon - 1, unit, d_cmp, dual_axis=per_m3), use_container_width=True,
        )

        st.dataframe(
            view,
            hide_index=True,
            use_container_width=True,
            column_config={
                "LNG":        st.column_config.NumberColumn(f"LNG 산업용 ({unit})", format=f"%.{d_cmp}f"),
                "LPG":        st.column_config.NumberColumn(f"LPG ({unit})", format=f"%.{d_cmp}f"),
                "차이":       st.column_config.NumberColumn("차이 (LPG − LNG)", format=f"%+.{d_cmp}f"),
                "LNG 절감률": st.column_config.NumberColumn(
                    "LNG 절감률 (%)", format="%.1f",
                    help="같은 열량 기준으로 LPG 대비 LNG가 저렴한 비율 (음수면 LNG가 더 비쌈)",
                ),
                **feature_column_config(),
            },
        )
        st.caption(
            f"LNG는 대성에너지 산업용 요금 기준입니다(도매요금 전망 + 공급비용 {retail_cost:,.{d_cmp + 2}f} {unit} 고정). "
            f"LPG는 SK가스 가정·상업용 공급가격을 {unit}로 환산한 값입니다. 두 요금 모두 VAT 별도입니다. "
            "JCC·JKM·브렌트유·환율 열은 해당 월 요금에 반영되는 지표 값입니다(괄호 안은 반영 시차)."
        )
        st.download_button(
            "⬇️ 비교표 다운로드 (CSV)",
            view.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"LNG_LPG_비교_{cmp_start}.csv",
            mime="text/csv",
            key="download_cmp",
        )

    # ── 연료별 전망
    with tab_lng:
        render_fuel_tab("LNG", view_data["LNG"], view_fcs["LNG"], "산업용 천연가스 도매요금 (원료비 + 가스공사 공급비용)",
                        unit, 1 if per_m3 else FUELS["LNG"]["digits"])
    with tab_lpg:
        render_fuel_tab("LPG", view_data["LPG"], view_fcs["LPG"], "SK가스 가정·상업용 공급가격 (VAT 별도)",
                        unit, 1 if per_m3 else FUELS["LPG"]["digits"])

    # ── 시나리오 계산기
    with tab_calc:
        st.markdown("국제 에너지 가격과 환율을 직접 입력하면 LNG·LPG 예상 요금을 바로 계산합니다.")
        latest_fx = float(master["USD_KRW"].dropna().iloc[-1])

        with st.form("scenario"):
            cols = st.columns(len(INDICATORS) + 1)
            values = {}
            for col, (c, (name, ind_unit, (lo, hi))) in zip(cols, INDICATORS.items()):
                lags = " · ".join(f"{f} {cfg['inputs'][c]}개월" for f, cfg in FUELS.items() if c in cfg["inputs"])
                values[c] = col.number_input(
                    f"{name} ({ind_unit})", lo, hi, round(float(master[c].dropna().iloc[-1]), 2),
                    step=1.0, format="%.2f",
                    help=f"요금 반영 시차: {lags}. 기본값은 최근 실적입니다.",
                )
            fx_in = cols[-1].number_input(
                "환율 (원/$)", 500.0, 3000.0, float(round(latest_fx)), step=10.0, format="%.0f",
                help="기본값은 최근 실적입니다.",
            )
            submitted = st.form_submit_button("계산하기", type="primary", use_container_width=True)

        if submitted:
            preds = {
                fuel: predict_all(models[fuel], scenario_X(cfg["inputs"], values, fx_in))[0]
                for fuel, cfg in FUELS.items()
            }
            # 모델은 원/MJ로 예측 → 표시 단위로 환산
            lng_whole = preds["LNG"] * factors["LNG"]
            lng_models = lng_whole + retail_cost
            lpg_models = preds["LPG"] * factors["LPG"]
            lng, lpg = lng_models.mean(), lpg_models.mean()

            inputs_text = " · ".join(f"{INDICATORS[c][0]} {v:.2f}" for c, v in values.items())
            st.markdown(f"**{inputs_text} · 환율 ₩{fx_in:,.0f}** 일 때")
            m1, m2 = st.columns(2)
            m1.metric(
                "LNG 산업용", f"{lng:,.{d_cmp}f} {unit}",
                delta=f"최근 실적 대비 {lng - view_data['lng_retail'].iloc[-1]:+,.{d_cmp}f}", delta_color="inverse",
            )
            m2.metric(
                "LPG", f"{lpg:,.{d_cmp}f} {unit}",
                delta=f"최근 실적 대비 {lpg - view_data['LPG'].iloc[-1]:+,.{d_cmp}f}", delta_color="inverse",
            )
            # 절감률은 같은 열량(MJ) 기준
            st.info(f"👉 {saving_text(lng / factors['LNG'], lpg / factors['LPG'])} (같은 열량 기준)")

            with st.expander("모델별 결과 보기"):
                st.dataframe(
                    pd.DataFrame({
                        "모델":                 list(models["LNG"]),
                        f"LNG 도매 ({unit})":   lng_whole,
                        f"LNG 산업용 ({unit})": lng_models,
                        f"LPG ({unit})":        lpg_models,
                    }),
                    hide_index=True, use_container_width=True,
                    column_config={
                        f"LNG 도매 ({unit})":   st.column_config.NumberColumn(format=f"%.{1 if per_m3 else 4}f"),
                        f"LNG 산업용 ({unit})": st.column_config.NumberColumn(format=f"%.{d_cmp}f"),
                        f"LPG ({unit})":        st.column_config.NumberColumn(format=f"%.{d_cmp}f"),
                    },
                )

    st.divider()
    with st.expander("📘 모델 설명"):
        render_model_guide(master, data, train_info, exclude_covid)
    with st.expander("📂 원본 데이터"):
        st.markdown(
            f"- [에너지 지표 시트](https://docs.google.com/spreadsheets/d/{SHEET_ID}) — "
            "`Master_Data`(JCC·브렌트유·JKM·환율), `gas_price`(LNG 도매요금)\n"
            f"- [요금비교 시트]({TARIFF_CSV.split('/export')[0]}) — "
            "LPG 가격, 원료비·공급비용, 산업용 요금"
        )
    st.caption(
        "본 전망은 과거 데이터에 기반한 통계적 추정치로, 실제 요금과 다를 수 있으며 참고용으로만 활용해 주시기 바랍니다. "
        "· 대성에너지 마케팅팀"
    )


main()
