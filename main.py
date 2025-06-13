import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ────────────────────────────────────────────────
# 설정
# ────────────────────────────────────────────────
st.set_page_config(page_title="대기오염·호흡기질환 상관 분석", layout="wide")
st.title("📊 연도·지역·오염물질별 대기오염 배출량 vs 호흡기 질환 진료자수")

# ────────────────────────────────────────────────
# 1. GitHub RAW 경로 + 로드 (백업: 업로드)
# ────────────────────────────────────────────────

st.sidebar.header("데이터 소스 설정")
GH_USER = st.sidebar.text_input("GitHub 사용자/조직명", value="YeoSH31")
BRANCH  = st.sidebar.text_input("브랜치명", value="main")
REPO    = "teamproject2"
RAW_ROOT = f"https://raw.githubusercontent.com/{GH_USER}/{REPO}/{BRANCH}/"

# 파일명(URL 인코딩 포함 그대로)
AIR_CSV  = "%EC%A0%84%EA%B5%AD_%EB%8C%80%EA%B8%B0%EC%98%A4%EC%97%BC%EB%AC%BC%EC%A7%88_%EB%B0%B0%EC%B6%9C%EB%9F%89.csv"
RESP_CSV = "%EC%A7%80%EC%97%AD%EB%B3%84_%ED%98%B8%ED%9D%A1%EA%B8%B0%EC%A7%88%ED%99%98%EC%A7%84%EB%A3%8C%EC%9D%B8%EC%9B%90.csv"

@st.cache_data(show_spinner=False)
def load_csv(url: str) -> pd.DataFrame:
    return pd.read_csv(url, encoding="cp949")

try:
    air_raw  = load_csv(RAW_ROOT + AIR_CSV)
    resp_raw = load_csv(RAW_ROOT + RESP_CSV)
    st.sidebar.success("✅ GitHub CSV 로드 완료")
except Exception as e:
    st.sidebar.warning("GitHub 로드 실패 → 파일 업로드 필요")
    air_file  = st.sidebar.file_uploader("전국_대기오염물질_배출량.csv", type="csv")
    resp_file = st.sidebar.file_uploader("지역별_호흡기질환진료인원.csv", type="csv")
    if not air_file or not resp_file:
        st.stop()
    air_raw  = pd.read_csv(air_file,  encoding="cp949")
    resp_raw = pd.read_csv(resp_file, encoding="cp949")
    st.sidebar.success("✅ 업로드 파일 로드 완료")

# ────────────────────────────────────────────────
# 2. 전처리
# ────────────────────────────────────────────────

poll_row   = air_raw.iloc[0]               # 0행: 오염물질 이름
air_df     = air_raw.drop(0).reset_index(drop=True)
REGION_COL = air_df.columns[0]             # 지역명 컬럼

# 숫자형 변환 (콤마 제거)
num_cols = [c for c in air_df.columns if c[:4].isdigit()]
air_df[num_cols] = air_df[num_cols].replace({",": ""}, regex=True).astype(float)

# 연도 교집합
resp_raw["진료년도"] = resp_raw["진료년도"].str.replace("년", "")
YEARS = sorted(set(c[:4] for c in num_cols) & set(resp_raw["진료년도"].unique()))

# 지역·오염물질 리스트
REGIONS = [r for r in air_df[REGION_COL] if r not in ("전국", "바다")]
POLLUTANTS = [poll_row[c] for c in air_df.columns if c.startswith(YEARS[0])]

# ────────────────────────────────────────────────
# 3. 사용자 파라미터
# ────────────────────────────────────────────────
st.sidebar.header("분석 파라미터")
SEL_YEAR   = st.sidebar.selectbox("연도", YEARS, index=len(YEARS)-1)
SEL_REGION = st.sidebar.selectbox("지역", REGIONS, index=REGIONS.index("경기도") if "경기도" in REGIONS else 0)
SEL_POLL   = st.sidebar.selectbox("오염물질", ["전체(모든 물질 합계)"] + POLLUTANTS)

# ────────────────────────────────────────────────
# 4. 선택 연도·오염물질 산점도 + Heatmap
# ────────────────────────────────────────────────

y_cols   = [c for c in air_df.columns if c.startswith(SEL_YEAR)]
poll_map = {c: poll_row[c] for c in y_cols}

air_sel = air_df[[REGION_COL] + y_cols].copy()
if SEL_POLL == "전체(모든 물질 합계)":
    air_sel["배출량"] = air_sel[y_cols].sum(axis=1)
    x_label = "총 배출량 (kg)"
else:
    use_col = next(c for c,n in poll_map.items() if n == SEL_POLL)
    air_sel["배출량"] = air_sel[use_col]
    x_label = f"{SEL_POLL} 배출량 (kg)"

resp_year = resp_raw[resp_raw["진료년도"] == SEL_YEAR].copy()
resp_year["비율"] = resp_year["주민등록인구별 진료실인원 비율"].str.replace("%", "").astype(float)

merged = pd.merge(air_sel[[REGION_COL,"배출량"]], resp_year[["시도","비율"]], left_on=REGION_COL, right_on="시도")
R_YEAR = merged["배출량"].corr(merged["비율"])

st.subheader(f"① {SEL_YEAR}년 {SEL_POLL} 배출 ↔ 질환 비율  (r = {R_YEAR:.3f})")
fig1 = px.scatter(
    merged, x="배출량", y="비율", hover_name="시도",
    labels={"배출량":x_label, "비율":"호흡기 질환 비율(%)"},
    template="plotly_white")
# 회귀선
m,b = np.polyfit(merged["배출량"], merged["비율"], 1)
line_x = np.linspace(merged["배출량"].min(), merged["배출량"].max(), 100)
fig1.add_scatter(x=line_x, y=m*line_x+b, mode="lines", name="회귀선", line=dict(dash="dash"))
# 선택 지역 강조
if SEL_REGION in merged["시도"].values:
    pt = merged[merged["시도"] == SEL_REGION].iloc[0]
    fig1.add_scatter(x=[pt["배출량"]], y=[pt["비율"]], mode="markers+text", marker=dict(size=12,color="#ff7f0e"), text=[SEL_REGION], textposition="bottom center", name=SEL_REGION)

st.plotly_chart(fig1, use_container_width=True)

# Heatmap 데이터
heat_rows = []
for col,name in poll_map.items():
    tmp = pd.merge(air_df[[REGION_COL,col]].rename(columns={col:"val"}), resp_year[["시도","비율"]], left_on=REGION_COL, right_on="시도")
    heat_rows.append({"오염물질":name, "r":round(tmp["val"].corr(tmp["비율"]),3)})
heat_rows.append({"오염물질":"전체", "r":round(R_YEAR,3)})
heat_df = pd.DataFrame(heat_rows).set_index("오염물질")

fig2 = px.imshow(heat_df, text_auto=True, zmin=-1, zmax=1, color_continuous_scale="RdBu", title=f"{SEL_YEAR}년 오염물질별 상관계수")
st.plotly_chart(fig2, use_container_width=True)

# ────────────────────────────────────────────────
# 5. 선택 지역·오염물질: 연도별 시계열 상관
# ────────────────────────────────────────────────

def yearly_emission(region: str, pollutant: str) -> pd.Series:
    values = {}
    for yr in YEARS:
        yr_cols = [c for c in air_df.columns if c.startswith(yr)]
        if pollutant == "전체(모든 물질 합계)":
            values[yr] = air_df.loc[air_df[REGION_COL]==region, yr_cols].sum(axis=1).values[0]
        else:
            col = next(c for c in yr_cols if poll_row[c] == pollutant)
            values[yr] = air_df.loc[air_df[REGION_COL]==region, col].values[0]
    return pd.Series(values)

emis_series  = yearly_emission(SEL_REGION, SEL_POLL)
ratio_series = resp_raw[resp_raw["시도"] == SEL_REGION].set_index("진료년도")["주민등록인구별 진료실인원 비율"].str.replace("%", "").astype(float)

ts = pd.DataFrame({"Emission":emis_series, "Ratio":ratio_series}).dropna()

st.subheader(f"② {SEL_REGION} – {SEL_POLL} 연도별 배출 ↔ 질환 비율")
if len(ts) >= 3:
    r_ts = ts["Emission"].corr(ts["Ratio"])
    st.write(f"연도별 Pearson r = **{r_ts:.3f}** (n={len(ts)})")
    fig3 = px.scatter(ts, x="Emission", y="Ratio", text=ts.index, labels={"Emission":x_label, "Ratio":"질환 비율(%)"}, template="plotly_white", title="연도별 산점도")
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.warning("연도별 상관을 계산하기에 데이터가 부족합니다 (3년 이상 필요)")

# ────────────────────────────────────────────────
# 끝
# ────────────────────────────────────────────────
