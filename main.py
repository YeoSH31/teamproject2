import streamlit as st
import pandas as pd, numpy as np
import plotly.express as px
from urllib.error import URLError

st.set_page_config(page_title="대기오염·호흡기질환 다중 상관 분석", layout="wide")
st.title("📊 연도·지역·오염물질별 대기오염 배출량과 호흡기 질환 진료자수 상관관계")

# ──────────────────────────────────────────────────────────────
# 1️⃣ GitHub → CSV 자동 로드  (+ 로컬 업로드 백업)              
# ──────────────────────────────────────────────────────────────

st.sidebar.header("데이터 소스 설정")
GH_USER = st.sidebar.text_input("GitHub 사용자/조직명", value="YOUR_GITHUB_ID")
BRANCH  = st.sidebar.text_input("브랜치명", value="main")
REPO    = "teamproject2"  # 고정
RAW_URL = f"https://raw.githubusercontent.com/{GH_USER}/{REPO}/{BRANCH}/"

AIR_CSV  = "전국_대기오염물질_배출량.csv"
RESP_CSV = "지역별_호흡기질환진료인원.csv"

@st.cache_data(show_spinner=False)
def load_csv(url: str) -> pd.DataFrame:
    return pd.read_csv(url, encoding="cp949")

air_raw = resp_raw = pd.DataFrame()
try:
    air_raw  = load_csv(RAW_URL + AIR_CSV)
    resp_raw = load_csv(RAW_URL + RESP_CSV)
    st.sidebar.success("✅ GitHub에서 CSV 로드 완료")
except Exception as e:
    st.sidebar.warning("GitHub 로드 실패 → 파일 업로드 필요")
    air_up  = st.sidebar.file_uploader(AIR_CSV, type="csv", key="air")
    resp_up = st.sidebar.file_uploader(RESP_CSV, type="csv", key="resp")
    if not air_up or not resp_up:
        st.stop()
    air_raw  = pd.read_csv(air_up,  encoding="cp949")
    resp_raw = pd.read_csv(resp_up, encoding="cp949")
    st.sidebar.success("✅ 업로드 파일 로드 완료")

# ──────────────────────────────────────────────────────────────
# 2️⃣ 공통 전처리 -------------------------------------------------------
# ──────────────────────────────────────────────────────────────

poll_row   = air_raw.iloc[0]
air_df     = air_raw.drop(0).reset_index(drop=True)
REGION_COL = air_df.columns[0]

# 연도 교집합
air_years  = sorted({c[:4] for c in air_df.columns if c[:4].isdigit()})
resp_raw["진료년도"] = resp_raw["진료년도"].str.replace("년", "")
YEARS      = sorted(set(air_years) & set(resp_raw["진료년도"].unique()))

@st.cache_data(show_spinner=False)
def to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    num_cols = [c for c in out.columns if c[:4].isdigit()]
    out[num_cols] = out[num_cols].replace({",": ""}, regex=True).astype(float)
    return out

air_num = to_numeric(air_df)
REGIONS = [r for r in air_num[REGION_COL] if r not in ("전국", "바다")]
POLLUTANTS = [poll_row[c] for c in air_num.columns if c.startswith(YEARS[0])]

# ──────────────────────────────────────────────────────────────
# 3️⃣ 사용자 선택 -------------------------------------------------------
# ──────────────────────────────────────────────────────────────

st.sidebar.subheader("분석 파라미터")
SEL_YEAR   = st.sidebar.selectbox("연도", YEARS, index=len(YEARS)-1)
SEL_REGION = st.sidebar.selectbox("지역", REGIONS, index=REGIONS.index("경기도") if "경기도" in REGIONS else 0)
SEL_POLL   = st.sidebar.selectbox("오염물질", ["전체(모든 물질 합계)"] + POLLUTANTS)

# ──────────────────────────────────────────────────────────────
# 4️⃣ 선택 연도·오염물질 산점도 + Heatmap -------------------------------
# ──────────────────────────────────────────────────────────────

year_cols = [c for c in air_num.columns if c.startswith(SEL_YEAR)]
poll_map  = {c: poll_row[c] for c in year_cols}
sub_air   = air_num[[REGION_COL] + year_cols].copy()

if SEL_POLL == "전체(모든 물질 합계)":
    sub_air["배출량"] = sub_air[year_cols].sum(axis=1)
    x_label = "총 배출량 (kg)"
else:
    use_col = next(c for c,n in poll_map.items() if n == SEL_POLL)
    sub_air["배출량"] = sub_air[use_col]
    x_label = f"{SEL_POLL} 배출량 (kg)"

# 질환 데이터 (선택 연도)
resp_year = resp_raw[resp_raw["진료년도"] == SEL_YEAR].copy()
resp_year["비율"] = resp_year["주민등록인구별 진료실인원 비율"].str.replace("%", "").astype(float)

MERGED = pd.merge(sub_air[[REGION_COL,"배출량"]], resp_year[["시도","비율"]], left_on=REGION_COL, right_on="시도")
R_YEAR = MERGED["배출량"].corr(MERGED["비율"])

st.subheader(f"① {SEL_YEAR}년 {SEL_POLL} 배출 ↔ 질환 비율 (r = {R_YEAR:.3f})")
fig = px.scatter(MERGED, x="배출량", y="비율", hover_name="시도",
                 labels={"배출량":x_label, "비율":"호흡기 질환 진료자수 비율 (%)"},
                 template="plotly_white")
# 회귀선
m,b = np.polyfit(MERGED["배출량"], MERGED["비율"], 1)
line_x = np.linspace(MERGED["배출량"].min(), MERGED["배출량"].max(), 100)
fig.add_scatter(x=line_x, y=m*line_x+b, mode="lines", name="회귀선", line=dict(dash="dash"))
# 선택 지역 강조
if SEL_REGION in MERGED["시도"].values:
    pt = MERGED[MERGED["시도"]==SEL_REGION].iloc[0]
    fig.add_scatter(x=[pt["배출량"]], y=[pt["비율"]], mode="markers+text",
                    marker=dict(size=12,color="#ff7f0e"), text=[SEL_REGION], textposition="bottom center", name=SEL_REGION)

st.plotly_chart(fig, use_container_width=True)

# Heatmap
heat_rows = []
for col,name in poll_map.items():
    tmp = pd.merge(air_num[[REGION_COL,col]].rename(columns={col:"val"}), resp_year[["시도","비율"]], left_on=REGION_COL, right_on="시도")
    heat_rows.append({"Pollutant":name, "r":round(tmp["val"].corr(tmp["비율"]),3)})
heat_rows.append({"Pollutant":"전체", "r":round(R_YEAR,3)})
heat_df = pd.DataFrame(heat_rows).set_index("Pollutant")

st.plotly_chart(px.imshow(heat_df, text_auto=True, zmin=-1, zmax=1, color_continuous_scale="RdBu", title=f"{SEL_YEAR}년 오염물질별 상관계수"), use_container_width=True)

# ──────────────────────────────────────────────────────────────
# 5️⃣ 선택 지역·오염물질: 연도별 시계열 상관 -----------------------------
# ──────────────────────────────────────────────────────────────

def yearly_emission(region:str, pollutant:str) -> pd.Series:
    vals = {}
    for yr in YEARS:
        ycols = [c for c in air_num.columns if c.startswith(yr)]
        if pollutant == "전체(모든 물질 합계)":
            vals[yr] = air_num.loc[air_num[REGION_COL]==region, ycols].sum(axis=1).values[0]
        else:
            col = next(c for c in ycols if poll_row[c] == pollutant)
            vals[yr] = air_num.loc[air_num[REGION_COL]==region, col].values[0]
    return pd.Series(vals)

emis_series  = yearly_emission(SEL_REGION, SEL_POLL)
ratio_series = resp_raw[resp_raw["시도"] == SEL_REGION].set_index("진료년도")["주민등록인구별 진료실인원 비율"].str.replace("%", "").astype(float)

ts = pd.DataFrame({"Emission":emis_series, "Ratio":ratio_series}).dropna()

st.subheader(f"② {SEL_REGION} — {SEL_POLL} 연도별 배출 ↔ 질환 비율")
if len(ts) >= 3:
    r_ts = ts["Emission"].corr(ts["Ratio"])
    st.write(f"연도별 Pearson r = **{r_ts:.3f}**")
    st.plotly_chart(px.scatter(ts, x="Emission", y="Ratio", text=ts.index,
                               labels={"Emission":x_label, "Ratio":"질환 비율(%)"},
                               template="plotly_white", title="연도별 상관 산점도"), use_container_width=True)
else:
    st.warning("연도별 상관을 계산하기에 데이터가 부족합니다. (3년 이상 필요)")

st.caption("※ 한글이 깨질 경우 서버에 NanumGothic 등 한글 폰트를 설치하고 fig.update_layout(font_family='NanumGothic') 한 줄을 추가하십시오.")
