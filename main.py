import streamlit as st
import pandas as pd, numpy as np
import plotly.express as px
from urllib.error import URLError

st.set_page_config(page_title="대기오염·호흡기질환 다중 상관 분석", layout="wide")
st.title("📊 연도·지역·오염물질별 대기오염 배출량과 호흡기 질환 진료자수 상관관계")

# ──────────────────────────────────────────────────────────────
# 1️⃣ GitHub → CSV 자동 로드  (+ 업로드 백업)  
#    • 기본 경로: https://raw.githubusercontent.com/<USER>/teamproject2/<branch>/  
#    • 로드 실패 시 사이드바 업로드 창 노출
# ──────────────────────────────────────────────────────────────

st.sidebar.header("데이터 소스 설정")
GH_USER  = st.sidebar.text_input("GitHub 사용자/조직명", value="YOUR_GITHUB_ID")
BRANCH   = st.sidebar.text_input("브랜치명", value="main")
REPO     = "teamproject2"  # 고정
RAW_ROOT = f"https://raw.githubusercontent.com/{GH_USER}/{REPO}/{BRANCH}/"

AIR_CSV  = "전국_대기오염물질_배출량.csv"
RESP_CSV = "지역별_호흡기질환진료인원.csv"

@st.cache_data(show_spinner=False)
def read_github_csv(raw_root: str, fname: str) -> pd.DataFrame:
    url = raw_root + fname
    return pd.read_csv(url, encoding="cp949")

# ➡️ GitHub 로드 시도
try:
    air_raw  = read_github_csv(RAW_ROOT, AIR_CSV)
    resp_raw = read_github_csv(RAW_ROOT, RESP_CSV)
    data_src = "github"
except Exception as e:
    st.warning(f"GitHub 데이터 로드 실패 → 직접 업로드 해 주세요.\n{e}")
    # 업로드 백업
    air_file  = st.sidebar.file_uploader(AIR_CSV, type="csv", key="air")
    resp_file = st.sidebar.file_uploader(RESP_CSV, type="csv", key="resp")
    if not air_file or not resp_file:
        st.stop()
    air_raw  = pd.read_csv(air_file,  encoding="cp949")
    resp_raw = pd.read_csv(resp_file, encoding="cp949")
    data_src = "upload"

st.sidebar.success(f"✅ 데이터 소스: {data_src}")

# ──────────────────────────────────────────────────────────────
# 2️⃣ 공통 전처리 -------------------------------------------------------
# ──────────────────────────────────────────────────────────────

poll_row   = air_raw.iloc[0]
air_df     = air_raw.drop(0).reset_index(drop=True)
REGION_COL = air_df.columns[0]

# 연도 목록 (air & resp 공통)
air_years  = sorted({c[:4] for c in air_df.columns if c[:4].isdigit()})
resp_raw["진료년도"] = resp_raw["진료년도"].str.replace("년", "")
YEARS      = sorted(set(air_years) & set(resp_raw["진료년도"].unique()))

@st.cache_data(show_spinner=False)
def numeric_air(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = [c for c in out.columns if c[:4].isdigit()]
    out[cols] = out[cols].replace({",": ""}, regex=True).astype(float)
    return out

air_num = numeric_air(air_df)
REGIONS = [r for r in air_df[REGION_COL] if r not in ("전국", "바다")]
# 모든 오염물질 리스트 (첫 연도 기준)
sample_cols = [c for c in air_num.columns if c.startswith(YEARS[0])]
POLLUTANTS  = [poll_row[c] for c in sample_cols]

# ──────────────────────────────────────────────────────────────
# 3️⃣ 사용자 파라미터 선택 ---------------------------------------------
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

st.subheader(f"① {SEL_YEAR}년 {SEL_POLL} 배출량 ↔ 질환 비율 (r = {R_YEAR:.3f})")
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
                    marker=dict(size=12,color="#ff7f0e"),
                    text=[SEL_REGION], textposition="bottom center", name=SEL_REGION)

st.plotly_chart(fig, use_container_width=True)

# Heatmap (선택 연도 모든 오염물질)
heat_rows = []
for col,name in poll_map.items():
    tmp = pd.merge(air_num[[REGION_COL,col]].rename(columns={col:"val"}), resp_year[["시도","비율"]], left_on=REGION_COL, right_on="시도")
    heat_rows.append({"Pollutant":name, "r":round(tmp["val"].corr(tmp["비율"]),3)})
heat_rows.append({"Pollutant":"전체", "r":round(R_YEAR,3)})
heat_df = pd.DataFrame(heat_rows).set_index("Pollutant")

st.plotly_chart(px.imshow(heat_df, text_auto=True, zmin=-1, zmax=1, color_continuous_scale="RdBu", title=f"{SEL_YEAR}년 오염물질별 상관계수"), use_container_width=True)

# ──────────────────────────────────────────────────────────────
# 5️⃣ 선택 지역·오염물질 연도별 시계열 산점도 ----------------------------
# ──────────────────────────────────────────────────────────────

def yearly_emission(region, poll):
    vals = {}
    for yr in YEARS:
        ycols = [c for c in air_num.columns if c.startswith(yr)]
        if poll == "전체(모든 물질 합계)":
            vals[yr] = air_num.loc[air_num[REGION_COL]==region, ycols].sum(axis=1).values[0]
        else:
            col = next(c for c in ycols if poll_row[c]==poll)
            vals[yr] = air_num.loc[air_num[REGION_COL]==region, col].values[0]
    return pd.Series(vals)

emis_series = yearly_emission(SEL_REGION, SEL_POLL)
ratio_series = resp_raw[resp_raw["시도"]==SEL_REGION].set_index("진료년도")["주민등록인구별 진료실인원 비율"].str.replace("%","\
