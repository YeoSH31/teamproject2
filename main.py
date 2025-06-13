import streamlit as st
import pandas as pd, numpy as np
import plotly.express as px

st.set_page_config(page_title="대기오염·호흡기질환 다중 상관 분석", layout="wide")
st.title("📊 연도·지역·오염물질별 대기오염 배출량과 호흡기 질환 진료자수 상관관계")

# ──────────────────────── 1. 입력 ────────────────────────
@st.cache_data(show_spinner=False)
def load_csv(src):
    if src is None:
        return pd.DataFrame()
    return pd.read_csv(src, encoding="cp949")

st.sidebar.header("CSV 입력")
air_path  = st.sidebar.text_input("전국_대기오염물질_배출량.csv 경로(또는 업로드)")
resp_path = st.sidebar.text_input("지역별_호흡기질환진료인원.csv 경로(또는 업로드)")

# 로컬 업로드 대체
if not air_path or not air_path.startswith("http"):
    air_path  = st.sidebar.file_uploader("전국_대기오염물질_배출량.csv", type="csv", key="air")
if not resp_path or not resp_path.startswith("http"):
    resp_path = st.sidebar.file_uploader("지역별_호흡기질환진료인원.csv", type="csv", key="resp")

air_raw  = load_csv(air_path)
resp_raw = load_csv(resp_path)
if air_raw.empty or resp_raw.empty:
    st.info("좌측 입력을 완료해 주세요."); st.stop()

# ──────────────────────── 2. 공통 전처리 ────────────────────────
poll_row   = air_raw.iloc[0]
air_df     = air_raw.drop(0).reset_index(drop=True)
region_col = air_df.columns[0]

air_years  = sorted({c[:4] for c in air_df.columns if c[:4].isdigit()})
resp_raw["진료년도"] = resp_raw["진료년도"].str.replace("년", "")
years      = sorted(set(air_years) & set(resp_raw["진료년도"].unique()))

@st.cache_data(show_spinner=False)
def make_numeric(df):
    out = df.copy()
    cols = [c for c in out.columns if c[:4].isdigit()]
    out[cols] = out[cols].replace({",": ""}, regex=True).astype(float)
    return out
air_num = make_numeric(air_df)

regions = [r for r in air_df[region_col] if r not in ("전국", "바다")]
sample_year_cols = [c for c in air_num.columns if c.startswith(years[0])]
pollutants = [poll_row[c] for c in sample_year_cols]

# ──────────────────────── 3. 사용자 선택 ────────────────────────
st.sidebar.subheader("분석 파라미터")
sel_year   = st.sidebar.selectbox("연도", years, index=len(years)-1)
sel_region = st.sidebar.selectbox("지역", regions, index=regions.index("경기도") if "경기도" in regions else 0)
sel_poll   = st.sidebar.selectbox("오염물질", ["전체(모든 물질 합계)"] + pollutants)

# ──────────────────────── 4. (연도·오염물질) 산점도 ────────────────────────
year_cols = [c for c in air_num.columns if c.startswith(sel_year)]
poll_map  = {c: poll_row[c] for c in year_cols}
sub_air   = air_num[[region_col] + year_cols].copy()

if sel_poll == "전체(모든 물질 합계)":
    sub_air["배출량"] = sub_air[year_cols].sum(axis=1)
    x_label = "총 배출량 (kg)"
else:
    use_col = next(c for c, n in poll_map.items() if n == sel_poll)
    sub_air["배출량"] = sub_air[use_col]
    x_label = f"{sel_poll} 배출량 (kg)"

resp_year = resp_raw[resp_raw["진료년도"] == sel_year].copy()
resp_year["비율"] = resp_year["주민등록인구별 진료실인원 비율"].str.replace("%", "").astype(float)

merged = pd.merge(sub_air[[region_col,"배출량"]],
                  resp_year[["시도","비율"]],
                  left_on=region_col, right_on="시도")

r_year = merged["배출량"].corr(merged["비율"])
st.subheader(f"① {sel_year}년 {sel_poll} 배출량 ↔ 질환 비율 (r = {r_year:.3f})")

fig = px.scatter(
    merged, x="배출량", y="비율", hover_name="시도",
    labels={"배출량":x_label, "비율":"호흡기 질환 진료자수 비율 (%)"},
    template="plotly_white",
)
m,b = np.polyfit(merged["배출량"], merged["비율"], 1)
fig.add_scatter(x=np.linspace(merged["배출량"].min(), merged["배출량"].max(),100),
                y=m*np.linspace(merged["배출량"].min(), merged["배출량"].max(),100)+b,
                mode="lines", name="회귀선", line=dict(dash="dash"))
# 선택 지역 강조
if sel_region in merged["시도"].values:
    row = merged[merged["시도"]==sel_region].iloc[0]
    fig.add_scatter(x=[row["배출량"]], y=[row["비율"]], mode="markers+text",
                    marker=dict(size=12,color="#ff7f0e"),
                    text=[sel_region], textposition="bottom center",
                    name=sel_region)
st.plotly_chart(fig, use_container_width=True)

# ───── Heatmap : 해당 연도 모든 오염물질 vs 질환 상관 ─────
heat = []
for col,name in poll_map.items():
    tmp = pd.merge(air_num[[region_col,col]].rename(columns={col:"val"}),
                   resp_year[["시도","비율"]],
                   left_on=region_col, right_on="시도")
    heat.append({"Pollutant":name, "r":round(tmp["val"].corr(tmp["비율"]),3)})
heat.append({"Pollutant":"전체", "r":round(r_year,3)})
heat_df = pd.DataFrame(heat).set_index("Pollutant")

st.plotly_chart(
    px.imshow(heat_df, text_auto=True, zmin=-1, zmax=1,
              color_continuous_scale="RdBu",
              title=f"{sel_year}년 오염물질별 상관계수 Heatmap"),
    use_container_width=True,
)

# ──────────────────────── 5. 선택 지역·오염물질 연도별 시계열 ────────────────────────
st.subheader(f"② {sel_region}의 연도별 {sel_poll} 배출량 ↔ 질환 비율")

def yearly_emission(region, pollutant):
    data = {}
    for yr in years:
        yr_cols = [c for c in air_num.columns if c.startswith(yr)]
        if pollutant == "전체(모든 물질 합계)":
            data[yr] = air_num.loc[air_num[region_col]==region, yr_cols].sum(axis=1).values[0]
        else:
            col = next(c for c in yr_cols if poll_row[c] == pollutant)
            data[yr] = air_num.loc[air_num[region_col]==region, col].values[0]
    return pd.Series(data)

emis = yearly_emission(sel_region, sel_poll)
ratio = resp_raw[resp_raw["시도"]==sel_region].set_index("진료년도")["주민등록인구별 진료실인원 비율"].str.replace("%","").astype(float)
ts = pd.DataFrame({"Emission":emis, "Ratio":ratio}).dropna()

if ts.shape[0] >= 3:
    r_ts = ts["Emission"].corr(ts["Ratio"])
    st.write(f"연도별 Pearson r = **{r_ts:.3f}**")
    st.plotly_chart(
        px.scatter(ts, x="Emission", y="Ratio", text=ts.index,
                   labels={"Emission":x_label, "Ratio":"질환 비율(%)"},
                   template="plotly_white",
                   title=f"{sel_region} {sel_poll} 배출 ↔ 질환 비율 (연도별)"),
        use_container_width=True,
    )
else:
    st.warning("연도별 상관계수를 계산하기에 데이터가 부족합니다.")

# ─────────────────────────────
st.caption("※ 글꼴이 깨질 경우 서버(또는 로컬)에 NanumGothic 등 한글 폰트를 설치한 뒤 "
           "fig.update_layout(font_family='NanumGothic') 한 줄을 추가해 주세요.")
