import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="대기오염·호흡기질환 상관 분석 대시보드",
    layout="wide",
)
st.title("📊 지역·연도별 대기오염 물질 배출량과 호흡기 질환 진료자수 상관관계 분석")

# -------------------------------------------------------------
# 1️⃣ 파일 업로드 --------------------------------------------------------
# -------------------------------------------------------------

st.sidebar.header("CSV 파일 업로드")
air_file = st.sidebar.file_uploader("전국_대기오염물질_배출량.csv", type=["csv"], key="air")
resp_file = st.sidebar.file_uploader("지역별_호흡기질환진료인원.csv", type=["csv"], key="resp")

@st.cache_data(show_spinner=False)
def load_csv(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()
    return pd.read_csv(uploaded_file, encoding="cp949")

if not air_file or not resp_file:
    st.info("좌측 사이드바에서 두 CSV 파일을 모두 업로드해 주세요.")
    st.stop()

# -------------------------------------------------------------
# 2️⃣ 데이터 전처리 ------------------------------------------------------
# -------------------------------------------------------------

air_raw = load_csv(air_file)
resp_raw = load_csv(resp_file)

pollutant_row = air_raw.iloc[0]  # 0번째 행: 오염물질 이름
air_df = air_raw.drop(0).reset_index(drop=True)
region_col = air_df.columns[0]   # 보통 "구분(1)" 등 지역명

# ➡️ 연도 추출 (4자리 숫자로 시작하는 컬럼)
all_years = sorted({c[:4] for c in air_df.columns if c[:4].isdigit()})
resp_raw["진료년도"] = resp_raw["진료년도"].str.replace("년", "")
common_years = sorted(set(all_years) & set(resp_raw["진료년도"].unique()))

# ➡️ 수치형 변환 (모든 연도 한번에)
@st.cache_data(show_spinner=False)
def to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df_num = df.copy()
    year_cols = [c for c in df.columns if c[:4].isdigit()]
    df_num[year_cols] = df_num[year_cols].replace({",": ""}, regex=True).astype(float)
    return df_num

air_num = to_numeric(air_df)

# -------------------------------------------------------------
# 3️⃣ 탭 구성 -----------------------------------------------------------
# -------------------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "연도·오염물질 산점도", "연도별 상관 Heatmap", "지역별 상관계수(연도 변화)",
])

# =============================================================
# 📌 TAB 1 ─ 연도·오염물질 산점도
# =============================================================
with tab1:
    st.header("① 선택 연도·오염물질 산점도 및 상관계수")
    col1, col2 = st.columns(2)
    sel_year = col1.selectbox("연도 선택", common_years, index=len(common_years) - 1)

    # 연도별 오염물질 컬럼 & 이름 매핑
    year_cols = [c for c in air_num.columns if c.startswith(sel_year)]
    poll_map = {c: pollutant_row[c] for c in year_cols}
    poll_options = ["전체(모든 물질 합계)"] + list(poll_map.values())
    sel_poll = col2.selectbox("오염물질 선택", poll_options)

    # 🔹 배출량 계산
    sub_df = air_num[~air_num[region_col].isin(["전국", "바다"])]
    if sel_poll == "전체(모든 물질 합계)":
        sub_df["배출량"] = sub_df[year_cols].sum(axis=1)
        x_label = "총 배출량 (kg)"
    else:
        sel_col = next(col for col, name in poll_map.items() if name == sel_poll)
        sub_df["배출량"] = sub_df[sel_col]
        x_label = f"{sel_poll} 배출량 (kg)"

    # 🔹 질환 데이터
    resp_year = resp_raw[resp_raw["진료년도"] == sel_year].copy()
    resp_year["비율"] = resp_year["주민등록인구별 진료실인원 비율"].str.replace("%", "").astype(float)

    merged = pd.merge(
        sub_df[[region_col, "배출량"]],
        resp_year[["시도", "비율"]],
        left_on=region_col,
        right_on="시도",
    )

    if merged.empty:
        st.error("선택한 연도·오염물질에 해당하는 데이터가 없습니다.")
    else:
        r_val = merged["배출량"].corr(merged["비율"])
        st.metric("Pearson r", f"{r_val:.3f}")

        fig = px.scatter(
            merged,
            x="배출량",
            y="비율",
            hover_name="시도",
            labels={"배출량": x_label, "비율": "호흡기 질환 진료자수 비율 (%)"},
            title=f"{sel_year}년 {sel_poll} 배출량 vs 호흡기 질환 비율",
            template="plotly_white",
        )
        # 회귀선
        m, b = np.polyfit(merged["배출량"], merged["비율"], 1)
        x_range = np.linspace(merged["배출량"].min(), merged["배출량"].max(), 100)
        fig.add_scatter(x=x_range, y=m * x_range + b, mode="lines", name="회귀선", line=dict(dash="dash"))

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📄 데이터 보기"):
            st.dataframe(
                merged.rename(columns={region_col: "지역"}).sort_values("비율", ascending=False).reset_index(drop=True),
                use_container_width=True,
            )

# =============================================================
# 📌 TAB 2 ─ 연도별 상관 Heatmap
# =============================================================
with tab2:
    st.header("② 연도별 오염물질-질환 상관계수 Heatmap")

    @st.cache_data(show_spinner=False)
    def compute_year_corr(air_numeric, resp_df, years, region_col, pollutant_row):
        records = []
        for yr in years:
            year_cols = [c for c in air_numeric.columns if c.startswith(yr)]
            poll_map = {c: pollutant_row[c] for c in year_cols}
            sub_air = air_numeric[~air_numeric[region_col].isin(["전국", "바다"])]
            resp_year = resp_df[resp_df["진료년도"] == yr].copy()
            if resp_year.empty:
                continue
            resp_year["비율"] = resp_year["주민등록인구별 진료실인원 비율"].str.replace("%", "").astype(float)

            # total
            merged_tot = pd.merge(
                sub_air[[region_col] + year_cols].assign(총배출=sub_air[year_cols].sum(axis=1))[[region_col, "총배출"]],
                resp_year[["시도", "비율"]],
                left_on=region_col,
                right_on="시도",
            )
            r_total = merged_tot["총배출"].corr(merged_tot["비율"])
            records.append({"Year": yr, "Pollutant": "전체", "r": round(r_total, 3)})

            for col, name in poll_map.items():
                merged = pd.merge(
                    sub_air[[region_col, col]],
                    resp_year[["시도", "비율"]],
                    left_on=region_col,
                    right_on="시도",
                )
                r_val = merged[col].corr(merged["비율"])
                records.append({"Year": yr, "Pollutant": name, "r": round(r_val, 3)})
        df = pd.DataFrame(records)
        return df.pivot(index="Pollutant", columns="Year", values="r")

    corr_pivot = compute_year_corr(air_num, resp_raw, common_years, region_col, pollutant_row)
    st.dataframe(corr_pivot, height=400, use_container_width=True)

    fig_hm = px.imshow(
        corr_pivot,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        origin="lower",
        title="연도별 상관계수 Heatmap (빨강:+ / 파랑:−)",
    )
    st.plotly_chart(fig_hm, use_container_width=True)

# =============================================================
# 📌 TAB 3 ─ 지역별 상관계수 (연도 변화)
# =============================================================
with tab3:
    st.header("③ 지역별 총 배출량·질환 상관계수 (2017–2021)")

    @st.cache_data(show_spinner=False)
    def compute_region_corr(air_numeric, resp_df, years, region_col):
        # 총 배출량 시계열
        total_by_year = {
            yr: air_numeric[[c for c in air_numeric.columns if c.startswith(yr)]].sum(axis=1)
            for yr in years
        }
        total_df = pd.DataFrame(total_by_year)
        total_df[region_col] = air_numeric[region_col]

        # 질환 시계열
        resp_df = resp_df[resp_df["진료년도"].isin(years)].copy()
        resp_df["ratio"] = resp_df["주민등록인구별 진료실인원 비율"].str.replace("%", "").astype(float)
        resp_pivot = resp_df.pivot(index="시도", columns="진료년도", values="ratio")

        records = []
        for region in total_df[region_col]:
            if region in ["전국", "바다"] or region not in resp_pivot.index:
                continue
            emission = total_df[total_df[region_col] == region][years].iloc[0]
            ratio = resp_pivot.loc[region, years]
            combined = pd.concat([emission, ratio], axis=1, keys=["emission", "ratio"]).dropna()
            if len(combined) >= 3:
                r_val = combined["emission"].corr(combined["ratio"])
                records.append({"Region": region, "r": round(r_val, 3)})
        return pd.DataFrame(records).sort_values("Region")

    region_corr_df = compute_region_corr(air_num, resp_raw, common_years, region_col)
    st.dataframe(region_corr_df, height=500, use_container_width=True)

    st.caption("연도별 총 배출량(kg)과 질환 진료자수 비율(%)의 Pearson r — 양(+)·음(−) 상관 여부를 참고하세요.")

