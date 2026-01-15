import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# =========================
# 基本設定
# =========================
st.set_page_config(page_title="株価・セクター分析", layout="wide")
st.title("株価分析アプリ（セクター比較・セクター内分析）")

PRICE_PERIOD = "6mo"
PRICE_INTERVAL = "1d"

# =========================
# 関数定義
# =========================
@st.cache_data(show_spinner=False)
def load_price(code):
    ticker = yf.Ticker(f"{code}.T")
    df = ticker.history(period=PRICE_PERIOD, interval=PRICE_INTERVAL)
    if df.empty:
        return None
    df = df.reset_index()
    df["Code"] = code
    return df

def calc_trend_r2(series):
    series = series.dropna()
    if len(series) < 3:
        return np.nan, np.nan

    x = np.arange(len(series)).reshape(-1, 1)
    y = series.values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(x, y)

    slope = model.coef_[0][0]
    r2 = model.score(x, y)

    return slope, r2

# =========================
# 銘柄リスト読み込み
# =========================
st.sidebar.header("① 銘柄リスト設定")

csv_file = st.sidebar.selectbox(
    "使用する銘柄リスト",
    ["銘柄リスト.csv", "銘柄リスト_test.csv"]
)

codes_df = pd.read_csv(csv_file)

codes_df["銘柄コード"] = codes_df["銘柄コード"].astype(str)

# 市場区分フィルタ
market_filters = st.sidebar.multiselect(
    "市場区分",
    options=["プライム", "スタンダード", "グロース"],
    default=["プライム", "スタンダード", "グロース"]
)

codes_df = codes_df[codes_df["市場区分"].isin(market_filters)]

# セクターフィルタ
selected_sector = st.sidebar.selectbox(
    "セクター（セクター内分析用）",
    options=["全体"] + sorted(codes_df["セクター"].unique().tolist())
)

# =========================
# 株価データ取得
# =========================
st.header("② 株価データ取得")

price_data = []

with st.spinner("株価データ取得中..."):
    for _, row in codes_df.iterrows():
        df = load_price(row["銘柄コード"])
        if df is None:
            continue

        df["Name"] = row["銘柄名"]
        df["Sector"] = row["セクター"]
        df["Market"] = row["市場区分"]

        price_data.append(df)

if not price_data:
    st.error("株価データを取得できませんでした")
    st.stop()

price_df = pd.concat(price_data)

# =========================
# 指標計算
# =========================
price_df = price_df.sort_values(["Code", "Date"])

# ROC
price_df["ROC"] = price_df.groupby("Code")["Close"].pct_change()

# 出来高変化率
price_df["Volume_MA"] = (
    price_df.groupby("Code")["Volume"]
    .rolling(5).mean().reset_index(level=0, drop=True)
)
price_df["Volume_Rate"] = price_df["Volume"] / price_df["Volume_MA"]

# =========================
# セクター内分析
# =========================
st.header("③ セクター内分析")

if selected_sector != "全体":
    sector_df = price_df[price_df["Sector"] == selected_sector]

    results = []

    for code in sector_df["Code"].unique():
        df_code = sector_df[sector_df["Code"] == code]

        for window in [5, 20, 60]:
            tmp = df_code.tail(window)

            slope, r2 = calc_trend_r2(tmp["ROC"])

            results.append({
                "銘柄コード": code,
                "銘柄名": tmp["Name"].iloc[-1],
                "期間": f"{window}日",
                "移動平均": tmp["ROC"].mean(),
                "相対強度": slope,
                "決定係数": r2,
                "出来高変化率": tmp["Volume_Rate"].iloc[-1]
            })

    sector_analysis_df = pd.DataFrame(results)
    st.dataframe(sector_analysis_df, use_container_width=True)

# =========================
# セクター比較分析（1セクター1行）
# =========================
st.header("④ セクター比較分析")

if st.button("セクター比較分析を実行"):
    results = []

    for sector in price_df["Sector"].dropna().unique():
        df_sector = price_df[price_df["Sector"] == sector]

        row = {"セクター": sector}

        for window in [5, 20, 60]:
            tmp = df_sector.groupby("Code").tail(window)

            ma = tmp.groupby("Code")["ROC"].mean().mean()
            sector_roc_ts = tmp.groupby("Date")["ROC"].mean()

            slope, r2 = calc_trend_r2(sector_roc_ts)

            row[f"{window}日間移動平均"] = ma
            row[f"{window}日間相対強度"] = slope
            row[f"{window}日間決定係数"] = r2

        results.append(row)

    sector_compare_df = pd.DataFrame(results)
    st.dataframe(sector_compare_df, use_container_width=True)

    st.success("セクター比較分析が完了しました")
