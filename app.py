import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression

# =========================
# Streamlit設定
# =========================
st.set_page_config(page_title="株価・セクター分析", layout="wide")
st.title("株価分析アプリ（セクター比較・セクター内分析）")

# =========================
# 定数
# =========================
PRICE_PERIOD = "6mo"
PRICE_INTERVAL = "1d"

# =========================
# 共通関数
# =========================
def calc_trend_r2(series):
    series = series.dropna()
    if len(series) < 3:
        return np.nan, np.nan

    x = np.arange(len(series)).reshape(-1, 1)
    y = series.values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(x, y)

    return model.coef_[0][0], model.score(x, y)

@st.cache_data(show_spinner=False)
def download_price(code):
    df = yf.download(
        f"{code}.T",
        period=PRICE_PERIOD,
        interval=PRICE_INTERVAL,
        auto_adjust=True,
        progress=False,
        threads=False
    )
    if df.empty:
        return None
    df = df.reset_index()
    df["Code"] = code
    return df

# =========================
# サイドバー：銘柄リスト選択
# =========================
st.sidebar.header("① 銘柄リスト選択")

csv_file = st.sidebar.selectbox(
    "使用する銘柄リスト",
    ["銘柄リスト.csv", "銘柄リスト_test.csv"]
)

code_df = pd.read_csv(csv_file)
code_df["銘柄コード"] = code_df["銘柄コード"].astype(str)

markets = st.sidebar.multiselect(
    "市場区分",
    ["プライム", "スタンダード", "グロース"],
    default=["プライム", "スタンダード", "グロース"]
)

code_df = code_df[code_df["市場区分"].isin(markets)]

sector_filter = st.sidebar.selectbox(
    "セクター（セクター内分析用）",
    ["全体"] + sorted(code_df["セクター"].unique())
)

# =========================
# データ取得ボタン
# =========================
st.header("② 株価データ取得")

price_df = None

if st.button("データ取得を実行"):
    with st.spinner("株価データ取得中..."):
        rows = []

        for _, row in code_df.iterrows():
            df = download_price(row["銘柄コード"])
            if df is None:
                continue

            df["Name"] = row["銘柄名"]
            df["Sector"] = row["セクター"]
            df["Market"] = row["市場区分"]
            rows.append(df)

        if not rows:
            st.error("株価データを取得できませんでした")
            st.stop()

        price_df = pd.concat(rows)
        st.success("株価データ取得が完了しました")

# まだ取得していない場合は停止
if price_df is None:
    st.info("上のボタンからデータ取得を実行してください")
    st.stop()

# =========================
# 前処理
# =========================
price_df = price_df.sort_values(["Code", "Date"])
price_df["ROC"] = price_df.groupby("Code")["Close"].pct_change()

price_df["Volume_MA"] = (
    price_df.groupby("Code")["Volume"]
    .rolling(5).mean()
    .reset_index(level=0, drop=True)
)
price_df["Volume_Rate"] = price_df["Volume"] / price_df["Volume_MA"]

# =========================
# セクター比較（1セクター1行）
# =========================
st.header("③ セクター比較分析")

if st.button("セクター比較分析を実行"):
    results = []

    for sector in price_df["Sector"].unique():
        df_sector = price_df[price_df["Sector"] == sector]
        row = {"セクター": sector}

        for w in [5, 20, 60]:
            tmp = df_sector.groupby("Code").tail(w)

            row[f"{w}日移動平均"] = tmp.groupby("Code")["ROC"].mean().mean()
            slope, r2 = calc_trend_r2(tmp.groupby("Date")["ROC"].mean())
            row[f"{w}日相対強度"] = slope
            row[f"{w}日決定係数"] = r2

        results.append(row)

    st.dataframe(pd.DataFrame(results), use_container_width=True)

# =========================
# セクター内分析（1銘柄1行）
# =========================
st.header("④ セクター内分析")

if sector_filter != "全体":
    df_sector = price_df[price_df["Sector"] == sector_filter]
else:
    df_sector = price_df.copy()

if st.button("セクター内分析を実行"):
    results = []

    for code, g in df_sector.groupby("Code"):
        g = g.sort_values("Date")
        row = {
            "銘柄コード": code,
            "銘柄名": g["Name"].iloc[-1]
        }

        for w in [5, 20, 60]:
            tmp = g.tail(w)
            slope, r2 = calc_trend_r2(tmp["ROC"])

            row[f"{w}日移動平均"] = tmp["ROC"].mean()
            row[f"{w}日相対強度"] = slope
            row[f"{w}日決定係数"] = r2
            row[f"{w}日出来高変化率"] = tmp["Volume_Rate"].iloc[-1]

        results.append(row)

    st.dataframe(pd.DataFrame(results), use_container_width=True)
    st.success("セクター内分析が完了しました")
