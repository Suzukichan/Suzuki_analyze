import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="株式セクター分析", layout="wide")

PERIOD = "6mo"
INTERVAL = "1d"
WINDOWS = [5, 20, 60]

# =========================
# 銘柄リスト読込
# =========================
@st.cache_data
def load_stock_list(csv_path):
    df = pd.read_csv(csv_path)

    df = df.rename(columns={
        "銘柄コード": "Code",
        "銘柄名": "Name",
        "セクター": "Sector"
    })

    required = {"Code", "Name", "Sector"}
    if not required.issubset(df.columns):
        raise ValueError("銘柄リストCSVの列構成が不正です")

    df["Code"] = df["Code"].astype(str)
    return df

# =========================
# 株価取得
# =========================
def fetch_price_data(codes):
    dfs = []
    for code in codes:
        try:
            df = yf.download(
                f"{code}.T",
                period=PERIOD,
                interval=INTERVAL,
                progress=False
            )
            if df.empty:
                continue
            df = df.reset_index()
            df["Code"] = code
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)

# =========================
# 指標計算
# =========================
def calculate_indicators(price_df):
    price_df = price_df.copy()
    price_df["Close"] = pd.to_numeric(price_df["Close"], errors="coerce")

    price_df["ROC"] = (
        price_df
        .groupby("Code")["Close"]
        .transform(lambda x: x.pct_change())
    )

    rows = []

    for window in WINDOWS:
        price_df[f"MA_{window}"] = (
            price_df
            .groupby("Code")["Close"]
            .transform(lambda x: x.rolling(window).mean())
        )

        for code, g in price_df.groupby("Code"):
            g = g.dropna(subset=[f"MA_{window}"])
            if len(g) < window:
                continue

            x = np.arange(len(g)).reshape(-1, 1)
            y = g["Close"].values

            model = LinearRegression()
            model.fit(x, y)

            rows.append({
                "Code": code,
                "Window": window,
                "MA": g[f"MA_{window}"].iloc[-1],
                "RS": g["ROC"].iloc[-window:].mean(),
                "R2": model.score(x, y)
            })

    return pd.DataFrame(rows)

# =========================
# セクター比較
# =========================
def sector_comparison(indicator_df, stock_df):
    merged = indicator_df.merge(stock_df, on="Code")
    rows = []

    for sector, g in merged.groupby("Sector"):
        row = {"Sector": sector}
        for w in WINDOWS:
            sub = g[g["Window"] == w]
            row[f"{w}日_MA"] = sub["MA"].mean()
            row[f"{w}日_RS"] = sub["RS"].mean()
            row[f"{w}日_R2"] = sub["R2"].mean()
        rows.append(row)

    return pd.DataFrame(rows)

# =========================
# セクター内分析
# =========================
def sector_detail(indicator_df, stock_df, sector):
    merged = indicator_df.merge(stock_df, on="Code")
    merged = merged[merged["Sector"] == sector]

    rows = []
    for code, g in merged.groupby("Code"):
        row = {
            "Code": code,
            "Name": g["Name"].iloc[0]
        }
        for w in WINDOWS:
            sub = g[g["Window"] == w]
            if sub.empty:
                continue
            row[f"{w}日_MA"] = sub["MA"].values[0]
            row[f"{w}日_RS"] = sub["RS"].values[0]
            row[f"{w}日_R2"] = sub["R2"].values[0]
        rows.append(row)

    return pd.DataFrame(rows)

# =========================
# UI
# =========================
st.title("株式セクター分析")

csv_file = st.selectbox(
    "銘柄リスト選択",
    ["銘柄リスト_test.csv", "銘柄リスト.csv"]
)

if st.button("▶ 実行"):
    stock_df = load_stock_list(csv_file)

    st.info("株価取得中…")
    price_df = fetch_price_data(stock_df["Code"].unique())

    if price_df.empty:
        st.error("株価データを取得できませんでした（RateLimitの可能性）")
        st.stop()

    indicator_df = calculate_indicators(price_df)

    st.subheader("セクター比較")
    sector_df = sector_comparison(indicator_df, stock_df)

    if sector_df.empty:
        st.error("セクター比較結果が空です")
        st.stop()

    st.dataframe(sector_df, use_container_width=True)

    sector_list = sector_df["Sector"].dropna().tolist()
    sector = st.selectbox("セクター選択", sector_list)

    st.subheader("セクター内分析")
    detail_df = sector_detail(indicator_df, stock_df, sector)
    st.dataframe(detail_df, use_container_width=True)
