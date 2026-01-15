import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="株式セクター分析", layout="wide")
st.title("株式セクター分析")

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

    df["Code"] = df["Code"].astype(str)
    return df[["Code", "Name", "Sector"]]

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

    if len(dfs) == 0:
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

    for code, g in price_df.groupby("Code"):
        g = g.dropna()

        for w in WINDOWS:
            if len(g) < w:
                continue

            ma = g["Close"].rolling(w).mean().iloc[-1]
            rs = g["ROC"].tail(w).mean()

            x = np.arange(w).reshape(-1, 1)
            y = g["Close"].tail(w).values

            model = LinearRegression().fit(x, y)

            rows.append({
                "Code": code,
                "Window": w,
                "MA": ma,
                "RS": rs,
                "R2": model.score(x, y)
            })

    return pd.DataFrame(rows)

# =========================
# セクター比較
# =========================
def sector_comparison(ind_df, stock_df):
    merged = ind_df.merge(stock_df, on="Code", how="inner")

    result = []

    for sector, g in merged.groupby("Sector"):
        row = {"Sector": sector}
        for w in WINDOWS:
            sub = g[g["Window"] == w]
            row[f"{w}日_MA"] = sub["MA"].mean()
            row[f"{w}日_RS"] = sub["RS"].mean()
            row[f"{w}日_R2"] = sub["R2"].mean()
        result.append(row)

    return pd.DataFrame(result)

# =========================
# セクター内分析
# =========================
def sector_detail(ind_df, stock_df, sector):
    merged = ind_df.merge(stock_df, on="Code", how="inner")
    merged = merged[merged["Sector"] == sector]

    rows = []

    for code, g in merged.groupby("Code"):
        row = {
            "Code": code,
            "Name": g["Name"].iloc[0]
        }
        for w in WINDOWS:
            sub = g[g["Window"] == w]
            if not sub.empty:
                row[f"{w}日_MA"] = sub["MA"].iloc[0]
                row[f"{w}日_RS"] = sub["RS"].iloc[0]
                row[f"{w}日_R2"] = sub["R2"].iloc[0]
        rows.append(row)

    return pd.DataFrame(rows)

# =========================
# UI
# =========================
csv_file = st.selectbox(
    "銘柄リスト選択",
    ["銘柄リスト_test.csv", "銘柄リスト.csv"]
)

if st.button("▶ 実行"):
    stock_df = load_stock_list(csv_file)

    st.info("株価データ取得中…")
    price_df = fetch_price_data(stock_df["Code"].tolist())

    if price_df.empty:
        st.error("株価データを取得できませんでした")
        st.stop()

    indicator_df = calculate_indicators(price_df)

    st.subheader("セクター比較")
    sector_df = sector_comparison(indicator_df, stock_df)

    if sector_df.empty:
        st.error("セクター比較結果が空です")
        st.stop()

    st.dataframe(sector_df, use_container_width=True)

    # 🔒 selectbox 防御
    sector_list = sector_df["Sector"].dropna().astype(str).tolist()

    if len(sector_list) == 0:
        st.error("選択可能なセクターがありません")
        st.stop()

    sector = st.selectbox("セクター選択", sector_list)

    st.subheader("セクター内分析")
    detail_df = sector_detail(indicator_df, stock_df, sector)
    st.dataframe(detail_df, use_container_width=True)
