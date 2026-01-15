import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression

# =========================
# 基本設定
# =========================
st.set_page_config(page_title="株式セクター分析", layout="wide")

PERIOD = "6mo"
INTERVAL = "1d"
WINDOWS = [5, 20, 60]

# =========================
# データ読み込み
# =========================
@st.cache_data
def load_stock_list(csv_path):
    return pd.read_csv(csv_path)

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

    # ROC（ValueError 回避）
    price_df["ROC"] = (
        price_df
        .groupby("Code")["Close"]
        .transform(lambda x: x.pct_change())
    )

    results = []

    for window in WINDOWS:
        ma_col = f"MA_{window}"
        price_df[ma_col] = (
            price_df
            .groupby("Code")["Close"]
            .transform(lambda x: x.rolling(window).mean())
        )

        for code, g in price_df.groupby("Code"):
            g = g.dropna(subset=[ma_col])
            if len(g) < window:
                continue

            y = g["Close"].values.reshape(-1, 1)
            x = np.arange(len(y)).reshape(-1, 1)

            model = LinearRegression()
            model.fit(x, y)
            r2 = model.score(x, y)

            results.append({
                "Code": code,
                "Window": window,
                "MA": g[ma_col].iloc[-1],
                "RS": g["ROC"].iloc[-window:].mean(),
                "R2": r2
            })

    return pd.DataFrame(results)

# =========================
# セクター比較（1セクター1行）
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
# セクター内分析（1銘柄1行）
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
            row[f"{w}日_MA"] = sub["MA"].values[0]
            row[f"{w}日_RS"] = sub["RS"].values[0]
            row[f"{w}日_R2"] = sub["R2"].values[0]
        rows.append(row)

    return pd.DataFrame(rows)

# =========================
# UI
# =========================
st.title("📊 株式セクター分析アプリ")

csv_file = st.selectbox(
    "銘柄リストを選択",
    ["銘柄リスト_test.csv", "銘柄リスト.csv"]
)

if st.button("▶ データ取得・分析実行"):
    stock_df = load_stock_list(csv_file)

    st.info("株価データ取得中...")
    price_df = fetch_price_data(stock_df["Code"].unique())

    if price_df.empty:
        st.error("株価データを取得できませんでした")
        st.stop()

    st.info("指標計算中...")
    indicator_df = calculate_indicators(price_df)

    st.success("分析完了")

    # セクター比較
    st.subheader("📈 セクター比較")
    sector_df = sector_comparison(indicator_df, stock_df)
    st.dataframe(sector_df, use_container_width=True)

    # セクター内分析
    st.subheader("🔍 セクター内分析")
    selected_sector = st.selectbox(
        "セクターを選択",
        sector_df["Sector"].unique()
    )

    detail_df = sector_detail(indicator_df, stock_df, selected_sector)
    st.dataframe(detail_df, use_container_width=True)
