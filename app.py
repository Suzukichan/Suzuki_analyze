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
# 銘柄リスト読込（堅牢版）
# =========================
@st.cache_data
def load_stock_list(csv_path):
    df = pd.read_csv(csv_path)

    # 列名正規化
    rename_map = {
        "銘柄コード": "Code",
        "code": "Code",
        "ticker": "Code",
        "銘柄名": "Name",
        "name": "Name",
        "セクター": "Sector",
        "sector": "Sector"
    }

    df = df.rename(columns=rename_map)

    required = {"Code", "Name", "Sector"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"銘柄リストCSVに必要な列が不足しています: {missing}")

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

            y = g["Close"].values.reshape(-1, 1)
            x = np.arange(len(y)).reshape(-1, 1)

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
# セクター比較（1セクター1行）
# =========================
def sector_comparison(indicator_df, stock_df):
    merged = indicator_df.merge(stock_df, on="Code")

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
st.title("株式セクター分析")

csv_file = st.selectbox(
    "銘柄リストを選択",
    ["銘柄リスト_test.csv", "銘柄リスト.csv"]
)

if st.button("▶ 実行"):
    try:
        stock_df = load_stock_list(csv_file)
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.info("株価データ取得中...")
    price_df = fetch_price_data(stock_df["Code"].unique())

    if price_df.empty:
        st.error("株価データを取得できませんでした")
        st.stop()

    indicator_df = calculate_indicators(price_df)

    st.subheader("セクター比較")
    sector_df = sector_comparison(indicator_df, stock_df)
    st.dataframe(sector_df, use_container_width=True)

    st.subheader("セクター内分析")
    sector = st.selectbox("セクター選択", sector_df["Sector"])
    detail_df = sector_detail(indicator_df, stock_df, sector)
    st.dataframe(detail_df, use_container_width=True)
