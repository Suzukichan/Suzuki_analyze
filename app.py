import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

# =========================
# Streamlit設定
# =========================
st.set_page_config(page_title="株価分析アプリ", layout="wide")
st.title("株価分析アプリ")

# =========================
# 定数
# =========================
CODE_FILE = "銘柄リスト.csv"
PRICE_FILE = "stock_close_all.csv"

# =========================
# 共通関数
# =========================
def calc_slope_r2(series: pd.Series):
    series = series.dropna()
    if len(series) < 2:
        return np.nan, np.nan
    y = series.values
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    return model.coef_[0], model.score(x, y)

# =========================
# 銘柄リスト読み込み
# =========================
code_df = pd.read_csv(CODE_FILE)
code_df = code_df[code_df["有効"] == 1].copy()
code_df["銘柄コード"] = code_df["銘柄コード"].astype(str)

code_name_map = dict(zip(code_df["銘柄コード"], code_df["銘柄名"]))

# =========================
# 株価CSV読み込み
# =========================
required_cols = ["Date", "Code", "Sector", "Market", "Close", "Volume"]

if os.path.exists(PRICE_FILE):
    price_df = pd.read_csv(PRICE_FILE, encoding="utf-8-sig")

    if not set(required_cols).issubset(price_df.columns):
        st.error("CSVファイルの形式が不正、または必要な列が不足しています。")
        st.stop()

    price_df["Date"] = pd.to_datetime(price_df["Date"], errors="coerce")
    price_df["Code"] = price_df["Code"].astype(str)
    price_df["Close"] = pd.to_numeric(price_df["Close"], errors="coerce")
    price_df["Volume"] = pd.to_numeric(price_df["Volume"], errors="coerce")

    price_df = price_df.dropna(subset=["Date", "Close", "Volume"])
    price_df = price_df.sort_values(["Code", "Date"])

    price_df["ROC"] = price_df.groupby("Code")["Close"].pct_change()
else:
    st.error("株価CSVが見つかりません")
    st.stop()

# ==================================================
# ① セクター比較
# ==================================================
st.header("① セクター比較")

sector_metrics = [
    "5日移動平均", "5日相対強度", "5日決定係数",
    "20日移動平均", "20日相対強度", "20日決定係数",
    "60日移動平均", "60日相対強度", "60日決定係数",
]

st.subheader("並び替え設定（複数キー）")

col1, col2 = st.columns(2)

with col1:
    sector_sort1 = st.selectbox("第1キー", sector_metrics)
    sector_order1 = st.radio("第1キー順序", ["降順", "昇順"], horizontal=True)

with col2:
    sector_sort2 = st.selectbox("第2キー（任意）", ["なし"] + sector_metrics)
    sector_order2 = st.radio(
        "第2キー順序", ["降順", "昇順"], horizontal=True, key="sector2"
    )

if st.button("セクター比較を実行"):
    results = []

    for sector, df_s in price_df.groupby("Sector"):
        row = {"セクター": sector}

        for w in [5, 20, 60]:
            tmp = df_s.groupby("Code").tail(w)

            ma = tmp.groupby("Code")["ROC"].mean().mean()
            _, r2 = calc_slope_r2(tmp.groupby("Date")["ROC"].mean())

            row[f"{w}日移動平均"] = ma
            row[f"{w}日相対強度"] = ma
            row[f"{w}日決定係数"] = r2

        results.append(row)

    df_result = pd.DataFrame(results)

    sort_cols = [sector_sort1]
    ascendings = [sector_order1 == "昇順"]

    if sector_sort2 != "なし":
        sort_cols.append(sector_sort2)
        ascendings.append(sector_order2 == "昇順")

    df_result = df_result.sort_values(by=sort_cols, ascending=ascendings)
    st.dataframe(df_result, use_container_width=True)

# ==================================================
# ② セクター内分析
# ==================================================
st.header("② セクター内分析")

sector_sel = st.selectbox(
    "セクター選択",
    sorted(price_df["Sector"].unique())
)

stock_metrics = sector_metrics + ["出来高変化率"]

st.subheader("並び替え設定（複数キー）")

col3, col4 = st.columns(2)

with col3:
    stock_sort1 = st.selectbox("第1キー", stock_metrics)
    stock_order1 = st.radio("第1キー順序", ["降順", "昇順"], horizontal=True)

with col4:
    stock_sort2 = st.selectbox("第2キー（任意）", ["なし"] + stock_metrics)
    stock_order2 = st.radio(
        "第2キー順序", ["降順", "昇順"], horizontal=True, key="stock2"
    )

if st.button("セクター内分析を実行"):
    df_f = price_df[price_df["Sector"] == sector_sel]
    rows = []

    for code, g in df_f.groupby("Code"):
        g = g.sort_values("Date")

        row = {
            "銘柄コード": code,
            "銘柄名": code_name_map.get(code, "")
        }

        for w in [5, 20, 60]:
            roc = g["ROC"].tail(w)
            row[f"{w}日移動平均"] = roc.mean()
            row[f"{w}日相対強度"] = roc.mean()
            _, r2 = calc_slope_r2(roc)
            row[f"{w}日決定係数"] = r2

        if len(g) >= 5:
            row["出来高変化率"] = (
                g["Volume"].iloc[-1] /
                g["Volume"].iloc[-5:-1].mean()
            )
        else:
            row["出来高変化率"] = np.nan

        rows.append(row)

    df_rows = pd.DataFrame(rows)

    sort_cols = [stock_sort1]
    ascendings = [stock_order1 == "昇順"]

    if stock_sort2 != "なし":
        sort_cols.append(stock_sort2)
        ascendings.append(stock_order2 == "昇順")

    df_rows = df_rows.sort_values(by=sort_cols, ascending=ascendings)
    st.dataframe(df_rows, use_container_width=True)
