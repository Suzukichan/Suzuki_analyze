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

# =====================================================
# ① セクター比較分析
# =====================================================
st.header("① セクター比較")

if st.button("セクター比較を実行", key="sector_exec"):
    results = []

    for sector, df_s in price_df.groupby("Sector"):
        row = {"セクター": sector}

        for w in [5, 20, 60]:
            tmp = df_s.groupby("Code").tail(w)

            mean_roc = tmp.groupby("Code")["ROC"].mean().mean()
            _, r2 = calc_slope_r2(tmp.groupby("Date")["ROC"].mean())

            row[f"{w}日移動平均"] = mean_roc
            row[f"{w}日相対強度"] = mean_roc
            row[f"{w}日決定係数"] = r2

        results.append(row)

    result_df = pd.DataFrame(results)

    # ---------- ソートUI ----------
    st.subheader("ソート設定（セクター比較）")

    col1, col2 = st.columns(2)

    with col1:
        sort_key1 = st.selectbox(
            "第1キー",
            result_df.columns[1:],
            key="sector_sort_key1"
        )
        order1 = st.radio(
            "第1キー順序",
            ["降順", "昇順"],
            horizontal=True,
            key="sector_sort_order1"
        )

    with col2:
        sort_key2 = st.selectbox(
            "第2キー（任意）",
            ["なし"] + list(result_df.columns[1:]),
            key="sector_sort_key2"
        )
        order2 = st.radio(
            "第2キー順序",
            ["降順", "昇順"],
            horizontal=True,
            key="sector_sort_order2"
        )

    # ---------- ソート処理 ----------
    sort_cols = [sort_key1]
    ascending = [order1 == "昇順"]

    if sort_key2 != "なし":
        sort_cols.append(sort_key2)
        ascending.append(order2 == "昇順")

    result_df = result_df.sort_values(by=sort_cols, ascending=ascending)

    st.dataframe(result_df, use_container_width=True)

# =====================================================
# ② セクター内分析
# =====================================================
st.header("② セクター内分析")

sector_sel = st.selectbox(
    "セクター選択",
    sorted(price_df["Sector"].unique()),
    key="sector_select"
)

if st.button("セクター内分析を実行", key="inner_exec"):
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
                g["Volume"].iloc[-1] / g["Volume"].iloc[-5:-1].mean()
            )
        else:
            row["出来高変化率"] = np.nan

        rows.append(row)

    stock_df = pd.DataFrame(rows)

    # ---------- ソートUI ----------
    st.subheader("ソート設定（セクター内）")

    col1, col2 = st.columns(2)

    with col1:
        stock_key1 = st.selectbox(
            "第1キー",
            stock_df.columns[2:],
            key="stock_sort_key1"
        )
        stock_order1 = st.radio(
            "第1キー順序",
            ["降順", "昇順"],
            horizontal=True,
            key="stock_sort_order1"
        )

    with col2:
        stock_key2 = st.selectbox(
            "第2キー（任意）",
            ["なし"] + list(stock_df.columns[2:]),
            key="stock_sort_key2"
        )
        stock_order2 = st.radio(
            "第2キー順序",
            ["降順", "昇順"],
            horizontal=True,
            key="stock_sort_order2"
        )

    # ---------- ソート処理 ----------
    sort_cols = [stock_key1]
    ascending = [stock_order1 == "昇順"]

    if stock_key2 != "なし":
        sort_cols.append(stock_key2)
        ascending.append(stock_order2 == "昇順")

    stock_df = stock_df.sort_values(by=sort_cols, ascending=ascending)

    st.dataframe(stock_df, use_container_width=True)
