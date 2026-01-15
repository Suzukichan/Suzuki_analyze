import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.linear_model import LinearRegression

# =========================
# Streamlit設定
# =========================
st.set_page_config(page_title="株価分析アプリ", layout="wide")
st.title("株価分析アプリ")

# =========================
# 定数
# =========================
CODE_FILE = "銘柄リスト_test.csv"
PRICE_FILE = "tse_price_60days.csv"
PERIOD = "90d"

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
if not os.path.exists(CODE_FILE):
    st.error("銘柄リスト.csv が見つかりません")
    st.stop()

code_df = pd.read_csv(CODE_FILE)

required_cols = {"銘柄コード", "銘柄名", "市場区分", "セクター", "有効"}
if not required_cols.issubset(code_df.columns):
    st.error("銘柄リストの列構成が不正です")
    st.write(code_df.columns.tolist())
    st.stop()

code_df = code_df[code_df["有効"] == 1].copy()
code_df["銘柄コード"] = code_df["銘柄コード"].astype(str)

# 銘柄名参照用辞書
code_name_map = dict(
    zip(code_df["銘柄コード"], code_df["銘柄名"])
)

# =========================
# 株価データ取得
# =========================
st.header("① 株価データ取得")

if st.button("株価データを取得（90日）"):
    rows = []

    progress = st.progress(0.0)
    status = st.empty()
    total = len(code_df)

    for i, row in code_df.iterrows():
        code = row["銘柄コード"]
        ticker = f"{code}.T"
        status.text(f"{ticker} 取得中...")

        try:
            df = yf.download(
                ticker,
                period=PERIOD,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False
            )

            if df.empty:
                continue

            df = df.reset_index()
            df["Code"] = code
            df["Sector"] = row["セクター"]
            df["Market"] = row["市場区分"]

            df = df[["Date", "Code", "Sector", "Market", "Close", "Volume"]]
            rows.append(df)

        except Exception:
            continue

        progress.progress((i + 1) / total)
        time.sleep(0.05)

    progress.empty()
    status.empty()

    if not rows:
        st.error("株価データを取得できませんでした")
        st.stop()

    price_df = pd.concat(rows, ignore_index=True)
    price_df.to_csv(PRICE_FILE, index=False, encoding="utf-8-sig")
    st.success("株価データ取得が完了しました")

# =========================
# 株価CSV読み込み
# =========================
if not os.path.exists(PRICE_FILE):
    st.warning("株価CSVが存在しません。上で取得してください。")
    st.stop()

price_df = pd.read_csv(PRICE_FILE)

expected_cols = {"Date", "Code", "Sector", "Market", "Close", "Volume"}
if not expected_cols.issubset(price_df.columns):
    st.error("株価CSVの列構成が不正です")
    st.write(price_df.columns.tolist())
    st.stop()

price_df["Date"] = pd.to_datetime(price_df["Date"], errors="coerce")
price_df = price_df.dropna(subset=["Date"])
price_df["Code"] = price_df["Code"].astype(str)
price_df = price_df.sort_values(["Code", "Date"])

# 騰落率（ROC）
price_df["ROC"] = price_df.groupby("Code")["Close"].pct_change()

# =========================
# セクター比較分析（1セクター1行）
# =========================
st.header("② セクター比較分析")

if st.button("セクター比較を実行"):
    results = []

    for sector, df_s in price_df.groupby("Sector"):
        row = {"セクター": sector}

        for w in [5, 20, 60]:
            tmp = df_s.groupby("Code").tail(w)

            # 騰落率ベース移動平均
            ma = tmp.groupby("Code")["ROC"].mean().mean()
            rs = ma  # 相対強度＝平均ROC

            slope, r2 = calc_slope_r2(
                tmp.groupby("Date")["ROC"].mean()
            )

            row[f"{w}日移動平均"] = ma
            row[f"{w}日相対強度"] = rs
            row[f"{w}日決定係数"] = r2

        results.append(row)

    st.success("セクター比較が完了しました")
    st.dataframe(pd.DataFrame(results), use_container_width=True)

# =========================
# セクター内分析（1銘柄1行）
# =========================
st.header("③ セクター内分析")

sector_sel = st.selectbox(
    "セクター選択",
    sorted(price_df["Sector"].unique())
)

markets_sel = st.multiselect(
    "市場区分",
    ["プライム", "スタンダード", "グロース"],
    default=["プライム", "スタンダード", "グロース"]
)

if st.button("セクター内分析を実行"):
    df_f = price_df[
        (price_df["Sector"] == sector_sel) &
        (price_df["Market"].isin(markets_sel))
    ]

    rows = []

    for code, g in df_f.groupby("Code"):
        g = g.sort_values("Date")

        row = {
            "銘柄コード": code,
            "銘柄名": code_name_map.get(code, "")
        }

        for w in [5, 20, 60]:
            roc = g["ROC"].tail(w)

            ma = roc.mean()
            slope, r2 = calc_slope_r2(roc)

            row[f"{w}日移動平均"] = ma
            row[f"{w}日相対強度"] = ma
            row[f"{w}日決定係数"] = r2

        if len(g) >= 5:
            row["出来高変化率"] = (
                g["Volume"].iloc[-1] /
                g["Volume"].iloc[-5:-1].mean()
            )
        else:
            row["出来高変化率"] = np.nan

        rows.append(row)

    st.success("セクター内分析が完了しました")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

