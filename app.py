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
st.title("株価分析アプリ（試作）")

# =========================
# 定数
# =========================
CODE_FILE = "銘柄リスト.csv"
PRICE_FILE = "tse_price_60days.csv"
PERIOD = "90d"

# =========================
# 共通関数
# =========================
def calc_trend_r2(series: pd.Series):
    if series is None or len(series) < 2:
        return np.nan, np.nan
    y = series.values
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    return model.coef_[0], model.score(x, y)

@st.cache_data(ttl=3600)
def download_one(ticker):
    return yf.download(
        ticker,
        period=PERIOD,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False
    )

# =========================
# 銘柄リスト読み込み
# =========================
st.header("株価データ取得")

if not os.path.exists(CODE_FILE):
    st.error("銘柄リスト.csv が見つかりません")
    st.stop()

code_df = pd.read_csv(CODE_FILE)
required_cols = {"銘柄コード", "銘柄名", "市場区分", "セクター", "有効"}
if not required_cols.issubset(code_df.columns):
    st.error("銘柄リストの列構成が不正です")
    st.stop()

code_df = code_df[code_df["有効"] == 1].copy()
code_df["銘柄コード"] = code_df["銘柄コード"].astype(str)

# =========================
# 株価データ取得
# =========================
if st.button("60日分の株価データを取得"):
    progress = st.progress(0.0)
    status = st.empty()

    all_rows = []
    total = len(code_df)

    for i, row in code_df.iterrows():
        code = row["銘柄コード"]
        ticker = f"{code}.T"
        status.text(f"{ticker} 取得中... ({len(all_rows)}/{total})")

        try:
            df = download_one(ticker)

            if df.empty:
                continue

            df = df.reset_index()
            df["Code"] = code
            df["Sector"] = row["セクター"]
            df["Market"] = row["市場区分"]

            df = df[["Date", "Code", "Sector", "Market", "Close", "Volume"]]
            all_rows.append(df)

        except Exception as e:
            st.warning(f"{ticker} 取得失敗: {e}")

        progress.progress((i + 1) / total)
        time.sleep(0.2)

    progress.empty()
    status.empty()

    if all_rows:
        price_df = pd.concat(all_rows, ignore_index=True)
        price_df.to_csv(PRICE_FILE, index=False, encoding="utf-8-sig")
        st.success("株価データ取得が完了しました")
    else:
        st.error("株価データを取得できませんでした")

# =========================
# 株価CSV読み込み
# =========================
price_df = None
if os.path.exists(PRICE_FILE):
    price_df = pd.read_csv(PRICE_FILE)
    price_df["Date"] = pd.to_datetime(price_df["Date"])
    price_df = price_df.sort_values(["Code", "Date"])
    price_df["ROC"] = price_df.groupby("Code")["Close"].pct_change()
else:
    st.info("まだ株価CSVがありません。先にデータ取得を実行してください。")

# =========================
# セクター比較分析
# =========================
st.header("セクター比較分析")

if st.button("セクター比較分析を実行"):
    if price_df is None or price_df.empty:
        st.warning("株価データがありません")
    else:
        result = []
        sectors = price_df["Sector"].dropna().unique()

        for sector in sectors:
            df = price_df[price_df["Sector"] == sector]

            for window in [5, 20, 60]:
                tmp = df.groupby("Code").tail(window)
                mean_roc = tmp["ROC"].mean()

                slope, r2 = calc_trend_r2(
                    tmp.groupby("Date")["ROC"].mean()
                )

                result.append({
                    "セクター": sector,
                    "期間": window,
                    "平均騰落率": mean_roc,
                    "トレンド傾き": slope,
                    "R2": r2
                })

        st.success("セクター比較分析が完了しました")
        st.dataframe(pd.DataFrame(result), use_container_width=True)

# =========================
# セクター内分析
# =========================
st.header("セクター内分析")

if price_df is not None and not price_df.empty:
    sector_sel = st.selectbox(
        "セクター選択",
        sorted(price_df["Sector"].dropna().unique())
    )
    markets = st.multiselect(
        "市場区分",
        ["プライム", "スタンダード", "グロース"],
        default=["プライム", "スタンダード", "グロース"]
    )

    if st.button("セクター内分析を実行"):
        df = price_df[
            (price_df["Sector"] == sector_sel) &
            (price_df["Market"].isin(markets))
        ]

        results = []

        for code, g in df.groupby("Code"):
            g = g.sort_values("Date")

            for window in [5, 20, 60]:
                roc = g["ROC"].tail(window)
                slope, r2 = calc_trend_r2(roc)

                vol_ratio = np.nan
                if len(g) >= 5:
                    vol_ratio = (
                        g["Volume"].iloc[-1] /
                        g["Volume"].iloc[-5:-1].mean()
                    )

                results.append({
                    "銘柄コード": code,
                    "期間": window,
                    "平均騰落率": roc.mean(),
                    "トレンド傾き": slope,
                    "R2": r2,
                    "出来高変化率": vol_ratio
                })

        st.success("セクター内分析が完了しました")
        st.dataframe(pd.DataFrame(results), use_container_width=True)
else:
    st.info("株価データ取得後に分析機能が有効になります")
