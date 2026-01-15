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
CODE_FILE = "銘柄リスト_test.csv"   # ← 本番時は 銘柄リスト.csv に戻す
PRICE_FILE = "tse_price_60days.csv"
PERIOD = "90d"

# =========================
# 共通関数
# =========================
def calc_trend_r2(series: pd.Series):
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
    st.error(f"{CODE_FILE} が見つかりません")
    st.stop()

code_df = pd.read_csv(CODE_FILE)

required_cols = {"銘柄コード", "銘柄名", "市場区分", "セクター", "有効"}
if not required_cols.issubset(code_df.columns):
    st.error("銘柄リストの列構成が不正です")
    st.write(code_df.columns.tolist())
    st.stop()

code_df = code_df[code_df["有効"] == 1].copy()
code_df["銘柄コード"] = code_df["銘柄コード"].astype(str)

# =========================
# 株価データ取得
# =========================
st.header("株価データ取得")

if st.button("60日分の株価データを取得"):
    progress = st.progress(0.0)
    status = st.empty()

    all_rows = []
    total = len(code_df)

    for i, row in enumerate(code_df.itertuples(), 1):
        code = row.銘柄コード
        ticker = f"{code}.T"
        status.text(f"{ticker} 取得中... ({i}/{total})")

        try:
            df = yf.download(
                ticker,
                period=PERIOD,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False
            )

            if df.empty or "Close" not in df.columns:
                continue

            df = df.reset_index()
            df["Code"] = code
            df["Sector"] = row.セクター
            df["Market"] = row.市場区分

            df = df[["Date", "Code", "Sector", "Market", "Close", "Volume"]]
            all_rows.append(df)

            time.sleep(0.2)  # RateLimit対策

        except Exception:
            continue

        progress.progress(i / total)

    progress.empty()
    status.empty()

    if not all_rows:
        st.error("株価データを取得できませんでした")
        st.stop()

    price_df = pd.concat(all_rows, ignore_index=True)
    price_df.to_csv(PRICE_FILE, index=False, encoding="utf-8-sig")

    st.success("すべての銘柄の株価データ取得が完了しました")

# =========================
# 株価CSV読み込み（重要修正版）
# =========================
if not os.path.exists(PRICE_FILE):
    st.warning("株価CSVがまだ存在しません。上のボタンから取得してください。")
    st.stop()

price_df = pd.read_csv(PRICE_FILE)

# 型の正規化（最重要）
price_df["Date"] = pd.to_datetime(price_df["Date"], errors="coerce")
price_df["Close"] = pd.to_numeric(price_df["Close"], errors="coerce")
price_df["Volume"] = pd.to_numeric(price_df["Volume"], errors="coerce")

price_df = price_df.dropna(subset=["Date", "Close"])
price_df = price_df.sort_values(["Code", "Date"])

# 騰落率
price_df["ROC"] = price_df.groupby("Code")["Close"].pct_change()

# =========================
# セクター比較分析
# =========================
st.header("セクター比較分析")

if st.button("セクター比較分析を実行"):
    results = []

    for sector in price_df["Sector"].unique():
        df_sector = price_df[price_df["Sector"] == sector]

        for window in [5, 20, 60]:
            tmp = df_sector.groupby("Code").tail(window)
            mean_roc = tmp.groupby("Code")["ROC"].mean().mean()
            slope, r2 = calc_trend_r2(tmp.groupby("Date")["ROC"].mean())

            results.append({
                "セクター": sector,
                "期間(日)": window,
                "平均騰落率": mean_roc,
                "トレンド傾き": slope,
                "決定係数(R2)": r2
            })

    st.success("セクター比較分析が完了しました")
    st.dataframe(pd.DataFrame(results), use_container_width=True)

# =========================
# セクター内分析
# =========================
st.header("セクター内分析")

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
                vol_ratio = g["Volume"].iloc[-1] / g["Volume"].iloc[-5:-1].mean()

            results.append({
                "銘柄コード": code,
                "期間(日)": window,
                "平均騰落率": roc.mean(),
                "トレンド傾き": slope,
                "決定係数(R2)": r2,
                "出来高変化率": vol_ratio
            })

    st.success("セクター内分析が完了しました")
    st.dataframe(pd.DataFrame(results), use_container_width=True)
