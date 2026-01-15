# fetch_price.py
import time
import pandas as pd
import yfinance as yf

CODE_FILE = "銘柄リスト.csv"
PRICE_FILE = "tse_price_60days.csv"
PERIOD = "90d"

def fetch_price_csv():
    code_df = pd.read_csv(CODE_FILE)

    code_df = code_df[code_df["有効"] == 1].copy()
    code_df["銘柄コード"] = code_df["銘柄コード"].astype(str)

    all_rows = []

    for _, row in code_df.iterrows():
        code = row["銘柄コード"]
        ticker = f"{code}.T"

        try:
            df = yf.download(
                ticker,
                period=PERIOD,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )

            if df.empty:
                continue

            df = df.reset_index()
            df["Code"] = code
            df["Name"] = row["銘柄名"]
            df["Sector"] = row["セクター"]
            df["Market"] = row["市場区分"]

            df = df[["Date", "Code", "Name", "Sector", "Market", "Close", "Volume"]]
            all_rows.append(df)

            time.sleep(0.2)  # RateLimit対策

        except Exception as e:
            print(f"{ticker} 取得失敗: {e}")

    if not all_rows:
        raise RuntimeError("株価データを1件も取得できませんでした")

    price_df = pd.concat(all_rows, ignore_index=True)
    price_df.to_csv(PRICE_FILE, index=False, encoding="utf-8-sig")
