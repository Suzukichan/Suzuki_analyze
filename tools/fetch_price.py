import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path


def fetch_price_csv(
    code_csv_path: str,
    output_csv_path: str,
    days: int = 60,
    progress_callback=None,
):
    """
    stock_codes.csv を読み込み、株価CSVを生成する
    """

    codes_df = pd.read_csv(code_csv_path)

    # 必須カラムチェック
    required_cols = ["銘柄コード", "銘柄名", "市場区分", "セクター", "有効"]
    for col in required_cols:
        if col not in codes_df.columns:
            raise ValueError(f"CSVに必須カラムがありません: {col}")

    # 有効銘柄のみ
    codes_df = codes_df[codes_df["有効"] == 1]

    end_date = datetime.today()
    start_date = end_date - timedelta(days=days * 2)

    all_prices = []
    total = len(codes_df)

    for idx, row in codes_df.iterrows():
        code = str(row["銘柄コード"])
        name = row["銘柄名"]
        sector = row["セクター"]

        ticker = f"{code}.T"

        try:
            df = yf.download(
                ticker,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                progress=False,
            )

            if df.empty:
                continue

            df = df.reset_index()
            df = df[["Date", "Close"]].tail(days)

            df["Code"] = code
            df["Name"] = name
            df["Sector"] = sector

            all_prices.append(df)

        except Exception as e:
            print(f"取得失敗: {code} ({e})")

        if progress_callback:
            progress_callback((len(all_prices) / total) * 100)

    if not all_prices:
        raise RuntimeError("株価データが1件も取得できませんでした")

    result_df = pd.concat(all_prices, ignore_index=True)
    result_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
