import threading
import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

# ===== パス解決（最重要） =====
TOOLS_DIR = Path(__file__).resolve().parent
ROOT_DIR = TOOLS_DIR.parent

sys.path.append(str(ROOT_DIR))

from tools.fetch_price import fetch_price_csv
from tools.git_utils import git_commit_and_push

CODE_CSV_PATH = TOOLS_DIR / "stock_codes.csv"
OUTPUT_CSV_PATH = ROOT_DIR / "tse_price_60days.csv"


def run_process(progress, status_var, run_button):
    try:
        run_button.config(state="disabled")
        status_var.set("株価データ取得中...")

        def progress_callback(value):
            progress["value"] = value
            progress.update()

        fetch_price_csv(
            code_csv_path=str(CODE_CSV_PATH),
            output_csv_path=str(OUTPUT_CSV_PATH),
            progress_callback=progress_callback,
        )

        status_var.set("GitHub に commit & push 中...")
        progress["value"] = 90

        git_commit_and_push(
            repo_dir=ROOT_DIR,
            message="Update stock price CSV"
        )

        progress["value"] = 100
        status_var.set("完了しました")
        messagebox.showinfo("完了", "CSV更新とGitHub反映が完了しました")

    except Exception as e:
        messagebox.showerror("エラー", str(e))

    finally:
        run_button.config(state="normal")
        progress["value"] = 0


def start():
    root = tk.Tk()
    root.title("株価CSV更新ツール")
    root.geometry("420x200")

    ttk.Label(root, text="株価データ更新 & GitHub Push", font=("Meiryo", 12, "bold")).pack(pady=10)

    status_var = tk.StringVar(value="待機中")
    ttk.Label(root, textvariable=status_var).pack(pady=5)

    progress = ttk.Progressbar(root, length=350, mode="determinate")
    progress.pack(pady=5)

    run_button = ttk.Button(
        root,
        text="実行（CSV生成 → commit → push）",
        command=lambda: threading.Thread(
            target=run_process,
            args=(progress, status_var, run_button),
            daemon=True
        ).start()
    )
    run_button.pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    start()