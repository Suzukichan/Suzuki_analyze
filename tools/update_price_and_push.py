import tkinter as tk
from tkinter import messagebox
import subprocess
import sys
import os

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__) + "/..")

def run_command(cmd):
    result = subprocess.run(
        cmd,
        cwd=PROJECT_DIR,
        shell=True,
        text=True,
        capture_output=True
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return result.stdout

def execute_all():
    try:
        status.set("① 株価データ取得中...")
        root.update()

        run_command(f"{sys.executable} fetch_price.py")

        status.set("② Git commit 中...")
        root.update()

        run_command("git add tse_price_60days.csv")
        run_command('git commit -m "update price csv"')

        status.set("③ GitHub に push 中...")
        root.update()

        run_command("git push")

        status.set("完了")
        messagebox.showinfo(
            "完了",
            "株価CSVの生成とGitHubへのpushが完了しました"
        )

    except Exception as e:
        messagebox.showerror("エラー", str(e))
        status.set("エラー")

# =========================
# GUI
# =========================
root = tk.Tk()
root.title("株価CSV更新ツール")
root.geometry("420x180")

label = tk.Label(
    root,
    text="株価CSVを更新してGitHubに反映します",
    font=("Meiryo", 11)
)
label.pack(pady=10)

btn = tk.Button(
    root,
    text="実行（CSV生成 → commit → push）",
    font=("Meiryo", 11),
    height=2,
    width=30,
    command=execute_all
)
btn.pack(pady=10)

status = tk.StringVar()
status.set("待機中")

status_label = tk.Label(root, textvariable=status)
status_label.pack(pady=5)

root.mainloop()
