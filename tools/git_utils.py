import subprocess
from pathlib import Path


def git_commit_and_push(repo_dir: Path, message: str):
    def run(cmd):
        subprocess.run(cmd, cwd=repo_dir, check=True, capture_output=True, text=True)

    # 変更有無チェック
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
    )

    if status.stdout.strip() == "":
        print("変更がないため commit をスキップします")
        return

    run(["git", "add", "."])
    run(["git", "commit", "-m", message])
    run(["git", "push"])
