"""
Watcher : attend la fin du fine-tune (PID lancé en arrière-plan), puis lance
automatiquement le benchmark du kernel event-driven (float32 vs int16 vs torch)
sur MaleCNS, et l'append au leaderboard. Logs -> bench_after_finetune.log
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "bench_after_finetune.log"
FINETUNE_PID = int(os.environ.get("FINETUNE_PID", "0"))  # PID ou 0 = dispo
FINETUNE_PID_FILE = ROOT / "finetune.pid"


def write(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def _pid_alive(pid):
    """Check Windows-safe : tasklist renvoie le PID s'il existe."""
    if pid <= 0:
        return False
    try:
        out = subprocess.run(
            ["tasklist", "/FI", "PID eq %d" % pid, "/NH"],
            capture_output=True, text=True, timeout=15,
        ).stdout
        return "python" in out.lower() and str(pid) in out
    except Exception:
        return False


def finetune_done():
    pid = FINETUNE_PID
    if pid <= 0 and FINETUNE_PID_FILE.exists():
        try:
            pid = int(FINETUNE_PID_FILE.read_text().strip())
        except ValueError:
            pid = 0
    # le process a disparu ET le log contient TERMINE
    if _pid_alive(pid):
        return False
    try:
        txt = (ROOT / "finetune.log").read_text(encoding="utf-8", errors="ignore")
        return "[finetune] TERMINE" in txt or "pas de généralisation" in txt
    except FileNotFoundError:
        return False


def main():
    write("=== watcher démarré %s ===" % time.strftime("%Y-%m-%d %H:%M:%S"))
    while not finetune_done():
        time.sleep(60)
        # progress echo périodique
        try:
            last = (ROOT / "finetune.log").read_text(encoding="utf-8", errors="ignore").strip().splitlines()
            if last:
                write("  [wait] %s" % last[-1])
        except Exception:
            pass

    write("=== fine-tune terminé, lancement du benchmark MaleCNS ===")
    r = subprocess.run(
        [sys.executable, "-B", "bench_native_alpha.py"],
        cwd=str(ROOT / "backend"),
        capture_output=True, text=True,
    )
    write("--- sortie benchmark ---")
    write(r.stdout)
    if r.returncode != 0:
        write("--- STDERR ---")
        write(r.stderr[-4000:] if r.stderr else "(vide)")
    write("=== watcher terminé %s (rc=%s) ===" % (
        time.strftime("%Y-%m-%d %H:%M:%S"), r.returncode))


if __name__ == "__main__":
    main()