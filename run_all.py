"""
Master script to run Tables 2–4 and Figure 3 in order.

This is the Python equivalent of:

  python3 src/Simulation_Table2.py > outputs/table2.txt 2>&1
  python3 src/Empirical_Table3.py  > outputs/table3.txt 2>&1
  python3 src/Empirical_Table4.py  > outputs/table4.txt 2>&1
  python3 src/Empirical_Figure3.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = REPO_ROOT / "outputs"


def run_to_file(script_path: str, output_path: Path, *, env_extra: dict[str, str] | None = None) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)

    with output_path.open("w", encoding="utf-8") as f:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=REPO_ROOT.as_posix(),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )

    if result.returncode != 0:
        raise SystemExit(f"Failed: {script_path} (see {output_path})")


def main() -> int:
    run_to_file("src/Simulation_Table2.py", OUTPUTS_DIR / "table2.txt")
    run_to_file("src/Empirical_Table3.py", OUTPUTS_DIR / "table3.txt")
    run_to_file("src/Empirical_Table4.py", OUTPUTS_DIR / "table4.txt")

    # Figure 3 (script is expected to save the figure itself).
    run_to_file(
        "src/Empirical_Figure3.py",
        OUTPUTS_DIR / "figure3.txt",
        env_extra={"MPLBACKEND": "Agg"},
    )

    print(f"Saved outputs under: {OUTPUTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
