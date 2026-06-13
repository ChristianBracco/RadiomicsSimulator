#!/usr/bin/env python3
"""
THERADIOMICS launcher cross-platform.

Uso:
  python start_theradiomics.py

Su Windows lancia start_theradiomics.bat.
Su macOS/Linux lancia start_theradiomics.sh.
"""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    system = platform.system().lower()

    if "windows" in system:
        script = root / "start_theradiomics.bat"
        if not script.exists():
            print(f"[ERRORE] File non trovato: {script}")
            return 1
        return subprocess.call(["cmd", "/c", str(script)], cwd=root)

    script = root / "start_theradiomics.sh"
    if not script.exists():
        print(f"[ERRORE] File non trovato: {script}")
        return 1

    try:
        script.chmod(script.stat().st_mode | 0o755)
    except Exception:
        pass

    return subprocess.call(["bash", str(script)], cwd=root)


if __name__ == "__main__":
    raise SystemExit(main())
