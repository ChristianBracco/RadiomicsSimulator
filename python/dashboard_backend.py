from pathlib import Path
import subprocess
import sys
import json
import shutil
import traceback
import threading
import time
import os
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware


BASE_DIR = Path(__file__).resolve().parent
SERVER_DIR = BASE_DIR.parent
UPLOAD_DIR = SERVER_DIR / "data" / "uploads"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = SERVER_DIR / "models"

DASHBOARD_FILE_CANDIDATES = [
    RESULTS_DIR / "THERADIOMICS_results_dashboard_backend.html",
    BASE_DIR / "THERADIOMICS_results_dashboard_backend.html",
    RESULTS_DIR / "THERADIOMICS_results_dashboard_v6_backend.html",
    BASE_DIR / "THERADIOMICS_results_dashboard_v6_backend.html"
]

app = FastAPI(
    title="THERADIOMICS Backend Dashboard API",
    version="1.1-live-logs"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# RUN STATE GLOBALE
# ============================================================
#
# Serve per lanciare run_analysis.py in background e leggere stdout/stderr
# riga per riga, così l'HTML può fare polling e mostrare log live.
# ============================================================

RUN_LOCK = threading.Lock()

RUN_STATE = {
    "running": False,
    "process": None,
    "returncode": None,
    "started_at": None,
    "ended_at": None,
    "logs": [],
    "last_error": None
}


def _now_iso():
    return datetime.now().isoformat(timespec="seconds")


def _append_log(stream, line):
    with RUN_LOCK:
        RUN_STATE["logs"].append({
            "time": _now_iso(),
            "stream": stream,
            "line": line.rstrip("\n\r")
        })

        # Evita crescita infinita in memoria.
        if len(RUN_STATE["logs"]) > 5000:
            RUN_STATE["logs"] = RUN_STATE["logs"][-5000:]


def _reader_thread(pipe, stream_name):
    try:
        for line in iter(pipe.readline, ""):
            if not line:
                break
            _append_log(
                stream_name,
                line
            )
    except Exception as exc:
        _append_log(
            "stderr",
            f"[log-reader-error {stream_name}] {exc}"
        )
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def _watch_process(proc):
    returncode = proc.wait()

    with RUN_LOCK:
        RUN_STATE["running"] = False
        RUN_STATE["returncode"] = returncode
        RUN_STATE["ended_at"] = _now_iso()
        RUN_STATE["process"] = None

    _append_log(
        "system",
        f"Process finished with return code {returncode}"
    )


def _json_or_none(path):
    if not path.exists():
        return None

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


@app.get("/")
def index():
    for candidate in DASHBOARD_FILE_CANDIDATES:
        if candidate.exists():
            return FileResponse(
                candidate
            )

    return PlainTextResponse(
        "Dashboard HTML not found. Copy THERADIOMICS_results_dashboard_backend.html into python/results or python folder.",
        status_code=404
    )


@app.get("/api/status")
def status():
    with RUN_LOCK:
        running = RUN_STATE["running"]
        returncode = RUN_STATE["returncode"]
        started_at = RUN_STATE["started_at"]
        ended_at = RUN_STATE["ended_at"]
        log_count = len(RUN_STATE["logs"])

    return {
        "base_dir": str(BASE_DIR),
        "upload_dir": str(UPLOAD_DIR),
        "results_dir": str(RESULTS_DIR),
        "models_dir": str(MODELS_DIR),
        "dataset_exists": (UPLOAD_DIR / "Features_all.xlsx").exists(),
        "analysis_results_exists": (RESULTS_DIR / "analysis_results.json").exists(),
        "candidate_model_comparison_exists": (RESULTS_DIR / "candidate_model_comparison.json").exists(),
        "nested_cv_fold_details_exists": (RESULTS_DIR / "nested_cv_fold_details.json").exists(),
        "run": {
            "running": running,
            "returncode": returncode,
            "started_at": started_at,
            "ended_at": ended_at,
            "log_count": log_count
        }
    }


@app.post("/api/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload del file Excel dal browser.

    Il backend lo salva sempre come:
    server/server/data/uploads/Features_all.xlsx
    """
    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    target = (
        UPLOAD_DIR
        / "Features_all.xlsx"
    )

    backup = None

    if target.exists():
        backup = (
            UPLOAD_DIR
            / "Features_all.previous.xlsx"
        )

        shutil.copy2(
            target,
            backup
        )

    with open(
        target,
        "wb"
    ) as f:
        content = await file.read()
        f.write(
            content
        )

    return {
        "ok": True,
        "uploaded_name": file.filename,
        "saved_as": str(target),
        "bytes": len(content),
        "previous_backup": str(backup) if backup else None
    }


@app.post("/api/run-analysis")
async def run_analysis(request: Request):
    """
    Avvia run_analysis.py in background.

    A differenza della vecchia versione, NON usa subprocess.run().
    Usa invece subprocess.Popen() + python -u per avere stdout/stderr
    non bufferizzati e visibili in HTML durante il run.

    Accetta parametri JSON dal frontend e li passa come variabili d'ambiente.
    """

    script = (
        BASE_DIR
        / "run_analysis.py"
    )

    if not script.exists():
        return JSONResponse(
            {
                "ok": False,
                "error": f"run_analysis.py not found: {script}"
            },
            status_code=404
        )

    # Parse optional JSON body with run parameters from the dashboard form.
    run_params = {}
    if request is not None:
        try:
            run_params = await request.json()
        except Exception:
            run_params = {}

    with RUN_LOCK:
        if RUN_STATE["running"]:
            return {
                "ok": True,
                "already_running": True,
                "message": "run_analysis.py is already running",
                "started_at": RUN_STATE["started_at"],
                "log_count": len(RUN_STATE["logs"])
            }

        RUN_STATE["running"] = True
        RUN_STATE["returncode"] = None
        RUN_STATE["started_at"] = _now_iso()
        RUN_STATE["ended_at"] = None
        RUN_STATE["logs"] = []
        RUN_STATE["last_error"] = None

    _append_log(
        "system",
        "Starting run_analysis.py..."
    )

    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        # Forward dashboard form parameters as environment variables
        # so that run_analysis.py picks them up via _env_* helpers.
        if run_params.get("pruning_threshold"):
            env["THERADIOMICS_PRUNING_THRESHOLD"] = str(run_params["pruning_threshold"])
        if run_params.get("top_n_final_model_features"):
            env["THERADIOMICS_TOP_N_FINAL_MODEL_FEATURES"] = str(run_params["top_n_final_model_features"])
        if run_params.get("n_sample_size_simulations"):
            env["THERADIOMICS_N_SAMPLE_SIZE_SIMULATIONS"] = str(run_params["n_sample_size_simulations"])
        if run_params.get("feature_selection_method"):
            env["THERADIOMICS_FEATURE_SELECTION_METHOD"] = str(run_params["feature_selection_method"])

        _append_log(
            "system",
            f"Run parameters from dashboard: {json.dumps(run_params)}"
        )

        proc = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(script)
            ],
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env
        )

        with RUN_LOCK:
            RUN_STATE["process"] = proc

        threading.Thread(
            target=_reader_thread,
            args=(proc.stdout, "stdout"),
            daemon=True
        ).start()

        threading.Thread(
            target=_reader_thread,
            args=(proc.stderr, "stderr"),
            daemon=True
        ).start()

        threading.Thread(
            target=_watch_process,
            args=(proc,),
            daemon=True
        ).start()

        return {
            "ok": True,
            "started": True,
            "pid": proc.pid,
            "started_at": RUN_STATE["started_at"]
        }

    except Exception as exc:
        with RUN_LOCK:
            RUN_STATE["running"] = False
            RUN_STATE["returncode"] = -1
            RUN_STATE["ended_at"] = _now_iso()
            RUN_STATE["last_error"] = str(exc)
            RUN_STATE["process"] = None

        _append_log(
            "stderr",
            traceback.format_exc()
        )

        return JSONResponse(
            {
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc()
            },
            status_code=500
        )


@app.get("/api/run-status")
def run_status(since: int = 0):
    """
    Restituisce log live.

    Parametri:
    - since: indice dell'ultimo log già letto dal browser.

    Output:
    - logs: solo le righe nuove;
    - next_since: nuovo indice da usare al prossimo polling.
    """
    with RUN_LOCK:
        logs = RUN_STATE["logs"][since:]
        next_since = len(RUN_STATE["logs"])

        return {
            "ok": True,
            "running": RUN_STATE["running"],
            "returncode": RUN_STATE["returncode"],
            "started_at": RUN_STATE["started_at"],
            "ended_at": RUN_STATE["ended_at"],
            "logs": logs,
            "next_since": next_since,
            "last_error": RUN_STATE["last_error"]
        }


@app.post("/api/stop-run")
def stop_run():
    """
    Interrompe il run corrente, se presente.
    """
    with RUN_LOCK:
        proc = RUN_STATE["process"]

    if proc is None:
        return {
            "ok": True,
            "message": "No process running"
        }

    try:
        proc.terminate()
        _append_log(
            "system",
            "Terminate requested from dashboard."
        )

        return {
            "ok": True,
            "message": "Terminate requested"
        }

    except Exception as exc:
        return JSONResponse(
            {
                "ok": False,
                "error": str(exc)
            },
            status_code=500
        )


@app.get("/api/results")
def get_results():
    analysis = _json_or_none(
        RESULTS_DIR / "analysis_results.json"
    )

    candidate = _json_or_none(
        RESULTS_DIR / "candidate_model_comparison.json"
    )

    nested = _json_or_none(
        RESULTS_DIR / "nested_cv_fold_details.json"
    )

    summary = None

    summary_path = (
        RESULTS_DIR
        / "summary.txt"
    )

    if summary_path.exists():
        summary = summary_path.read_text(
            encoding="utf-8",
            errors="replace"
        )

    return {
        "ok": True,
        "analysis": analysis,
        "candidate": candidate,
        "nested": nested,
        "summary": summary,
        "paths": {
            "analysis_results": str(RESULTS_DIR / "analysis_results.json"),
            "candidate_model_comparison": str(RESULTS_DIR / "candidate_model_comparison.json"),
            "nested_cv_fold_details": str(RESULTS_DIR / "nested_cv_fold_details.json"),
            "summary": str(summary_path)
        }
    }


@app.get("/api/summary")
def get_summary():
    path = (
        RESULTS_DIR
        / "summary.txt"
    )

    if not path.exists():
        return PlainTextResponse(
            "summary.txt not found",
            status_code=404
        )

    return PlainTextResponse(
        path.read_text(
            encoding="utf-8",
            errors="replace"
        )
    )
