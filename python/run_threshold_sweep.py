from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
ANALYSIS_JSON = RESULTS_DIR / "analysis_results.json"
SUMMARY_TXT = RESULTS_DIR / "summary.txt"
SCRIPT = BASE_DIR / "run_analysis.py"


def _parse_threshold_number(raw: str) -> float | None:
    """
    Parse one threshold.

    Supports both decimal dot (0.85) and decimal comma (0,85) when the
    threshold is a single token. Values must be inside (0, 1).
    """
    raw = str(raw).strip()
    if not raw:
        return None

    try:
        value = float(raw.replace(",", "."))
    except ValueError:
        print(f"[WARN] Ignoro soglia non valida: {raw!r}")
        return None

    if 0 < value < 1:
        return value

    print(f"[WARN] Ignoro soglia fuori range 0-1: {value}")
    return None


def _parse_thresholds(raw: str | None) -> list[float]:
    """
    Parse pruning thresholds passed from the dashboard.

    Accepted forms:
      - 0.70,0.75,0.80       decimal dots + comma list separator
      - 0.70;0.75;0.80       decimal dots + semicolon separator
      - 0,70;0,75;0,80       decimal commas + semicolon separator

    Important: if decimal commas are used, separate thresholds with semicolons.
    """
    if not raw:
        return [0.85]

    raw = str(raw).strip()

    # If semicolons are present, they are the list separator and decimal commas
    # are allowed inside each token.
    if ";" in raw:
        parts = raw.split(";")
    else:
        # Default dashboard notation: 0.70,0.75,0.80
        parts = raw.split(",")

    values: list[float] = []
    for part in parts:
        value = _parse_threshold_number(part)
        if value is not None:
            values.append(value)

    # Round for stable de-duplication, then sort.
    values = sorted(set(round(v, 6) for v in values))
    return values or [0.85]


def _safe_label(threshold: float) -> str:
    return str(threshold).replace(".", "")


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pick_best_model(candidate_models: dict) -> dict | None:
    models = candidate_models.get("summary") or candidate_models.get("models") or []
    ok = [m for m in models if isinstance(m, dict) and m.get("status") == "ok"]
    if not ok:
        return None

    def score(m: dict) -> float:
        group = m.get("group_cv_mean_auc") or 0
        loo = m.get("loocv_auc") or 0
        n_features = m.get("n_features") or len(m.get("features") or []) or 99
        return ((float(group) + float(loo)) / 2.0) - max(0, int(n_features) - 1) * 0.002

    return sorted(ok, key=score, reverse=True)[0]


def _extract_row(threshold: float, analysis: dict) -> dict:
    candidate_models = analysis.get("candidate_models") or {}
    best = _pick_best_model(candidate_models)
    cv = analysis.get("cv") or {}
    loocv = analysis.get("loocv") or {}
    sample_design = analysis.get("sample_size_design") or {}

    return {
        "status": "ok",
        "returncode": 0,
        "threshold": float(threshold),
        "patients": analysis.get("patients"),
        "lesions": analysis.get("lesions"),
        "responders": analysis.get("responders"),
        "non_responders": analysis.get("non_responders"),
        "features_before_pruning": analysis.get("features_before_pruning"),
        "features_after_pruning": analysis.get("features_after_pruning"),
        "removed_features_count": analysis.get("removed_features_count"),
        "nested_cv_mean_auc": cv.get("mean_auc"),
        "nested_cv_std_auc": cv.get("std_auc"),
        "loocv_auc": loocv.get("auc"),
        "loocv_sensitivity": loocv.get("sensitivity"),
        "loocv_specificity": loocv.get("specificity"),
        "best_model_id": best.get("model_id") if best else None,
        "best_model_label": best.get("label") if best else None,
        "best_model_n_features": best.get("n_features") if best else None,
        "best_model_group_cv_auc": best.get("group_cv_mean_auc") if best else None,
        "best_model_loocv_auc": best.get("loocv_auc") if best else None,
        "observed_prevalence": analysis.get("observed_prevalence") or sample_design.get("prevalence"),
        "primary_model_features": analysis.get("primary_model_features"),
        "feature_selection_method": analysis.get("feature_selection_method") or (analysis.get("run_configuration") or {}).get("feature_selection_method"),
        "selected_feature_selection_method_used": analysis.get("selected_feature_selection_method_used") or (analysis.get("run_configuration") or {}).get("selected_feature_selection_method_used"),
        "analysis_results_file": f"analysis_results_threshold_{_safe_label(threshold)}.json",
        "summary_file": f"summary_threshold_{_safe_label(threshold)}.txt",
    }


def _failure_row(threshold: float, code: int, message: str) -> dict:
    return {
        "status": "failed",
        "returncode": int(code),
        "threshold": float(threshold),
        "error": message,
        "analysis_results_file": None,
        "summary_file": None,
    }


def _save_sweep(rows: list[dict], thresholds: list[float], started_at: str, status: str) -> Path:
    sweep = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "started_at": started_at,
        "status": status,
        "thresholds": thresholds,
        "n_thresholds_requested": len(thresholds),
        "n_thresholds_completed": len([r for r in rows if r.get("status") == "ok"]),
        "n_thresholds_failed": len([r for r in rows if r.get("status") == "failed"]),
        "rows": rows,
        "note": (
            "Each threshold was run as a complete independent analysis. "
            "The standard analysis_results.json contains the last successful threshold only; "
            "threshold-specific JSON files are copied next to this summary. "
            "Failed thresholds are kept in the table instead of hiding the problem."
        ),
    }

    path = RESULTS_DIR / "threshold_sweep_results.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(sweep, f, indent=2, ensure_ascii=False)
    return path


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    thresholds = _parse_thresholds(os.environ.get("THERADIOMICS_PRUNING_THRESHOLDS"))

    print("============================================================")
    print("THERADIOMICS PRUNING THRESHOLD SWEEP")
    print("============================================================")
    print("Thresholds requested:", ", ".join(f"{t:.3f}" for t in thresholds))
    print("Parsed threshold count:", len(thresholds))
    print("")

    rows: list[dict] = []
    started_at = datetime.now().isoformat(timespec="seconds")

    for i, threshold in enumerate(thresholds, start=1):
        print("------------------------------------------------------------")
        print(f"SWEEP {i}/{len(thresholds)} | PRUNING_THRESHOLD={threshold:.3f}")
        print("------------------------------------------------------------")

        env = os.environ.copy()
        env["THERADIOMICS_PRUNING_THRESHOLD"] = str(threshold)
        env["PYTHONUNBUFFERED"] = "1"

        code = subprocess.call([sys.executable, "-u", str(SCRIPT)], cwd=str(BASE_DIR), env=env)
        if code != 0:
            message = f"run_analysis.py failed for threshold={threshold:.3f} with code {code}"
            print(f"[ERROR] {message}")
            rows.append(_failure_row(threshold, code, message))
            # Save after every threshold so the dashboard can show partial results.
            _save_sweep(rows, thresholds, started_at, status="partial_failed")
            continue

        analysis = _read_json(ANALYSIS_JSON)
        row = _extract_row(threshold, analysis)
        rows.append(row)

        label = _safe_label(threshold)
        if ANALYSIS_JSON.exists():
            shutil.copy2(ANALYSIS_JSON, RESULTS_DIR / f"analysis_results_threshold_{label}.json")
        if SUMMARY_TXT.exists():
            shutil.copy2(SUMMARY_TXT, RESULTS_DIR / f"summary_threshold_{label}.txt")

        print(
            f"Threshold {threshold:.3f}: features {row.get('features_before_pruning')} -> {row.get('features_after_pruning')} | "
            f"NestedCV {row.get('nested_cv_mean_auc')} | LOOCV {row.get('loocv_auc')} | best {row.get('best_model_label')}"
        )

        # Save after every successful threshold too, so the dashboard never shows
        # an old/partial file without an explicit status.
        _save_sweep(rows, thresholds, started_at, status="running")

    failed = len([r for r in rows if r.get("status") == "failed"])
    status = "ok" if failed == 0 else "partial_failed"
    path = _save_sweep(rows, thresholds, started_at, status=status)

    print("")
    print("Saved threshold sweep summary:")
    print(path)
    print(f"Sweep status: {status}")
    print(f"Completed: {len(rows) - failed}/{len(thresholds)} | Failed: {failed}")

    # Return 0 even for partial_failed so the dashboard refreshes and shows which
    # threshold failed instead of silently displaying an old sweep.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
