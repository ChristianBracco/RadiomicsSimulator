import { existsSync, mkdirSync, copyFileSync, readdirSync, statSync, readFileSync } from "fs";
import { join, dirname, resolve, basename } from "path";

type LogEntry = {
  time: string;
  stream: "system" | "stdout" | "stderr";
  line: string;
};

type RunState = {
  running: boolean;
  proc: ReturnType<typeof Bun.spawn> | null;
  returncode: number | null;
  startedAt: string | null;
  endedAt: string | null;
  logs: LogEntry[];
  lastError: string | null;
};

const ROOT_DIR = dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1"));

/**
 * ROOT_DIR deve essere la cartella che contiene server.ts.
 *
 * Nel tuo Mac:
 *   /Users/cbracco/Downloads/RadiomicsSimulator-main
 *
 * Quindi:
 *   data/uploads  = ROOT_DIR/data/uploads
 *   python        = ROOT_DIR/python
 *   results       = ROOT_DIR/python/results
 *   models        = ROOT_DIR/models
 *
 * Le patch precedenti usavano resolve(ROOT_DIR, "..") per DATA/MODELS:
 * su Mac finiva in /Users/cbracco/Downloads/models, cioè cartella sbagliata.
 */
const SERVER_DIR = ROOT_DIR;
const PYTHON_DIR = resolve(ROOT_DIR, "python");
const DATA_UPLOAD_DIR = resolve(ROOT_DIR, "data", "uploads");
const RESULTS_DIR = resolve(PYTHON_DIR, "results");
const MODELS_DIR = resolve(ROOT_DIR, "models");

const DATASET_NAME = "Features_all.xlsx";
const DEFAULT_PORT = Number(Bun.env.PORT ?? "8765");
const PYTHON_EXE = Bun.env.PYTHON_EXE ?? "python";

const DASHBOARD_CANDIDATES = [
  resolve(ROOT_DIR, "results", "THERADIOMICS_results_dashboard_backend.html"),
  resolve(ROOT_DIR, "public", "dashboard.html"),
  resolve(RESULTS_DIR, "THERADIOMICS_results_dashboard_backend.html"),
];

const runState: RunState = {
  running: false,
  proc: null,
  returncode: null,
  startedAt: null,
  endedAt: null,
  logs: [],
  lastError: null,
};

function nowIso(): string {
  return new Date().toISOString().replace(/\.\d{3}Z$/, "");
}

function appendLog(stream: LogEntry["stream"], line: string): void {
  runState.logs.push({
    time: nowIso(),
    stream,
    line: line.replace(/[\r\n]+$/, ""),
  });

  if (runState.logs.length > 5000) {
    runState.logs = runState.logs.slice(-5000);
  }
}

function jsonResponse(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data, null, 2), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "access-control-allow-origin": "*",
      "access-control-allow-methods": "GET,POST,OPTIONS",
      "access-control-allow-headers": "content-type",
    },
  });
}

function textResponse(text: string, status = 200): Response {
  return new Response(text, {
    status,
    headers: {
      "content-type": "text/plain; charset=utf-8",
      "access-control-allow-origin": "*",
    },
  });
}

function safeReadJson(path: string): unknown | null {
  try {
    if (!existsSync(path)) return null;

    /**
     * Do NOT use Bun.file(path).textSync().
     *
     * On some Bun/macOS versions FileBlob has async .text(),
     * but no .textSync(). In that case /api/results returned:
     *
     *   {"_error":"TypeError: Bun.file(...).textSync is not a function"}
     *
     * which is truthy, so the dashboard assigned state.analysis,
     * but state.analysis.patients was undefined.
     *
     * readFileSync is stable here and returns the real JSON.
     */
    const text = readFileSync(path, "utf-8");
    return JSON.parse(text);
  } catch (err) {
    return {
      _error: String(err),
      _path: path,
    };
  }
}

async function readStreamLines(
  stream: ReadableStream<Uint8Array> | null,
  streamName: "stdout" | "stderr",
): Promise<void> {
  if (!stream) return;

  const reader = stream
    .pipeThrough(new TextDecoderStream())
    .getReader();

  let buffer = "";

  try {
    while (true) {
      const { value, done } = await reader.read();

      if (done) break;

      buffer += value;
      const lines = buffer.split(/\r?\n/);
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        appendLog(streamName, line);
      }
    }

    if (buffer.trim().length > 0) {
      appendLog(streamName, buffer);
    }
  } catch (err) {
    appendLog("stderr", `[${streamName} reader error] ${String(err)}`);
  }
}

function findDashboardPath(): string | null {
  for (const candidate of DASHBOARD_CANDIDATES) {
    if (existsSync(candidate)) return candidate;
  }

  return null;
}

async function serveDashboard(): Promise<Response> {
  const path = findDashboardPath();

  if (!path) {
    return textResponse(
      "Dashboard HTML non trovata. Copia results/THERADIOMICS_results_dashboard_backend.html accanto a server.ts oppure in python/results.",
      404,
    );
  }

  return new Response(Bun.file(path), {
    headers: {
      "content-type": "text/html; charset=utf-8",
    },
  });
}

function getStatusPayload() {
  return {
    root_dir: ROOT_DIR,
    server_dir: SERVER_DIR,
    python_dir: PYTHON_DIR,
    upload_dir: DATA_UPLOAD_DIR,
    results_dir: RESULTS_DIR,
    models_dir: MODELS_DIR,
    dataset_exists: existsSync(resolve(DATA_UPLOAD_DIR, DATASET_NAME)),
    analysis_results_exists: existsSync(resolve(RESULTS_DIR, "analysis_results.json")),
    candidate_model_comparison_exists: existsSync(resolve(RESULTS_DIR, "candidate_model_comparison.json")),
    nested_cv_fold_details_exists: existsSync(resolve(RESULTS_DIR, "nested_cv_fold_details.json")),
    threshold_sweep_results_exists: existsSync(resolve(RESULTS_DIR, "threshold_sweep_results.json")),
    files: {
      dataset: resolve(DATA_UPLOAD_DIR, DATASET_NAME),
      analysis_results: resolve(RESULTS_DIR, "analysis_results.json"),
      candidate_model_comparison: resolve(RESULTS_DIR, "candidate_model_comparison.json"),
      nested_cv_fold_details: resolve(RESULTS_DIR, "nested_cv_fold_details.json"),
      threshold_sweep_results: resolve(RESULTS_DIR, "threshold_sweep_results.json"),
      final_model: resolve(MODELS_DIR, "theradiomics_threshold_085.joblib"),
      final_model_metadata: resolve(MODELS_DIR, "theradiomics_threshold_085_metadata.json"),
      candidate_models_metadata: resolve(MODELS_DIR, "theradiomics_threshold_085_candidate_models_metadata.json"),
    },
    run: {
      running: runState.running,
      returncode: runState.returncode,
      started_at: runState.startedAt,
      ended_at: runState.endedAt,
      log_count: runState.logs.length,
      last_error: runState.lastError,
    },
  };
}

async function handleUploadDataset(req: Request): Promise<Response> {
  mkdirSync(DATA_UPLOAD_DIR, {
    recursive: true,
  });

  const form = await req.formData();
  const file = form.get("file");

  if (!(file instanceof File)) {
    return jsonResponse({
      ok: false,
      error: "Campo multipart 'file' mancante.",
    }, 400);
  }

  const target = resolve(DATA_UPLOAD_DIR, DATASET_NAME);
  let backup: string | null = null;

  if (existsSync(target)) {
    backup = resolve(DATA_UPLOAD_DIR, "Features_all.previous.xlsx");
    copyFileSync(target, backup);
  }

  await Bun.write(target, file);

  return jsonResponse({
    ok: true,
    uploaded_name: file.name,
    saved_as: target,
    bytes: file.size,
    previous_backup: backup,
  });
}


function parseThresholdNumber(value: unknown): number | null {
  if (value === null || value === undefined) return null;
  const n = Number(String(value).replace(",", "."));
  if (!Number.isFinite(n) || n <= 0 || n >= 1) return null;
  return n;
}

function parseThresholdList(value: unknown): number[] {
  if (Array.isArray(value)) {
    return Array.from(new Set(value.map(parseThresholdNumber).filter((x): x is number => x !== null))).sort((a, b) => a - b);
  }

  if (typeof value === "string") {
    return Array.from(new Set(
      value
        .replace(/;/g, ",")
        .split(",")
        .map((x) => parseThresholdNumber(x.trim()))
        .filter((x): x is number => x !== null)
    )).sort((a, b) => a - b);
  }

  const single = parseThresholdNumber(value);
  return single === null ? [] : [single];
}


function parseFeatureSelectionMethod(value: unknown): string | null {
  if (value === null || value === undefined) return null;
  const raw = String(value).trim().toLowerCase().replace(/[\s-]+/g, "_");
  const aliases: Record<string, string> = {
    l1: "lasso",
    logistic_l1: "lasso",
    l1_logistic: "lasso",
    point_biserial: "pearson",
    pointbiserial: "pearson",
    corr: "pearson",
    correlation: "pearson",
    univariate_auc: "auc",
    roc_auc: "auc",
    mann_whitney: "mannwhitney",
    mann_whitney_u: "mannwhitney",
    mw: "mannwhitney",
    mwu: "mannwhitney",
    mi: "mutual_info",
    mutual_information: "mutual_info",
  };
  const normalized = aliases[raw] ?? raw;
  const allowed = new Set(["lasso", "pearson", "spearman", "auc", "mannwhitney", "mutual_info"]);
  return allowed.has(normalized) ? normalized : null;
}

async function readRunOptions(req: Request): Promise<Record<string, unknown>> {
  const contentType = req.headers.get("content-type") || "";
  if (!contentType.includes("application/json")) return {};

  try {
    const body = await req.json();
    return body && typeof body === "object" ? body as Record<string, unknown> : {};
  } catch {
    return {};
  }
}

async function startPythonAnalysis(req: Request): Promise<Response> {
  if (runState.running) {
    return jsonResponse({
      ok: true,
      already_running: true,
      message: "Un'analisi è già in esecuzione.",
      started_at: runState.startedAt,
      log_count: runState.logs.length,
    });
  }

  const options = await readRunOptions(req);
  const thresholds = parseThresholdList(options.pruning_thresholds ?? options.pruningThresholds);
  const singleThreshold = parseThresholdNumber(options.pruning_threshold ?? options.pruningThreshold);

  const runSweep = thresholds.length > 1;
  const scriptName = runSweep ? "run_threshold_sweep.py" : "run_analysis.py";
  const script = resolve(PYTHON_DIR, scriptName);

  if (!existsSync(script)) {
    return jsonResponse({
      ok: false,
      error: `${scriptName} non trovato: ${script}`,
    }, 404);
  }

  const envOverrides: Record<string, string> = {};

  if (runSweep) {
    envOverrides.THERADIOMICS_PRUNING_THRESHOLDS = thresholds.join(",");
  } else if (singleThreshold !== null) {
    envOverrides.THERADIOMICS_PRUNING_THRESHOLD = String(singleThreshold);
  } else if (thresholds.length === 1) {
    envOverrides.THERADIOMICS_PRUNING_THRESHOLD = String(thresholds[0]);
  }

  const topN = Number(options.top_n_final_model_features ?? options.topNFinalModelFeatures ?? NaN);
  if (Number.isFinite(topN) && topN >= 1) {
    envOverrides.THERADIOMICS_TOP_N_FINAL_MODEL_FEATURES = String(Math.floor(topN));
    envOverrides.THERADIOMICS_PRIMARY_MODEL_FEATURES = String(Math.floor(topN));
  }

  const nSim = Number(options.n_sample_size_simulations ?? options.nSampleSizeSimulations ?? NaN);
  if (Number.isFinite(nSim) && nSim >= 10) {
    envOverrides.THERADIOMICS_N_SAMPLE_SIZE_SIMULATIONS = String(Math.floor(nSim));
  }


  const featureSelectionMethod = parseFeatureSelectionMethod(
    options.feature_selection_method ?? options.featureSelectionMethod
  );
  if (featureSelectionMethod) {
    envOverrides.THERADIOMICS_FEATURE_SELECTION_METHOD = featureSelectionMethod;
  }

  runState.running = true;
  runState.proc = null;
  runState.returncode = null;
  runState.startedAt = nowIso();
  runState.endedAt = null;
  runState.logs = [];
  runState.lastError = null;

  appendLog("system", `Starting ${scriptName}...`);
  appendLog("system", `cwd: ${PYTHON_DIR}`);
  appendLog("system", `command: ${PYTHON_EXE} -u ${script}`);
  appendLog("system", `run options: ${JSON.stringify({runSweep, thresholds, singleThreshold, envOverrides})}`);

  try {
    const proc = Bun.spawn([PYTHON_EXE, "-u", script], {
      cwd: PYTHON_DIR,
      stdout: "pipe",
      stderr: "pipe",
      env: {
        ...Bun.env,
        ...envOverrides,
        PYTHONUNBUFFERED: "1",
      },
    });

    runState.proc = proc;

    readStreamLines(proc.stdout, "stdout");
    readStreamLines(proc.stderr, "stderr");

    proc.exited
      .then((code) => {
        runState.running = false;
        runState.returncode = code;
        runState.endedAt = nowIso();
        runState.proc = null;
        appendLog("system", `Process finished with return code ${code}`);
      })
      .catch((err) => {
        runState.running = false;
        runState.returncode = -1;
        runState.endedAt = nowIso();
        runState.proc = null;
        runState.lastError = String(err);
        appendLog("stderr", String(err));
      });

    return jsonResponse({
      ok: true,
      started: true,
      started_at: runState.startedAt,
      pid: proc.pid,
      script: scriptName,
      run_sweep: runSweep,
      thresholds,
      pruning_threshold: envOverrides.THERADIOMICS_PRUNING_THRESHOLD ?? null,
      feature_selection_method: envOverrides.THERADIOMICS_FEATURE_SELECTION_METHOD ?? null,
    });
  } catch (err) {
    runState.running = false;
    runState.returncode = -1;
    runState.endedAt = nowIso();
    runState.proc = null;
    runState.lastError = String(err);
    appendLog("stderr", String(err));

    return jsonResponse({
      ok: false,
      error: String(err),
    }, 500);
  }
}

function runStatus(url: URL): Response {
  const since = Number(url.searchParams.get("since") ?? "0");
  const safeSince = Number.isFinite(since) && since >= 0 ? Math.floor(since) : 0;

  return jsonResponse({
    ok: true,
    running: runState.running,
    returncode: runState.returncode,
    started_at: runState.startedAt,
    ended_at: runState.endedAt,
    logs: runState.logs.slice(safeSince),
    next_since: runState.logs.length,
    last_error: runState.lastError,
  });
}

function stopRun(): Response {
  if (!runState.proc) {
    return jsonResponse({
      ok: true,
      message: "Nessun processo in esecuzione.",
    });
  }

  try {
    runState.proc.kill();
    appendLog("system", "Stop requested from dashboard.");
    return jsonResponse({
      ok: true,
      message: "Stop richiesto.",
    });
  } catch (err) {
    return jsonResponse({
      ok: false,
      error: String(err),
    }, 500);
  }
}


function resultFileResponse(filename: string): Response {
  const path = resolve(RESULTS_DIR, filename);

  if (!existsSync(path)) {
    return jsonResponse({
      ok: false,
      error: "Result file not found",
      requested_file: filename,
      expected_path: path,
      results_dir: RESULTS_DIR,
    }, 404);
  }

  const lower = filename.toLowerCase();
  const contentType =
    lower.endsWith(".json") ? "application/json; charset=utf-8" :
    lower.endsWith(".txt") ? "text/plain; charset=utf-8" :
    "application/octet-stream";

  return new Response(Bun.file(path), {
    headers: {
      "content-type": contentType,
      "cache-control": "no-store, no-cache, must-revalidate",
      "access-control-allow-origin": "*",
    },
  });
}

function resultAliasFilename(pathname: string): string | null {
  const aliases: Record<string, string> = {
    "/analysis_results.json": "analysis_results.json",
    "/candidate_model_comparison.json": "candidate_model_comparison.json",
    "/nested_cv_fold_details.json": "nested_cv_fold_details.json",
    "/summary.txt": "summary.txt",
    "/threshold_sweep_results.json": "threshold_sweep_results.json",
    "/results/threshold_sweep_results.json": "threshold_sweep_results.json",
    "/python/results/threshold_sweep_results.json": "threshold_sweep_results.json",

    "/results/analysis_results.json": "analysis_results.json",
    "/results/candidate_model_comparison.json": "candidate_model_comparison.json",
    "/results/nested_cv_fold_details.json": "nested_cv_fold_details.json",
    "/results/summary.txt": "summary.txt",

    "/python/results/analysis_results.json": "analysis_results.json",
    "/python/results/candidate_model_comparison.json": "candidate_model_comparison.json",
    "/python/results/nested_cv_fold_details.json": "nested_cv_fold_details.json",
    "/python/results/summary.txt": "summary.txt",
  };

  return aliases[pathname] ?? null;
}


async function getResults(): Promise<Response> {
  const summaryPath = resolve(RESULTS_DIR, "summary.txt");
  let summary: string | null = null;

  try {
    if (existsSync(summaryPath)) {
      summary = await Bun.file(summaryPath).text();
    }
  } catch {
    summary = null;
  }

  return jsonResponse({
    ok: true,
    run: {
      running: runState.running,
      returncode: runState.returncode,
      started_at: runState.startedAt,
      ended_at: runState.endedAt,
      last_error: runState.lastError,
    },
    analysis: safeReadJson(resolve(RESULTS_DIR, "analysis_results.json")),
    candidate: safeReadJson(resolve(RESULTS_DIR, "candidate_model_comparison.json")),
    nested: safeReadJson(resolve(RESULTS_DIR, "nested_cv_fold_details.json")),
    sweep: safeReadJson(resolve(RESULTS_DIR, "threshold_sweep_results.json")),
    summary,
    paths: {
      analysis_results: resolve(RESULTS_DIR, "analysis_results.json"),
      candidate_model_comparison: resolve(RESULTS_DIR, "candidate_model_comparison.json"),
      nested_cv_fold_details: resolve(RESULTS_DIR, "nested_cv_fold_details.json"),
      threshold_sweep_results: resolve(RESULTS_DIR, "threshold_sweep_results.json"),
      summary: summaryPath,
    },
  });
}

async function listModels(): Promise<Response> {
  const metadataCandidates = [
    resolve(MODELS_DIR, "theradiomics_threshold_085_metadata.json"),
    resolve(MODELS_DIR, "theradiomics_threshold_085_candidate_models_metadata.json"),
  ];

  const metadata = metadataCandidates
    .filter((p) => existsSync(p))
    .map((p) => ({
      path: p,
      name: basename(p),
      json: safeReadJson(p),
    }));

  let files: Array<{
    name: string;
    path: string;
    size_bytes: number;
    modified_at: string;
    kind: "joblib" | "json" | "other";
  }> = [];

  try {
    if (existsSync(MODELS_DIR)) {
      files = readdirSync(MODELS_DIR)
        .map((name) => {
          const p = resolve(MODELS_DIR, name);
          const st = statSync(p);
          const lower = name.toLowerCase();

          const kind =
            lower.endsWith(".joblib") ? "joblib" :
            lower.endsWith(".json") ? "json" :
            "other";

          return {
            name,
            path: p,
            size_bytes: st.size,
            modified_at: st.mtime.toISOString(),
            kind,
          };
        })
        .sort((a, b) => a.name.localeCompare(b.name));
    }
  } catch (err) {
    return jsonResponse({
      ok: false,
      error: String(err),
      models_dir: MODELS_DIR,
      metadata,
      files,
    }, 500);
  }

  return jsonResponse({
    ok: true,
    models_dir: MODELS_DIR,
    metadata,
    files,
    joblib_files: files.filter((f) => f.kind === "joblib"),
    json_files: files.filter((f) => f.kind === "json"),
  });
}

async function serveStatic(pathname: string): Promise<Response | null> {
  const allowedRoots = [
    resolve(ROOT_DIR, "public"),
    resolve(ROOT_DIR, "results"),
    RESULTS_DIR,
  ];

  for (const root of allowedRoots) {
    const clean = pathname.replace(/^\/+/, "");
    const full = resolve(root, clean.replace(/^public\//, "").replace(/^results\//, ""));

    if (!full.startsWith(root)) continue;
    if (!existsSync(full)) continue;

    const ext = full.toLowerCase().split(".").pop();
    const contentType =
      ext === "html" ? "text/html; charset=utf-8" :
      ext === "css" ? "text/css; charset=utf-8" :
      ext === "js" ? "text/javascript; charset=utf-8" :
      ext === "json" ? "application/json; charset=utf-8" :
      "application/octet-stream";

    return new Response(Bun.file(full), {
      headers: {
        "content-type": contentType,
      },
    });
  }

  return null;
}

Bun.serve({
  port: DEFAULT_PORT,

  async fetch(req) {
    const url = new URL(req.url);
    const path = url.pathname;

    if (req.method === "OPTIONS") {
      return new Response(null, {
        headers: {
          "access-control-allow-origin": "*",
          "access-control-allow-methods": "GET,POST,OPTIONS",
          "access-control-allow-headers": "content-type",
        },
      });
    }

    try {

      if (req.method === "GET") {
        const alias = resultAliasFilename(path);
        if (alias) {
          return resultFileResponse(alias);
        }
      }

      if (req.method === "GET" && path === "/") {
        return await serveDashboard();
      }



      if (req.method === "GET" && path === "/api/debug-json") {
        const analysis = safeReadJson(resolve(RESULTS_DIR, "analysis_results.json"));
        const candidate = safeReadJson(resolve(RESULTS_DIR, "candidate_model_comparison.json"));
        const nested = safeReadJson(resolve(RESULTS_DIR, "nested_cv_fold_details.json"));

        return jsonResponse({
          ok: true,
          results_dir: RESULTS_DIR,
          analysis_type: analysis === null ? "null" : Array.isArray(analysis) ? "array" : typeof analysis,
          analysis_is_error: !!(analysis && typeof analysis === "object" && "_error" in analysis),
          analysis_keys: analysis && typeof analysis === "object" ? Object.keys(analysis as Record<string, unknown>).slice(0, 30) : [],
          analysis_patients: analysis && typeof analysis === "object" ? (analysis as Record<string, unknown>).patients : null,
          analysis_lesions: analysis && typeof analysis === "object" ? (analysis as Record<string, unknown>).lesions : null,
          candidate_type: candidate === null ? "null" : Array.isArray(candidate) ? "array" : typeof candidate,
          candidate_is_error: !!(candidate && typeof candidate === "object" && "_error" in candidate),
          candidate_keys: candidate && typeof candidate === "object" ? Object.keys(candidate as Record<string, unknown>).slice(0, 30) : [],
          nested_type: nested === null ? "null" : Array.isArray(nested) ? "array" : typeof nested,
          nested_is_error: !!(nested && typeof nested === "object" && "_error" in nested),
          nested_keys: nested && typeof nested === "object" ? Object.keys(nested as Record<string, unknown>).slice(0, 30) : [],
        });
      }

      if (req.method === "GET" && path === "/api/debug-paths") {
        return jsonResponse({
          ok: true,
          note: "These are the real paths used by Bun. Result JSON aliases /analysis_results.json and /results/analysis_results.json are served from python/results.",
          root_dir: ROOT_DIR,
          python_dir: PYTHON_DIR,
          upload_dir: DATA_UPLOAD_DIR,
          results_dir: RESULTS_DIR,
          models_dir: MODELS_DIR,
          files: {
            analysis_results: {
              path: resolve(RESULTS_DIR, "analysis_results.json"),
              exists: existsSync(resolve(RESULTS_DIR, "analysis_results.json")),
            },
            candidate_model_comparison: {
              path: resolve(RESULTS_DIR, "candidate_model_comparison.json"),
              exists: existsSync(resolve(RESULTS_DIR, "candidate_model_comparison.json")),
            },
            nested_cv_fold_details: {
              path: resolve(RESULTS_DIR, "nested_cv_fold_details.json"),
              exists: existsSync(resolve(RESULTS_DIR, "nested_cv_fold_details.json")),
            },
            summary: {
              path: resolve(RESULTS_DIR, "summary.txt"),
              exists: existsSync(resolve(RESULTS_DIR, "summary.txt")),
            },
            threshold_sweep_results: {
              path: resolve(RESULTS_DIR, "threshold_sweep_results.json"),
              exists: existsSync(resolve(RESULTS_DIR, "threshold_sweep_results.json")),
            },
            final_model: {
              path: resolve(MODELS_DIR, "theradiomics_threshold_085.joblib"),
              exists: existsSync(resolve(MODELS_DIR, "theradiomics_threshold_085.joblib")),
            },
          },
          aliases: [
            "/api/results",
            "/analysis_results.json",
            "/results/analysis_results.json",
            "/candidate_model_comparison.json",
            "/results/candidate_model_comparison.json",
            "/nested_cv_fold_details.json",
            "/results/nested_cv_fold_details.json",
            "/summary.txt",
            "/results/summary.txt",
            "/threshold_sweep_results.json",
            "/results/threshold_sweep_results.json",
          ],
        });
      }

      if (req.method === "GET" && path === "/api/status") {
        return jsonResponse(getStatusPayload());
      }

      if (req.method === "POST" && (path === "/api/upload-dataset" || path === "/api/upload")) {
        return await handleUploadDataset(req);
      }

      if (req.method === "POST" && path === "/api/run-analysis") {
        return await startPythonAnalysis(req);
      }

      if (req.method === "GET" && path === "/api/run-status") {
        return runStatus(url);
      }

      if (req.method === "POST" && path === "/api/stop-run") {
        return stopRun();
      }

      if (req.method === "GET" && path === "/api/results") {
        return await getResults();
      }

      if (req.method === "GET" && path === "/api/models") {
        return await listModels();
      }

      if (req.method === "GET") {
        const staticResponse = await serveStatic(path);
        if (staticResponse) return staticResponse;
      }

      return jsonResponse({
        ok: false,
        error: "Route non trovata.",
        method: req.method,
        path,
      }, 404);
    } catch (err) {
      return jsonResponse({
        ok: false,
        error: String(err),
        stack: err instanceof Error ? err.stack : null,
      }, 500);
    }
  },
});

console.log("============================================================");
console.log("THERADIOMICS Bun Python Bridge - stable v14 + result aliases");
console.log("============================================================");
console.log(`Dashboard: http://127.0.0.1:${DEFAULT_PORT}`);
console.log(`ROOT_DIR:   ${ROOT_DIR}`);
console.log(`PYTHON_DIR: ${PYTHON_DIR}`);
console.log(`UPLOAD_DIR: ${DATA_UPLOAD_DIR}`);
console.log(`RESULTS:    ${RESULTS_DIR}`);
console.log("============================================================");
