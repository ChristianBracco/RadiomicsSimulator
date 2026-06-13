let state = {analysis:null,candidate:null,nested:null,sweep:null,summary:null};
const API = location.origin && location.protocol.startsWith("http") ? location.origin : "http://127.0.0.1:8765";

function log(msg){
  const el=document.getElementById("console");
  el.textContent += "\\n" + msg;
  el.scrollTop = el.scrollHeight;
}
function clearLog(msg=""){document.getElementById("console").textContent=msg || "Pronto."}
function setProgress(p){document.getElementById("progressBar").style.width = `${Math.max(0,Math.min(100,p))}%`}
function fmt(x,d=3){if(x===undefined||x===null||!Number.isFinite(Number(x)))return "—";return Number(x).toFixed(d)}
function pct(x,d=1){if(x===undefined||x===null||!Number.isFinite(Number(x)))return "—";return (Number(x)*100).toFixed(d)+"%"}

function escHtml(s){return String(s??"").replace(/[&<>"']/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]))}
function helpTip(text){return `<span class="help" tabindex="0" data-tip="${escHtml(text)}">?</span>`}
function thTip(label,text){return `${label}${helpTip(text)}`}
function designLabelHuman(label){
  const map={
    analytically_supported:"supportato",
    predictor_ok_event_rate_uncertain:"predittori ok, event-rate incerto",
    exploratory_only:"solo esplorativo",
    legacy:"legacy"
  };
  return map[label]||label||"legacy";
}
function designLabelTip(label){
  const map={
    analytically_supported:"Per questo N sono rispettati sia il guardrail sull'event-rate/prevalenza sia quello sul numero di predittori ammessi dallo shrinkage.",
    predictor_ok_event_rate_uncertain:"Il numero di predittori è compatibile con lo shrinkage impostato, ma il campione non basta ancora per stimare la prevalenza/event-rate con il margine scelto.",
    exploratory_only:"Il campione è utile solo come esplorazione: non soddisfa almeno uno dei guardrail analitici.",
    legacy:"Risultato prodotto da una vecchia versione del JSON senza guardrail. Rilancia l'analisi."
  };
  return map[label]||"Etichetta di disegno non riconosciuta.";
}
function renderSampleExplanation(a){
  const box=document.getElementById("sampleExplanation");
  if(!box)return;
  const sample=a?.sample_size||{};
  const firstArr=Object.values(sample).find(v=>Array.isArray(v)&&v.length) || [];
  const first=firstArr[0]||{};
  const design=a?.sample_size_design||{};
  const prevalence=design.prevalence ?? a?.design_prevalence ?? a?.observed_prevalence ?? first.prevalence;
  const riskMargin=first.risk_margin;
  const r2=first.r2_cs_adj;
  const k=first.final_predictors_k ?? a?.primary_model_features;
  const shrinkage=design.target_shrinkage ?? first.target_shrinkage;
  const delta=design.delta_nagelkerke ?? first.delta_nagelkerke;
  const nsim=first.n_simulations_requested;
  const evalMode=first.evaluation_mode;
  box.innerHTML=`
    <strong>Come leggere questa sezione.</strong><br>
    La parte AUC è ancora la simulazione Monte Carlo classica: per ogni N genera dati sintetici coerenti con l'AUC target e misura quante simulazioni superano la soglia di successo. I nuovi campi non cambiano l'AUC: aggiungono un controllo metodologico su prevalenza/event-rate e numero massimo di predittori sostenibile.
    <div class="explainGrid">
      <div class="explainCard"><div class="k">Prevalenza usata</div><div class="v">${pct(prevalence)}</div><div class="small">Letta dal dataset o dal parametro di disegno.</div></div>
      <div class="explainCard"><div class="k">Margine event-rate</div><div class="v">±${pct(riskMargin)}</div><div class="small">Usato nella colonna N event-rate della tabella Monte Carlo.</div></div>
      <div class="explainCard"><div class="k">R² Cox-Snell adj</div><div class="v">${fmt(r2,3)}</div><div class="small">Ipotesi sulla forza attesa del modello.</div></div>
      <div class="explainCard"><div class="k">Predittori finali k</div><div class="v">${k??"—"}</div><div class="small">Numero di variabili che vogliamo nel modello finale.</div></div>
      <div class="explainCard"><div class="k">Shrinkage target</div><div class="v">${fmt(shrinkage,2)}</div><div class="small">Valore conservativo: 0.90 limita l'overfitting.</div></div>
      <div class="explainCard"><div class="k">δ Nagelkerke</div><div class="v">${fmt(delta,2)}</div><div class="small">Massima differenza accettata tra R² apparente e corretto.</div></div>
      <div class="explainCard"><div class="k">Simulazioni per N</div><div class="v">${nsim??"—"}</div><div class="small">Aumentabile in run_analysis.py per risultati più stabili.</div></div>
      <div class="explainCard"><div class="k">Modalità</div><div class="v">${evalMode||"—"}</div><div class="small">Biomarcatore sintetico calibrato sull'AUC target.</div></div>
    </div>
    <div class="small" style="margin-top:10px">
      Nota: questi valori non sono hardcoded nell'HTML. La dashboard li legge dal JSON prodotto da <code>run_analysis.py</code>. Al momento si modificano nei parametri Python e poi si rilancia l'analisi; la dashboard è volutamente read-only.
    </div>`;
}
function aucClass(x){x=Number(x); if(!Number.isFinite(x))return ""; if(x>=.8)return "good"; if(x>=.7)return "warn"; return "bad"}
function getGroupAuc(m){return m?.group_cv?.mean_auc ?? m?.groupcv_auc ?? m?.group_cv_auc ?? m?.cv_auc ?? null}
function getLoocvAuc(m){return m?.loocv?.auc ?? m?.loocv_auc ?? null}
function getSensitivity(m){return m?.group_cv?.sensitivity ?? m?.loocv?.sensitivity ?? null}
function getSpecificity(m){return m?.group_cv?.specificity ?? m?.loocv?.specificity ?? null}
function short(s){return String(s||"").replace(/^original__/,"").replaceAll("__"," · ")}
function models(){
  if (Array.isArray(state.candidate)) return state.candidate;
  if (Array.isArray(state.candidate?.models)) return state.candidate.models;
  if (Array.isArray(state.candidate?.summary)) return state.candidate.summary;
  if (Array.isArray(state.candidate?.candidate_models)) return state.candidate.candidate_models;
  if (Array.isArray(state.candidate?.results)) return state.candidate.results;
  return [];
}

function metric(label,value,note,klass=""){
  return `<div class="card"><div class="label">${label}</div><div class="value ${klass}">${value}</div><div class="note">${note||""}</div></div>`
}

function render(){
  const a=state.analysis||{};
  const c=state.candidate||{};
  const ok=models().filter(m=>m.status==="ok");
  const best=[...ok].sort((x,y)=>score(y)-score(x))[0];

  document.getElementById("cards").innerHTML=[
    metric("Pazienti",a.patients??"—","unità statistica", "good"),
    metric("Lesioni",a.lesions??"—","righe caricate"),
    metric("Responder",a.responders??"—","classe positiva", "warn"),
    metric("Non responder",a.non_responders??"—","classe negativa"),
    metric("Feature",`${a.features_before_pruning??"—"} → ${a.features_after_pruning??"—"}`,`rimosse ${a.removed_features_count??"—"}`,"cyan"),
    metric("Nested CV",fmt(a.cv?.mean_auc),`± ${fmt(a.cv?.std_auc)}`,aucClass(a.cv?.mean_auc)),
    metric("LOOCV",fmt(a.loocv?.auc),`sens ${fmt(a.loocv?.sensitivity)} · spec ${fmt(a.loocv?.specificity)}`,aucClass(a.loocv?.auc)),
    metric("Permutation p",fmt(a.permutation?.p_value,3),`max random ${fmt(a.permutation?.max_random_auc)}`,(a.permutation?.p_value<.05?"good":"warn")),
    metric("Best model",best?.label??"—",best?`Group ${fmt(getGroupAuc(best))} · LOOCV ${fmt(getLoocvAuc(best))}`:"—","good"),
    metric("Modelli OK",ok.length,`${models().length} totali`)
  ].join("");

  renderBootstrap(a);
  renderModels();
  renderThresholdSweep();
  renderFolds(a);
  renderSample(a);
  renderGuardrails(a.sample_size_design||{});
  renderVerdict(best);
  document.getElementById("rawA").textContent=JSON.stringify(state.analysis,null,2);
  document.getElementById("rawC").textContent=JSON.stringify(state.candidate,null,2);
}

function score(m){
  const g=Number(getGroupAuc(m)||0), l=Number(getLoocvAuc(m)||0), n=Number(m.n_features||m.features?.length||99);
  return ((g+l)/2)-Math.max(0,n-1)*.002;
}

function renderVerdict(best){
  if(!best){document.getElementById("verdict").innerHTML="Nessun modello OK disponibile. Rilancia il backend con la patch aggiornata.";return}
  const b1=models().find(m=>m.model_id?.includes("one_feature")||m.label?.includes("1 feature"));
  const b3=models().find(m=>m.model_id?.includes("three_feature")||m.label?.includes("3 feature - top bootstrap"));
  document.getElementById("verdict").innerHTML=`
    <strong>Modello migliore per score interno:</strong> ${best.label}<br>
    GroupCV AUC ${fmt(getGroupAuc(best))}, LOOCV AUC ${fmt(getLoocvAuc(best))}, feature ${best.n_features}.
    <br><br>
    ${b1?.status==="ok" ? `Il modello a 1 feature (${b1.features?.join(", ")}) ha GroupCV ${fmt(getGroupAuc(b1))} e LOOCV ${fmt(getLoocvAuc(b1))}.<br>` : ""}
    ${b3?.status==="ok" ? `Il modello a 3 feature bootstrap-stable ha GroupCV ${fmt(getGroupAuc(b3))} e LOOCV ${fmt(getLoocvAuc(b3))}.<br>` : ""}
    <br>Con classi sbilanciate, guarda sempre anche sensibilità e specificità, non solo AUC.
  `;
}

function renderBootstrap(a){
  const boot=(a.bootstrap||[]).slice(0,12);
  const max=Math.max(...boot.map(x=>Number(x.frequency||0)),.001);
  document.getElementById("bootstrapBars").innerHTML=boot.map(x=>`
    <div class="bar" title="${x.feature}">
      <div class="barName">${short(x.feature)}</div>
      <div class="track"><div class="fill" style="width:${Number(x.frequency||0)/max*100}%"></div></div>
      <div class="small">${pct(x.frequency)}</div>
    </div>
  `).join("") || `<div class="small">Bootstrap non disponibile.</div>`;
}


function renderThresholdSweep(){
  const table=document.getElementById("thresholdSweepTable");
  if(!table)return;
  const rows=(state.sweep?.rows||[]);
  if(!rows.length){
    table.innerHTML="<thead><tr><th>Stato</th></tr></thead><tbody><tr><td>Nessuno sweep caricato. Inserisci più soglie nel campo sweep e lancia il backend.</td></tr></tbody>";
    return;
  }
  const body=rows.map(r=>`<tr>
    <td><strong>${fmt(r.threshold,2)}</strong></td>
    <td>${r.features_before_pruning??"—"} → ${r.features_after_pruning??"—"}<br><span class="small">rimosse ${r.removed_features_count??"—"}</span></td>
    <td><span class="tag ${aucClass(r.nested_cv_mean_auc)}">${fmt(r.nested_cv_mean_auc)}</span><br><span class="small">± ${fmt(r.nested_cv_std_auc)}</span></td>
    <td><span class="tag ${aucClass(r.loocv_auc)}">${fmt(r.loocv_auc)}</span><br><span class="small">sens ${fmt(r.loocv_sensitivity)} · spec ${fmt(r.loocv_specificity)}</span></td>
    <td><strong>${r.best_model_label||"—"}</strong><br><span class="small">feature ${r.best_model_n_features??"—"} · Group ${fmt(r.best_model_group_cv_auc)} · LOO ${fmt(r.best_model_loocv_auc)}</span></td>
    <td>${r.analysis_results_file?`<span class="tag info">${r.analysis_results_file}</span>`:"—"}</td>
  </tr>`).join("");
  table.innerHTML=`<thead><tr>
    <th>${thTip("Soglia","Soglia di correlation pruning usata. Più bassa = pruning più aggressivo; più alta = più feature residue.")}</th>
    <th>${thTip("Feature dopo pruning","Numero di feature residue dopo il filtro di correlazione. Serve a capire quanto cambia lo spazio delle feature.")}</th>
    <th>${thTip("Nested CV","AUC media e deviazione standard della validazione nested per quella soglia.")}</th>
    <th>${thTip("LOOCV","AUC leave-one-out, più sensibilità e specificità a soglia 0.5.")}</th>
    <th>${thTip("Best candidate","Miglior modello candidato secondo score interno, utile per capire se cambia la firma selezionata.")}</th>
    <th>${thTip("File","JSON salvato per quel run: permette di riaprire il dettaglio della soglia.")}</th>
  </tr></thead><tbody>${body}</tbody>`;
}

function renderModels(){
  const rows=models().map(m=>{
    const status=m.status==="ok"?`<span class="tag good">ok</span>`:m.status==="skipped"?`<span class="tag bad">skipped</span>`:`<span class="tag warn">${m.status||"?"}</span>`;
    const fam=m.family==="bootstrap_stable"?`<span class="tag info">bootstrap</span>`:m.family==="current_run_lasso_topk"?`<span class="tag warn">LASSO top-k</span>`:`<span class="tag">${m.family||"model"}</span>`;
    const feats=(m.features||[]).map(f=>`<div>${f}</div>`).join("") || `<span class="small">${m.skip_reason||"Nessuna feature"}</span>`;
    return `<tr>
      <td>${status}<br>${fam}</td>
      <td><strong>${m.label||m.model_id}</strong><div class="small">${m.description||""}</div></td>
      <td>${m.n_features??(m.features||[]).length}</td>
      <td>${feats}</td>
      <td><span class="tag ${aucClass(getGroupAuc(m))}">${fmt(getGroupAuc(m))}</span></td>
      <td><span class="tag ${aucClass(getLoocvAuc(m))}">${fmt(getLoocvAuc(m))}</span></td>
      <td>${fmt(getSensitivity(m))}</td>
      <td>${fmt(getSpecificity(m))}</td>
      <td>${m.selection_source||"—"}</td>
    </tr>`
  }).join("");
  document.getElementById("modelTable").innerHTML=`
    <thead><tr><th>Status</th><th>Modello</th><th>N</th><th>Feature</th><th>GroupCV AUC</th><th>LOOCV AUC</th><th>Sens</th><th>Spec</th><th>Sorgente</th></tr></thead>
    <tbody>${rows||"<tr><td colspan='9'>Nessun candidato caricato</td></tr>"}</tbody>`;
}

function foldPatients(f){
  if(Array.isArray(f.test_patients))return f.test_patients;
  if(Array.isArray(f.predictions))return f.predictions.map(x=>x.patient).filter(Boolean);
  const all=state.nested?.all_predictions||state.analysis?.cv?.all_predictions||[];
  if(Array.isArray(all))return [...new Set(all.filter(x=>String(x.fold)===String(f.fold)).map(x=>x.patient).filter(Boolean))];
  return [];
}

function renderFolds(a){
  let folds=a.cv?.fold_details||state.nested?.fold_details||[];
  if(!folds.length && Array.isArray(a.cv?.folds)) folds=a.cv.folds.map((auc,i)=>({fold:i+1,auc}));
  document.getElementById("foldGrid").innerHTML=folds.map(f=>{
    const pats=foldPatients(f);
    return `<div class="fold">
      <h3>Fold ${f.fold}</h3>
      <div class="auc ${aucClass(f.auc)}">${fmt(f.auc)}</div>
      <div class="small">Test patients: ${f.n_test??f.test_patients_count??pats.length??"—"}</div>
      <div class="patientList">${pats.map(p=>`<span>${p}</span>`).join("")||"<span>Lista non presente</span>"}</div>
      <div class="chips">${(f.selected_features||f.features||[]).slice(0,8).map(x=>`<span title="${x}">${x}</span>`).join("")}</div>
    </div>`
  }).join("") || `<div class="small">Nested CV debug non caricato.</div>`;
}

function renderSample(a){
  renderSampleExplanation(a);
  const sample=a?.sample_size||{};
  const keys=Object.keys(sample||{});
  const nice={theoretical_auc_070:"Teorica AUC 0.70",theoretical_auc_080:"Teorica AUC 0.80",theoretical_auc_090:"Teorica AUC 0.90",observed_nested_cv_auc:"Osservata Nested CV",observed_loocv_auc:"Osservata LOOCV",auc_070:"Teorica AUC 0.70",auc_080:"Teorica AUC 0.80",auc_090:"Teorica AUC 0.90"};
  const rows=keys.map(k=>{
    const arr=sample[k]||[];
    const hit=arr.find(r=>Number(r.power)>=.8);
    const last=arr[arr.length-1]||{};
    const first=arr[0]||{};
    const designCounts=arr.reduce((acc,r)=>{const lab=r.design_label||"legacy"; acc[lab]=(acc[lab]||0)+1; return acc;},{});
    const designText=Object.entries(designCounts).map(([lab,n])=>`${designLabelHuman(lab)}: ${n}`).join(" · ");
    const trend=arr.map(r=>{
      const lab=r.design_label||"legacy";
      const cls=lab==="analytically_supported"?"good":(lab==="predictor_ok_event_rate_uncertain"?"warn":"bad");
      const title=`N=${r.N}; power=${pct(r.power,0)}; eventi/non-eventi=${r.expected_events??"NA"}/${r.expected_nonevents??"NA"}; k_max=${fmt(r.k_max_raw,2)}; N event-rate=${r.n_required_event_rate??"NA"}; N shrinkage=${r.n_required_shrinkage??"NA"}; ${designLabelTip(lab)}`;
      return `<span class="tag ${cls}" title="${escHtml(title)}">N${r.N}:${pct(r.power,0)}</span>`;
    }).join(" ");
    return `<tr>
      <td><strong>${nice[k]||k}</strong><br><span class="small">prev ${pct(first.prevalence)} · R² ${fmt(first.r2_cs_adj,3)} · k finale ${first.final_predictors_k??"—"}</span></td>
      <td>${hit?hit.N:"non raggiunto"}</td>
      <td>${fmt(last.mean_auc)}</td>
      <td>${pct(last.power)}</td>
      <td>${last.expected_events??"—"}/${last.expected_nonevents??"—"}</td>
      <td>${fmt(last.k_max_raw,2)} <span class="small">floor ${last.k_max_floor??"—"}</span></td>
      <td>${last.n_required_event_rate??"—"}</td>
      <td>${last.n_required_shrinkage??"—"}</td>
      <td><span class="small">${designText||"legacy results: rerun analysis to fill guardrails"}</span><br>${trend}</td>
    </tr>`
  }).join("");
  document.getElementById("sampleTable").innerHTML=`<thead><tr>
    <th>${thTip("Scenario","Tipo di simulazione. Le righe teoriche usano AUC target prefissate; le righe osservate usano l'AUC ottenuta da Nested CV o LOOCV.")}</th>
    <th>${thTip("N power≥80%","Primo N, tra quelli simulati, in cui almeno l'80% delle simulazioni supera la soglia AUC di successo. Se non raggiunto, nessun N testato arriva a potenza 80%.")}</th>
    <th>${thTip("AUC a N max","AUC media ottenuta al massimo N simulato, non necessariamente la migliore. Serve per vedere dove tende la simulazione.")}</th>
    <th>${thTip("Power a N max","Percentuale di simulazioni riuscite al massimo N simulato. È la vecchia lettura intuitiva basata su AUC.")}</th>
    <th>${thTip("Eventi/non-eventi a N max","Numero atteso di responder e non-responder al massimo N simulato, calcolato dalla prevalenza di disegno.")}</th>
    <th>${thTip("k max a N max","Numero massimo di predittori teoricamente sostenibile al massimo N simulato secondo shrinkage e R² ipotizzato. Il valore floor è arrotondato per difetto.")}</th>
    <th>${thTip("N event-rate","Campione minimo richiesto solo per stimare la prevalenza/event-rate con il margine scelto nella simulazione Monte Carlo, di default ±10%.")}</th>
    <th>${thTip("N shrinkage","Campione minimo richiesto per sostenere k predittori finali con shrinkage target, di default 0.90, e R² Cox-Snell adjusted scelto.")}</th>
    <th>${thTip("Andamento + guardrail","Ogni chip mostra N:power. Verde = tutti i guardrail ok; giallo = predittori ok ma event-rate incerto; rosso = solo esplorativo.")}</th>
  </tr></thead><tbody>${rows}</tbody>`;
}


function renderGuardrails(design){
  const box=document.getElementById("guardrailInterpretation");
  const table=document.getElementById("guardrailTable");
  if(!box||!table)return;
  if(!design || !Object.keys(design).length){
    box.innerHTML="Guardrail analitici non presenti nel JSON corrente. Rilancia <code>run_analysis.py</code> con la patch aggiornata.";
    table.innerHTML="";
    return;
  }
  box.innerHTML=`<strong>Prevalenza di disegno:</strong> ${pct(design.prevalence)} · <strong>N corrente:</strong> ${design.current_n} · <strong>eventi/non-eventi attesi:</strong> ${design.current_expected_events}/${design.current_expected_nonevents}<br>${design.interpretation||""}<br><span class="small">Questa sezione non è Monte Carlo: è il controllo analitico derivato da prevalenza, shrinkage e numero di predittori. Serve a capire se un'AUC promettente è anche metodologicamente difendibile.</span>`;
  const er=(design.event_rate_guardrails||[]).map(x=>`<tr><td>Event-rate precision ${helpTip("Campione minimo per stimare la prevalenza dell'evento con un margine assoluto scelto. Non dipende dal numero di feature.")}</td><td>margine ±${pct(x.risk_margin)}</td><td>${x.n_required}</td><td>—</td><td>—</td></tr>`).join("");
  const pr=(design.predictor_guardrails||[]).flatMap(block=>{
    const kmax=block.k_max_current_n||{};
    return (block.n_required_by_predictor_count||[]).map(x=>`<tr><td>Shrinkage / predictors ${helpTip("Campione minimo per limitare l'overfitting del modello, dato un numero di predittori finali k e una performance attesa R² Cox-Snell adjusted.")}</td><td>R² ${fmt(block.r2_cs_adj,3)} · k=${x.final_predictors_k}</td><td>${x.n_required_shrinkage}</td><td>${fmt(kmax.k_max_raw,2)}</td><td>${kmax.k_max_floor??"—"}</td></tr>`);
  }).join("");
  table.innerHTML=`<thead><tr>
    <th>${thTip("Criterio","Famiglia del controllo analitico: stima della prevalenza/event-rate oppure shrinkage/numero di predittori.")}</th>
    <th>${thTip("Scenario","Assunzione usata nel calcolo: margine di errore per event-rate oppure R² e numero di predittori per shrinkage.")}</th>
    <th>${thTip("N richiesto","Numerosità minima indipendente richiesta da quello specifico criterio.")}</th>
    <th>${thTip("k max a N corrente","Quanti predittori sarebbero sostenibili con il numero di pazienti attuali.")}</th>
    <th>${thTip("k max floor","k max arrotondato per difetto: è il numero intero realmente utilizzabile.")}</th>
  </tr></thead><tbody>${er}${pr}</tbody>`;
}


async function uploadExcel(){
  clearLog("Upload dataset...");
  setProgress(8);
  const file=document.getElementById("excelFile").files[0];
  if(!file){alert("Seleziona prima un file Excel.");return}
  const fd=new FormData();
  fd.append("file",file);
  const r=await fetch(API+"/api/upload-dataset",{method:"POST",body:fd});
  const j=await r.json();
  setProgress(100);
  log(JSON.stringify(j,null,2));
}

let livePollTimer = null;
let logSince = 0;

function appendLogEntry(entry){
  const el=document.getElementById("console");
  const prefix = entry.stream === "stderr"
    ? "[ERR] "
    : entry.stream === "system"
      ? "[SYS] "
      : "";
  el.textContent += "\n" + prefix + entry.line;
  el.scrollTop = el.scrollHeight;
}

async function pollRunStatus(){
  const r = await fetch(API + "/api/run-status?since=" + logSince, {cache:"no-store"});
  const j = await r.json();

  if (Array.isArray(j.logs)) {
    j.logs.forEach(appendLogEntry);
  }

  logSince = j.next_since ?? logSince;

  if (j.running) {
    // progresso euristico: non conosciamo la durata totale
    const current = parseFloat(document.getElementById("progressBar").style.width) || 15;
    setProgress(Math.min(92, current + 1.5));
    return;
  }

  if (livePollTimer) {
    clearInterval(livePollTimer);
    livePollTimer = null;
  }

  setProgress(j.returncode === 0 ? 100 : 100);
  log("Run terminato con return code: " + j.returncode);

  await refreshResults();
}


function getRunOptionsFromForm(){
  const singleRaw=document.getElementById("pruningThresholdInput")?.value||"";
  const sweepRaw=document.getElementById("pruningThresholdsInput")?.value||"";
  const topNRaw=document.getElementById("topNFinalInput")?.value||"1";
  const nSimRaw=document.getElementById("nSimInput")?.value||"100";
  const payload={
    pruning_threshold: singleRaw,
    pruning_thresholds: sweepRaw,
    top_n_final_model_features: Number(topNRaw),
    n_sample_size_simulations: Number(nSimRaw)
  };
  return payload;
}

async function runBackend(){
  const runOptions=getRunOptionsFromForm();
  const isSweep=String(runOptions.pruning_thresholds||"").trim().length>0;
  clearLog((isSweep?"Avvio sweep soglie di pruning":"Avvio run_analysis.py") + ". I log compariranno qui in tempo reale...");
  setProgress(10);
  logSince = 0;

  log("Run options: " + JSON.stringify(runOptions));
  const r = await fetch(API + "/api/run-analysis", {method:"POST", headers:{"content-type":"application/json"}, body:JSON.stringify(runOptions)});
  const j = await r.json();

  if (!j.ok) {
    log("ERRORE avvio backend: " + (j.error || JSON.stringify(j)));
    setProgress(100);
    return;
  }

  if (j.already_running) {
    log("Un run è già in corso. Mi aggancio ai log live...");
  } else {
    log("Run avviato. Script: " + (j.script || "run_analysis.py") + " · PID: " + (j.pid ?? "n/d"));
  }

  if (livePollTimer) clearInterval(livePollTimer);
  livePollTimer = setInterval(() => pollRunStatus().catch(e => log("ERRORE polling: " + e.message)), 1000);
  await pollRunStatus();
}

async function stopBackend(){
  log("Richiesta stop run...");
  const r = await fetch(API + "/api/stop-run", {method:"POST"});
  const j = await r.json();
  log(JSON.stringify(j, null, 2));
}

async function fetchJsonLogged(url, label){
  log("FETCH " + label + ": " + url);
  const r = await fetch(url, {cache:"no-store"});
  log(label + " HTTP status: " + r.status + " " + r.statusText);
  log(label + " content-type: " + (r.headers.get("content-type") || "n/d"));

  const text = await r.text();
  log(label + " response length: " + text.length + " chars");

  if(!r.ok){
    log(label + " response preview: " + text.slice(0, 500));
    throw new Error(label + " failed with HTTP " + r.status);
  }

  try{
    const json = JSON.parse(text);
    log(label + " JSON keys: " + Object.keys(json || {}).join(", "));
    return json;
  }catch(e){
    log(label + " JSON parse error: " + e.message);
    log(label + " response preview: " + text.slice(0, 500));
    throw e;
  }
}

async function refreshResults(){
  log("CLICK ricevuto su Aggiorna risultati");
  log("location.href: " + location.href);
  log("location.protocol: " + location.protocol);
  log("API base: " + API);

  if(location.protocol === "file:"){
    log("ATTENZIONE: stai aprendo da file://. Il fetch automatico può non funzionare. Apri http://127.0.0.1:8765");
  }

  let status = null;
  try{
    status = await fetchJsonLogged(API + "/api/debug-paths", "debug-paths");
    log("debug results_dir: " + (status.results_dir || "n/d"));
    log("debug models_dir: " + (status.models_dir || "n/d"));
  }catch(e){
    log("debug-paths non disponibile: " + e.message);
    try{
      status = await fetchJsonLogged(API + "/api/status", "status");
      log("status results_dir: " + (status.results_dir || "n/d"));
      log("status models_dir: " + (status.models_dir || "n/d"));
    }catch(e2){
      log("status non disponibile: " + e2.message);
    }
  }

  let j = null;
  try{
    j = await fetchJsonLogged(API + "/api/results", "api-results");
  }catch(e){
    log("ERRORE /api/results: " + e.message);
    j = {};
  }

  if(j && j.analysis){
    state.analysis = j.analysis;
    log("state.analysis assegnato da /api/results");
  }else{
    log("/api/results non contiene analysis. Provo fallback diretto /analysis_results.json");
    try{
      state.analysis = await fetchJsonLogged(API + "/analysis_results.json", "analysis-alias-root");
      log("state.analysis assegnato da /analysis_results.json");
    }catch(e1){
      log("fallback /analysis_results.json fallito: " + e1.message);
      try{
        state.analysis = await fetchJsonLogged(API + "/results/analysis_results.json", "analysis-alias-results");
        log("state.analysis assegnato da /results/analysis_results.json");
      }catch(e2){
        log("fallback /results/analysis_results.json fallito: " + e2.message);
      }
    }
  }

  if(j && j.candidate){
    state.candidate = j.candidate;
    log("state.candidate assegnato da /api/results");
  }else{
    log("/api/results non contiene candidate. Provo fallback candidate JSON");
    try{
      state.candidate = await fetchJsonLogged(API + "/candidate_model_comparison.json", "candidate-alias-root");
      log("state.candidate assegnato da /candidate_model_comparison.json");
    }catch(e1){
      log("fallback candidate root fallito: " + e1.message);
      try{
        state.candidate = await fetchJsonLogged(API + "/results/candidate_model_comparison.json", "candidate-alias-results");
        log("state.candidate assegnato da /results/candidate_model_comparison.json");
      }catch(e2){
        log("fallback candidate results fallito: " + e2.message);
      }
    }
  }

  if(j && j.nested){
    state.nested = j.nested;
    log("state.nested assegnato da /api/results");
  }else{
    log("/api/results non contiene nested. Provo fallback nested JSON");
    try{
      state.nested = await fetchJsonLogged(API + "/nested_cv_fold_details.json", "nested-alias-root");
      log("state.nested assegnato da /nested_cv_fold_details.json");
    }catch(e1){
      log("fallback nested root fallito: " + e1.message);
      try{
        state.nested = await fetchJsonLogged(API + "/results/nested_cv_fold_details.json", "nested-alias-results");
        log("state.nested assegnato da /results/nested_cv_fold_details.json");
      }catch(e2){
        log("fallback nested results fallito: " + e2.message);
      }
    }
  }

  if(j && j.sweep){
    state.sweep = j.sweep;
    log("state.sweep assegnato da /api/results");
  }

  if(j && j.summary){
    state.summary = j.summary;
    log("state.summary assegnato da /api/results");
  }

  log("STATE CHECK: analysis=" + !!state.analysis + ", candidate=" + !!state.candidate + ", nested=" + !!state.nested + ", sweep=" + !!state.sweep + ", summary=" + !!state.summary);
  if(state.analysis){
    log("analysis patients=" + state.analysis.patients + ", lesions=" + state.analysis.lesions + ", nested_auc=" + (state.analysis.cv && state.analysis.cv.mean_auc));
  }

  try{
    render();
    log("render() completato senza errori.");
  }catch(e){
    log("ERRORE dentro render(): " + e.message);
    if(e.stack) log(e.stack);
    throw e;
  }

  log("Risultati aggiornati.");
}

async function readJsonFile(input,key){
  const file=input.files[0]; if(!file)return;
  const text=await file.text();
  state[key]=JSON.parse(text);
  if(key==="nested" && state.analysis){
    state.analysis.cv=state.analysis.cv||{};
    if(state.nested.fold_details)state.analysis.cv.fold_details=state.nested.fold_details;
    if(state.nested.all_predictions)state.analysis.cv.all_predictions=state.nested.all_predictions;
  }
  render();
}

document.getElementById("uploadExcelBtn").onclick=()=>uploadExcel().catch(e=>log("ERRORE upload: "+e.message));
document.getElementById("runBtn").onclick=()=>runBackend().catch(e=>log("ERRORE run: "+e.message));
document.getElementById("stopBtn").onclick=()=>stopBackend().catch(e=>log("ERRORE stop: "+e.message));
const refreshButton = document.getElementById("refreshBtn");
refreshButton.onclick = async () => {
  log("Handler onclick refreshBtn partito.");
  try {
    await refreshResults();
  } catch(e) {
    log("ERRORE refresh handler: " + (e.stack || e.message || e));
  }
};
refreshButton.addEventListener("click", () => {
  log("Evento DOM click intercettato su refreshBtn.");
});
document.getElementById("jsonBtn").onclick=()=>document.getElementById("analysisJson").scrollIntoView({behavior:"smooth"});
document.getElementById("analysisJson").onchange=e=>readJsonFile(e.target,"analysis");
document.getElementById("candidateJson").onchange=e=>readJsonFile(e.target,"candidate");
document.getElementById("nestedJson").onchange=e=>readJsonFile(e.target,"nested");


function openGuide(){
  const guideUrl = location.protocol === "file:" ? "guide.html" : "/guide.html";
  window.open(guideUrl, "_blank");
}
const guideButton = document.getElementById("guideBtn");
if (guideButton) {
  guideButton.onclick = openGuide;
}

refreshResults().catch(()=>render());
