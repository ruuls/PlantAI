/** The detect-in-field tab: method, parameters, run control and live progress. */

import { api } from '../api.js';
import { state, setState } from '../state.js';
import { $, clear, el, fmt, fmt2, notifyError, slider, toast } from '../ui.js';

const METHODS = [
  ['csv', 'Upload CSV (precomputed detections)',
    'Load an existing prediction table from the sidebar instead of running the model.'],
  ['field', 'Run model on full field',
    'Tiles the whole orthomosaic, scores every tile, deduplicates overlap, then georeferences.'],
  ['quick', 'Quick count on selected AOI',
    'Scores only the latest drawn polygon — fast enough to work plot by plot.'],
];

// Full-field runs use heavy overlap so nothing is lost on a tile seam; an AOI
// is small enough that lighter overlap keeps it interactive.
const PRESETS = {
  field: { side: 0.7, front: 0.8 },
  quick: { side: 0.3, front: 0.3 },
};

let stream = null;
let onRunComplete = () => {};

/** Re-seed the parameters once server defaults and the method are both known. */
export function applyDefaults(config) {
  const preset = PRESETS[state.method] || PRESETS.field;
  setState({
    detect: {
      tile: config.tile_size,
      conf: config.conf,
      dedup: config.dedup_radius_m,
      ...preset,
    },
  });
  document.querySelectorAll('#tile-seg .seg-btn').forEach((b) => {
    b.classList.toggle('is-on', Number(b.dataset.tile) === state.detect.tile);
  });
  renderParams();
}

export function initDetectTab({ onComplete }) {
  onRunComplete = onComplete || (() => {});
  renderMethods();
  renderParams();
  renderSteps(null);

  $('#tile-seg').addEventListener('click', (e) => {
    const button = e.target.closest('.seg-btn');
    if (!button) return;
    setState({ detect: { ...state.detect, tile: Number(button.dataset.tile) } });
    $('#tile-seg').querySelectorAll('.seg-btn').forEach((b) => b.classList.toggle('is-on', b === button));
  });

  $('#run-detect').addEventListener('click', startRun);
  $('#cancel-detect').addEventListener('click', cancelRun);
  $('#download-run-csv').addEventListener('click', async () => {
    if (!state.jobDetectionId) return;
    try {
      await api.downloadDetections(state.jobDetectionId, 0);
    } catch (err) {
      notifyError(err, 'Download failed');
    }
  });
}

export function renderMethods({ applyPreset = false } = {}) {
  if (applyPreset) {
    const preset = PRESETS[state.method];
    if (preset) {
      setState({ detect: { ...state.detect, ...preset } });
      renderParams();
    }
  }
  const host = clear($('#method-list'));
  for (const [id, name, desc] of METHODS) {
    const on = state.method === id;
    host.append(el('label', {
      class: `method-row ${on ? 'is-on' : ''}`,
      onClick: () => {
        const preset = PRESETS[id];
        setState({
          method: id,
          detect: preset ? { ...state.detect, ...preset } : state.detect,
        });
        renderMethods();
        renderParams();
        syncRunButton();
      },
    },
      el('span', { class: 'radio' }),
      el('span', { class: 'method-copy' },
        el('span', { class: 'method-title', text: name }),
        el('span', { class: 'method-desc', text: desc }),
      ),
    ));
  }
  syncRunButton();
}

function renderParams() {
  const host = clear($('#detect-params'));
  const d = state.detect;
  const bind = (key) => (v) => {
    setState({ detect: { ...state.detect, [key]: v } });
    return fmt2(v);
  };

  host.append(
    slider({ label: 'Confidence', min: 0, max: 1, step: 0.05, value: d.conf, display: fmt2(d.conf), onInput: bind('conf'), width: '152px' }),
    slider({ label: 'Dedup radius (m)', min: 0, max: 0.5, step: 0.05, value: d.dedup, display: fmt2(d.dedup), onInput: bind('dedup'), width: '152px' }),
    slider({ label: 'Side overlap', min: 0, max: 0.9, step: 0.05, value: d.side, display: fmt2(d.side), onInput: bind('side'), width: '152px' }),
    slider({ label: 'Front overlap', min: 0, max: 0.9, step: 0.05, value: d.front, display: fmt2(d.front), onInput: bind('front'), width: '152px' }),
  );

  $('#detect-note').textContent = state.method === 'quick'
    ? 'Quick-count defaults use lower overlap because the AOI is small. Raise it if plants sit near tile edges.'
    : 'Defaults are recommended. Higher overlap catches more edge detections but lengthens the run.';
}

export function syncRunButton() {
  const button = $('#run-detect');
  const hint = $('#run-hint');
  const running = state.job && !['done', 'error', 'cancelled'].includes(state.job.state);

  if (state.method === 'csv') {
    button.textContent = 'Load a CSV from the sidebar';
    button.disabled = true;
    hint.textContent = 'This method does not run the model.';
    return;
  }

  button.textContent = running
    ? 'Running…'
    : state.method === 'quick' ? 'Run quick AI count on selected AOI' : 'Run detection on full field';

  const blockers = [];
  if (!state.dataset) blockers.push('load an orthomosaic');
  if (!state.model.loaded) blockers.push('load a model');
  if (state.method === 'quick' && !state.aoi) blockers.push('draw an AOI');

  button.disabled = running || blockers.length > 0;
  hint.textContent = running
    ? 'Streaming tiles to the model'
    : blockers.length ? `First ${blockers.join(', then ')}.`
      : state.job?.state === 'done' ? 'Complete — results loaded to the field map'
        : 'Uses the settings above';

  $('#cancel-detect').classList.toggle('is-hidden', !running);
}

async function startRun() {
  const d = state.detect;
  try {
    const job = await api.startDetection({
      dataset_id: state.dataset.id,
      mode: state.method === 'quick' ? 'aoi' : 'field',
      aoi: state.method === 'quick' ? state.aoi : null,
      tile_size: d.tile,
      conf: d.conf,
      side_overlap: d.side,
      front_overlap: d.front,
      dedup_radius_m: d.dedup,
    });
    setState({ job, jobDetectionId: null });
    renderSteps(job);
    syncRunButton();
    follow(job.id);
  } catch (err) {
    notifyError(err, 'Could not start detection');
  }
}

async function cancelRun() {
  if (!state.job) return;
  try {
    await api.cancelJob(state.job.id);
    toast('Cancelling after the current tile batch…', 'info');
  } catch (err) {
    notifyError(err, 'Could not cancel the run');
  }
}

/**
 * Follow a run over server-sent events, falling back to polling if the stream
 * cannot be opened (some corporate proxies buffer SSE into uselessness).
 */
function follow(jobId) {
  stream?.close();
  let sawEvent = false;

  stream = new EventSource(api.jobEventsUrl(jobId));
  stream.onmessage = (event) => {
    sawEvent = true;
    let job;
    try { job = JSON.parse(event.data); } catch { return; }
    setState({ job });
    renderSteps(job);
    syncRunButton();
    if (['done', 'error', 'cancelled'].includes(job.state)) finish(job);
  };
  stream.addEventListener('end', () => stream?.close());
  stream.onerror = () => {
    stream?.close();
    stream = null;
    if (!sawEvent) poll(jobId);
  };
}

async function poll(jobId) {
  try {
    const job = await api.job(jobId);
    setState({ job });
    renderSteps(job);
    syncRunButton();
    if (['done', 'error', 'cancelled'].includes(job.state)) { finish(job); return; }
  } catch (err) {
    notifyError(err, 'Lost track of the detection run');
    return;
  }
  setTimeout(() => poll(jobId), 1200);
}

function finish(job) {
  stream?.close();
  stream = null;
  syncRunButton();

  if (job.state === 'error') {
    notifyError(new Error(job.error), 'Detection failed');
    return;
  }
  if (job.state === 'cancelled') {
    toast('Detection cancelled.', 'warn');
    return;
  }
  setState({ jobDetectionId: job.detection_id });
  $('#download-run-csv').disabled = false;
  toast(
    `Detection complete — ${fmt(job.stats.plants)} plants, ${fmt(job.stats.weeds)} weeds `
    + `from ${fmt(job.stats.tiles)} tiles in ${job.elapsed.toFixed(0)}s.`,
    'ok', 'Run finished',
  );
  onRunComplete(job);
}

export function renderSteps(job) {
  const host = clear($('#progress-steps'));
  const steps = job?.steps || [
    { key: 'tile', label: 'Tiling raster window', progress: 0, note: 'queued' },
    { key: 'infer', label: 'Model inference', progress: 0, note: 'queued' },
    { key: 'dedupe', label: 'Removing duplicates', progress: 0, note: 'queued' },
    { key: 'project', label: 'Converting to map coordinates', progress: 0, note: 'queued' },
  ];

  steps.forEach((step, index) => {
    const done = step.progress >= 1;
    host.append(el('div', { class: 'step-row' },
      el('span', { class: `step-badge ${done ? 'is-done' : ''}`, text: done ? '✓' : String(index + 1) }),
      el('div', { class: 'step-body' },
        el('span', { text: step.label }),
        el('div', { class: 'progress' },
          el('div', { class: 'progress-fill', style: { width: `${Math.round(step.progress * 100)}%` } })),
      ),
      el('span', { class: 'num', text: step.note || 'queued' }),
    ));
  });
}
