/** The left dock: layers, basemap, sources and filters. */

import { api } from '../api.js';
import { state, setState, touch } from '../state.js';
import { $, clear, dropzone, el, fmt, notifyError, radioRow, slider, toast } from '../ui.js';
import { BASEMAPS } from '../map.js';

const LAYER_DEFS = [
  ['plants', 'Plants', 'var(--plant)'],
  ['weeds', 'Weeds', 'var(--weed)'],
  ['heat', 'Weed density heatmap', '#e8a33d'],
  ['aoi', 'AOI polygon', 'var(--aoi)'],
];

const SOURCE_DEFS = [
  ['csv', 'Upload plant/weed CSV'],
  ['field', 'Run model on full field'],
  ['quick', 'Quick count on selected AOI'],
];

let dockActions = null;

/**
 * Wire the dock's listeners once, then render.
 *
 * Kept separate from `renderSidebar` on purpose: the earlier version re-ran
 * the wiring on every state change, so each `addEventListener` stacked and one
 * click on "Load plant/weed locations" uploaded the CSV several times over.
 */
export function initSidebar(actions) {
  dockActions = actions;
  onSourceChanged = actions.onSourceChanged || (() => {});
  wireModel(actions);
  wireOrtho(actions);
  wireCsv(actions);
  renderSidebar();
}

/** Repaint the dock's stateful controls. Safe to call as often as you like. */
export function renderSidebar() {
  if (!dockActions) return;
  renderLayers(dockActions);
  renderBasemaps(dockActions);
  renderSources();
  renderFilters(dockActions);
}

// ────────────────────────────────────────────────── layers ──
function renderLayers(actions) {
  const host = clear($('#layer-list'));
  for (const [key, label, color] of LAYER_DEFS) {
    const on = state.layers[key];
    host.append(el('label', {
      class: `layer-row ${on ? 'is-on' : ''}`,
      onClick: () => {
        state.layers[key] = !state.layers[key];
        touch('layers');
        renderLayers(actions);
        actions.onLayersChanged();
      },
    },
      el('span', { class: 'checkbox', text: on ? '✓' : '' }),
      el('span', { class: 'swatch', style: { background: color } }),
      el('span', { class: 'layer-text', text: label }),
      el('span', { class: 'spacer' }),
      el('span', { class: 'num', id: `layer-count-${key}` }),
    ));
  }
  updateLayerCounts();
}

export function updateLayerCounts() {
  const counts = { plants: 0, weeds: 0 };
  const { points, minConf } = state;
  for (let i = 0; i < points.count; i += 1) {
    if (points.conf[i] < minConf) continue;
    if (points.cls[i] === 1) counts.weeds += 1;
    else counts.plants += 1;
  }
  const plantNode = $('#layer-count-plants');
  const weedNode = $('#layer-count-weeds');
  if (plantNode) plantNode.textContent = points.count ? fmt(counts.plants) : '';
  if (weedNode) weedNode.textContent = points.count ? fmt(counts.weeds) : '';
}

// ───────────────────────────────────────────────── basemap ──
function renderBasemaps(actions) {
  const host = clear($('#basemap-grid'));
  for (const name of BASEMAPS) {
    host.append(el('button', {
      class: `seg-btn ${state.basemap === name ? 'is-on' : ''}`,
      text: name,
      onClick: () => {
        if (name === 'Orthomosaic' && !state.dataset) {
          toast('Load an orthomosaic first, or stay on a web basemap.', 'warn');
          return;
        }
        setState({ basemap: name });
        renderBasemaps(actions);
        actions.onBasemapChanged(name);
      },
    }));
  }
}

// ───────────────────────────────────────────────── sources ──
let onSourceChanged = () => {};

function renderSources() {
  const host = clear($('#source-list'));
  for (const [id, label] of SOURCE_DEFS) {
    host.append(radioRow(label, state.detectionSource === id, () => {
      setState({ detectionSource: id, method: id === 'quick' ? 'quick' : id === 'field' ? 'field' : 'csv' });
      renderSources();
      $('#csv-source-body').classList.toggle('is-hidden', id !== 'csv');
      $('#run-source-body').classList.toggle('is-hidden', id === 'csv');
      onSourceChanged();
    }));
  }
}

// ───────────────────────────────────────────────── filters ──
function renderFilters(actions) {
  const host = clear($('#filter-sliders'));
  host.append(
    slider({
      label: 'Min confidence', min: 0, max: 1, step: 0.05, value: state.minConf,
      display: `${Math.round(state.minConf * 100)}%`,
      onInput: (v) => { setState({ minConf: v }); actions.onFilterChanged(); return `${Math.round(v * 100)}%`; },
    }),
    slider({
      label: 'Target plants / acre', min: 20000, max: 400000, step: 10000, value: state.standTarget,
      display: fmt(state.standTarget),
      onInput: (v) => { setState({ standTarget: v }); actions.onTargetChanged(); return fmt(v); },
    }),
    slider({
      label: 'Alert below (% of target)', min: 50, max: 100, step: 5, value: state.alertPct,
      display: `${state.alertPct}%`,
      onInput: (v) => { setState({ alertPct: v }); actions.onTargetChanged(); return `${v}%`; },
    }),
    slider({
      label: 'Max home zoom', min: 16, max: 24, step: 1, value: state.homeZoom,
      display: String(state.homeZoom),
      onInput: (v) => { setState({ homeZoom: v }); return String(v); },
    }),
  );
}

// ─────────────────────────────────────────────────── model ──
function setModelStatus(text, kind) {
  const node = $('#model-status');
  node.className = `status-line ${kind === 'ok' ? 'is-ok' : kind === 'err' ? 'is-err' : ''}`;
  clear(node).append(
    el('span', { class: `dot dot-${kind === 'ok' ? 'ok' : kind === 'err' ? 'err' : 'idle'}` }),
    el('span', { text }),
  );
}

export function reflectModel(status) {
  setState({ model: status });
  if (status.loaded) {
    setModelStatus(`Model loaded · ${(status.class_names || []).join(', ') || 'no labels'}`, 'ok');
    if (status.path) $('#model-path').value = status.path;
  } else if (status.error) {
    setModelStatus(status.error, 'err');
  } else {
    setModelStatus('No model loaded', 'idle');
  }
}

function wireModel() {
  $('#model-load').addEventListener('click', async () => {
    const path = $('#model-path').value.trim();
    if (!path) { toast('Enter the path to a .pt checkpoint.', 'warn'); return; }
    setModelStatus('Loading…', 'idle');
    try {
      reflectModel(await api.loadModel(path));
      toast('Model loaded.', 'ok');
    } catch (err) {
      reflectModel({ loaded: false, error: err.message });
      notifyError(err, 'Could not load the model');
    }
  });

  $('#model-browse').addEventListener('click', () => pickFromLibrary('model', (file) => {
    $('#model-path').value = file.path;
    $('#model-load').click();
  }));
}

// ───────────────────────────────────────────── orthomosaic ──
function wireOrtho(actions) {
  const modeHost = $('#ortho-mode');
  modeHost.addEventListener('click', (e) => {
    const button = e.target.closest('.seg-btn');
    if (!button) return;
    const mode = button.dataset.mode;
    setState({ orthoMode: mode });
    modeHost.querySelectorAll('.seg-btn').forEach((b) => b.classList.toggle('is-on', b === button));
    $('#ortho-path-body').classList.toggle('is-hidden', mode !== 'path');
    $('#ortho-upload-body').classList.toggle('is-hidden', mode !== 'upload');
  });

  $('#ortho-load').addEventListener('click', async () => {
    const path = $('#ortho-path').value.trim();
    if (!path) { toast('Enter the path to a GeoTIFF on the server.', 'warn'); return; }
    try {
      const dataset = await api.datasetByPath(path);
      await actions.onDatasetLoaded(dataset);
    } catch (err) {
      notifyError(err, 'Could not load the orthomosaic');
    }
  });

  $('#ortho-browse').addEventListener('click', () => pickFromLibrary('raster', (file) => {
    $('#ortho-path').value = file.path;
    $('#ortho-load').click();
  }));

  const zone = $('#ortho-drop');
  const input = $('#ortho-file');
  const progressLine = $('#ortho-upload-progress');
  const fill = progressLine.querySelector('.progress-fill');
  const label = progressLine.querySelector('.num');

  dropzone(zone, input, async ([file]) => {
    progressLine.classList.remove('is-hidden');
    fill.style.width = '0%';
    label.textContent = '0%';
    try {
      const dataset = await api.uploadDataset(file, (ratio) => {
        const percent = Math.round(ratio * 100);
        fill.style.width = `${percent}%`;
        label.textContent = `${percent}%`;
      });
      await actions.onDatasetLoaded(dataset);
    } catch (err) {
      notifyError(err, 'Upload failed');
    } finally {
      setTimeout(() => progressLine.classList.add('is-hidden'), 900);
    }
  });

  $('#build-overviews').addEventListener('click', async (e) => {
    const button = e.currentTarget;
    button.disabled = true;
    button.textContent = 'Building overviews…';
    try {
      const result = await api.buildOverviews(state.dataset.id);
      setState({ dataset: { ...state.dataset, ...result } });
      reflectOverviews(state.dataset);
      actions.onOverviewsBuilt();
      toast(result.message || 'Overviews built.', 'ok');
    } catch (err) {
      notifyError(err, 'Could not build overviews');
    } finally {
      button.disabled = false;
      button.textContent = 'Build overviews now';
    }
  });
}

export function reflectOverviews(dataset) {
  const notice = $('#overview-notice');
  const needed = Boolean(dataset) && !dataset.has_overviews && dataset.size_bytes > 200e6;
  notice.classList.toggle('is-hidden', !needed);
}

// ───────────────────────────────────────────────────── CSV ──
function wireCsv(actions) {
  const zone = $('#csv-drop');
  const input = $('#csv-file');
  const button = $('#csv-load');

  const setZone = dropzone(zone, input, ([file]) => {
    setState({ pendingCsv: file });
    setZone(`<b>${file.name}</b><br><span class="faint">${(file.size / 1024).toFixed(0)} KB · ready to load</span>`, true);
    button.disabled = false;
  });

  button.addEventListener('click', async () => {
    if (!state.pendingCsv) return;
    button.disabled = true;
    button.textContent = 'Loading…';
    try {
      const result = await api.uploadDetectionCsv(
        state.pendingCsv, $('#csv-crs').value.trim(), state.dataset?.id,
      );
      await actions.onDetectionsLoaded(result.id, result.summary);
      const notes = [result.import?.class_note, result.import?.confidence_note].filter(Boolean);
      toast(
        `${fmt(result.count)} detections loaded${notes.length ? ` — ${notes.join(' ')}` : ''}`,
        notes.length ? 'warn' : 'ok',
      );
    } catch (err) {
      notifyError(err, 'Could not load the CSV');
    } finally {
      button.disabled = false;
      button.textContent = 'Load plant/weed locations';
    }
  });
}

// ───────────────────────────────────────────── file library ──
async function pickFromLibrary(kind, onPick) {
  let listing;
  try {
    listing = await api.library(kind);
  } catch (err) {
    notifyError(err, 'Could not read the library');
    return;
  }
  if (!listing.files.length) {
    toast(
      `No ${kind === 'model' ? '.pt files' : 'GeoTIFFs'} found under ${listing.roots.join(', ') || 'the configured library'}. `
      + 'Type a path instead, or set LIBRARY_DIRS on the server.',
      'warn',
    );
    return;
  }
  showPicker(kind, listing.files, onPick);
}

function showPicker(kind, files, onPick) {
  const backdrop = el('div', {
    style: {
      position: 'fixed', inset: '0', background: 'rgba(6,8,11,0.72)', zIndex: '3000',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
    },
    onClick: (e) => { if (e.target === backdrop) backdrop.remove(); },
  });

  const list = el('div', { style: { maxHeight: '52vh', overflowY: 'auto' } });
  for (const file of files) {
    list.append(el('button', {
      class: 'wide-btn',
      style: { marginBottom: '4px', display: 'block' },
      onClick: () => { backdrop.remove(); onPick(file); },
    },
      el('div', { text: file.name, style: { color: 'var(--ink)' } }),
      el('div', { class: 'num', text: `${file.dir} · ${file.size_label}` }),
    ));
  }

  backdrop.append(el('div', {
    class: 'card',
    style: { width: 'min(560px, 90vw)' },
  },
    el('div', { class: 'card-head', text: kind === 'model' ? 'Model library' : 'Raster library' }),
    el('div', { class: 'card-body' }, list),
    el('div', { class: 'card-foot' },
      el('button', { class: 'btn', text: 'Cancel', onClick: () => backdrop.remove() })),
  ));

  document.body.append(backdrop);
}
