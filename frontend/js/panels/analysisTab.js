/** The field-analysis tab: scope, plot geometry, stand results and exports. */

import { api } from '../api.js';
import { analysisBody, state, setState } from '../state.js';
import {
  $, acres, clear, el, fillTable, fmt, fmt2, notifyError, pct, radioRow, setBanner, statCell, toast,
} from '../ui.js';

let analysisRequest = 0;
let onAnalysisUpdated = () => {};

const SCOPES = [
  ['aoi', 'Drawn AOI from map'],
  ['all', 'All loaded detections'],
];

export function initAnalysisTab({ onUpdated }) {
  onAnalysisUpdated = onUpdated || (() => {});
  renderScopes();
  renderGeoFields();

  $('#basis-seg').addEventListener('click', (e) => {
    const button = e.target.closest('.seg-btn');
    if (!button) return;
    setState({ basis: button.dataset.basis });
    $('#basis-seg').querySelectorAll('.seg-btn').forEach((b) => b.classList.toggle('is-on', b === button));
    renderGeoFields();
    refreshAnalysis();
  });

  $('#plot-id').addEventListener('change', (e) => setState({ plotId: e.target.value.trim() || 'plot_001' }));

  document.querySelectorAll('[data-export]').forEach((button) => {
    button.addEventListener('click', async () => {
      if (!state.detectionId) { toast('Load detections first.', 'warn'); return; }
      try {
        await api.downloadAnalysis(state.detectionId, button.dataset.export, analysisBody());
      } catch (err) {
        notifyError(err, 'Export failed');
      }
    });
  });
}

function renderScopes() {
  const host = clear($('#scope-choices'));
  for (const [id, label] of SCOPES) {
    host.append(radioRow(label, state.scope === id, () => {
      setState({ scope: id });
      renderScopes();
      refreshAnalysis();
    }));
  }
  $('#scope-hint').textContent = state.scope === 'aoi'
    ? 'Counting inside the latest drawn polygon only.'
    : 'Counting every detection in the loaded set.';
}

function renderGeoFields() {
  const host = clear($('#geo-fields'));
  if (state.basis !== 'rows') {
    $('#geometry-note').textContent = 'Density comes straight from the mapped area of the selected scope.';
    return;
  }
  $('#geometry-note').textContent =
    'Plot area = rows × row length × row spacing. This is how research plots are scored; '
    + 'it also makes plants per foot of row meaningful.';

  const fields = [
    ['rows', 'Rows counted', 1, 1],
    ['row_length_ft', 'Row length (ft)', 0.5, 0.1],
    ['row_spacing_in', 'Row spacing (in)', 0.5, 1],
  ];
  for (const [key, label, step, min] of fields) {
    host.append(el('div', { class: 'field-col' },
      el('span', { class: 'field-label', text: label }),
      el('input', {
        class: 'num-input', type: 'number', step, min, value: state.geometry[key],
        onChange: (e) => {
          const value = Math.max(min, Number(e.target.value) || min);
          e.target.value = value;
          setState({ geometry: { ...state.geometry, [key]: value } });
          refreshAnalysis();
        },
      }),
    ));
  }
}

export function syncScopeControls() {
  renderScopes();
}

/** Ask the server for the metrics behind the current scope and basis. */
export async function refreshAnalysis() {
  if (!state.detectionId) {
    setState({ analysis: null });
    renderAnalysis();
    onAnalysisUpdated();
    return;
  }
  if (state.scope === 'aoi' && !state.aoi) {
    setState({ analysis: null });
    renderAnalysis('Draw an area of interest on the map, or switch the scope to all detections.');
    onAnalysisUpdated();
    return;
  }

  const stamp = ++analysisRequest;
  try {
    const result = await api.analysis(state.detectionId, analysisBody());
    if (stamp !== analysisRequest) return;
    setState({ analysis: result });
    renderAnalysis();
    onAnalysisUpdated();
  } catch (err) {
    if (stamp !== analysisRequest) return;
    setState({ analysis: null });
    renderAnalysis(err.message);
    onAnalysisUpdated();
  }
}

export function renderAnalysis(message) {
  const banner = $('#analysis-banner');
  const result = state.analysis;

  if (!result) {
    setBanner(banner, 'amber', 'No stand estimate',
      message || 'Load detections to run the stand-count analysis.');
    clear($('#stand-stats')).append(
      ...['Plants', 'Weeds', 'Total detections', 'Weed pressure', 'Area (acres)',
        'Plants / acre', 'Weeds / acre', 'Plants / ft', '% of target', 'Total row length']
        .map((label) => statCell(label, '—')),
    );
    fillTable($('#rows-table'), ['Row', 'Length ft', 'Plants', 'Weeds', 'Plants/ft'], [],
      'Switch the basis to plot row geometry to see per-row values.');
    fillTable($('#class-table'), ['Class', 'Count', 'Avg conf', 'Min conf'], [], 'No detections loaded.');
    renderConfidenceChart([]);
    return;
  }

  const m = result.metrics;
  setBanner(banner, m.tone, m.status, result.status_detail);
  if (result.warning) {
    banner.append(el('div', {
      class: 'banner-detail',
      style: { marginTop: '6px', opacity: '1', color: 'var(--tone-amber-ink)' },
      text: result.warning,
    }));
  }

  clear($('#stand-stats')).append(
    statCell('Plants', fmt(m.plants)),
    statCell('Weeds', fmt(m.weeds)),
    statCell('Total detections', fmt(m.total)),
    statCell('Weed pressure', pct(m.weed_pressure_pct)),
    statCell('Area (acres)', acres(m.area_acres)),
    statCell('Plants / acre', fmt(m.plants_per_acre)),
    statCell('Weeds / acre', fmt(m.weeds_per_acre)),
    statCell('Plants / ft', m.plants_per_ft ? fmt2(m.plants_per_ft) : '—'),
    statCell('% of target', pct(m.pct_of_target)),
    statCell('Total row length', m.total_row_length_ft ? `${fmt(m.total_row_length_ft)} ft` : '—'),
  );

  fillTable($('#rows-table'),
    ['Row', 'Length ft', 'Plants', 'Weeds', 'Plants/ft'],
    result.rows.map((row) => [
      row.row_index, row.row_length_ft.toFixed(1), fmt(row.plants), fmt(row.weeds), fmt2(row.plants_per_ft),
    ]),
    'Switch the basis to plot row geometry to see per-row values.');

  fillTable($('#class-table'),
    ['Class', 'Count', 'Avg conf', 'Min conf'],
    result.class_confidence.map((entry) => [
      el('span', {},
        el('span', {
          class: 'cell-dot',
          style: { background: entry.class === 'Plant' ? 'var(--plant)' : 'var(--weed)' },
        }),
        entry.class),
      fmt(entry.count),
      entry.avg_conf === null ? '—' : `${entry.avg_conf.toFixed(1)}%`,
      entry.min_conf === null ? '—' : `${entry.min_conf.toFixed(1)}%`,
    ]),
    'No detections in this scope.');

  renderConfidenceChart(result.confidence_bands);
}

function renderConfidenceChart(bands) {
  const host = clear($('#conf-chart'));
  const values = bands.length ? bands : [{ label: '—', count: 0 }];
  const peak = Math.max(1, ...values.map((b) => b.count));

  values.forEach((band, index) => {
    const height = Math.max(2, (band.count / peak) * 110);
    host.append(el('div', { class: 'bar-col' },
      el('span', { class: 'num', text: fmt(band.count) }),
      el('div', { class: `bar ${index >= 4 ? 'is-high' : ''}`, style: { height: `${height}px` } }),
      el('span', { class: 'bar-label', text: band.label }),
    ));
  });
}
