/** The field-map tab: KPI strip, AOI panel and stand status. */

import { api } from '../api.js';
import { state, setState, filteredSummary } from '../state.js';
import { $, acres, clear, el, fmt, headlineCell, notifyError, pct, setBanner, statCell } from '../ui.js';

let aoiRequest = 0;

export function renderHeadline() {
  const s = filteredSummary();
  clear($('#headline')).append(
    headlineCell('Plants', fmt(s.plants)),
    headlineCell('Weeds', fmt(s.weeds)),
    headlineCell('Total', fmt(s.total)),
    headlineCell('Weed pressure', pct(s.weed_pressure_pct)),
  );
  $('#legend-plants').textContent = fmt(s.plants);
  $('#legend-weeds').textContent = fmt(s.weeds);
  $('#status-total').textContent = `${fmt(s.total)} detections`;
}

export function renderAoiPanel() {
  const counts = state.aoiCounts;
  const host = clear($('#aoi-stats'));

  if (!counts) {
    $('#aoi-sub').textContent = state.aoi
      ? 'Counting…'
      : 'Draw a polygon to scope counts';
    host.append(
      statCell('Plants', '—'), statCell('Weeds', '—'),
      statCell('Total', '—'), statCell('Weed pressure', '—'),
      statCell('Plants / acre', '—'), statCell('Weeds / acre', '—'),
    );
    return;
  }

  $('#aoi-sub').textContent = `Latest polygon only · ${acres(counts.area_acres)} acres`;
  host.append(
    statCell('Plants', fmt(counts.plants)),
    statCell('Weeds', fmt(counts.weeds)),
    statCell('Total', fmt(counts.total)),
    statCell('Weed pressure', pct(counts.weed_pressure_pct)),
    statCell('Plants / acre', fmt(counts.plants_per_acre)),
    statCell('Weeds / acre', fmt(counts.weeds_per_acre)),
  );
}

export function renderStandStatus() {
  const metrics = state.analysis?.metrics;
  const banner = $('#map-banner');
  const bar = $('#map-stand-bar');
  $('#map-target-label').textContent = `target ${fmt(state.standTarget)}/ac`;

  if (!metrics) {
    setBanner(banner, 'amber', 'No stand estimate',
      state.points.count
        ? 'Draw an area of interest, or switch the analysis scope to all detections.'
        : 'Load detections and draw an area to assess the stand.');
    bar.style.width = '0%';
    return;
  }

  setBanner(banner, metrics.tone, metrics.status, state.analysis.status_detail);
  bar.style.width = `${Math.max(0, Math.min(100, metrics.pct_of_target))}%`;
  bar.style.background = `var(--tone-${metrics.tone}-line)`;
}

/**
 * Refresh the AOI counts.
 *
 * Requests are stamped so a slow reply for an old polygon can never overwrite
 * the numbers for the shape the user is looking at now.
 */
export async function refreshAoiCounts() {
  if (!state.detectionId || !state.aoi) {
    setState({ aoiCounts: null });
    renderAoiPanel();
    return;
  }
  const stamp = ++aoiRequest;
  renderAoiPanel();
  try {
    const counts = await api.aoiCounts(state.detectionId, state.aoi, state.minConf);
    if (stamp !== aoiRequest) return;
    setState({ aoiCounts: counts });
    renderAoiPanel();
  } catch (err) {
    if (stamp !== aoiRequest) return;
    setState({ aoiCounts: null });
    renderAoiPanel();
    notifyError(err, 'Could not count inside the AOI');
  }
}

export function updateMapEmptyState() {
  const empty = $('#map-empty');
  const hide = Boolean(state.dataset) || state.points.count > 0 || state.basemap !== 'Orthomosaic';
  empty.classList.toggle('is-hidden', hide);
}
