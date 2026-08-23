/**
 * Console bootstrap: builds the tab bar, wires the panels together and keeps
 * the shared state, the map and the status bar in step.
 */

import { api, decodePoints } from './api.js';
import { analysisBody, state, setState } from './state.js';
import {
  $, bytes, clear, el, fmt, notifyError, toast,
} from './ui.js';
import {
  TILE_NOTES, clearAoi, fitPoints, getMap, goHome, initMap, invalidate,
  redraw, reloadOrtho, setAoiVisible, setBasemap, setOrthomosaic,
} from './map.js';
import { initSidebar, reflectModel, reflectOverviews, renderSidebar, updateLayerCounts } from './panels/sidebar.js';
import {
  refreshAoiCounts, renderAoiPanel, renderHeadline, renderStandStatus, updateMapEmptyState,
} from './panels/mapTab.js';
import { initAnalysisTab, refreshAnalysis, renderAnalysis, syncScopeControls } from './panels/analysisTab.js';
import { applyDefaults, initDetectTab, renderMethods, syncRunButton } from './panels/detectTab.js';
import { initImagesTab } from './panels/imagesTab.js';

const TABS = [
  ['map', 'Field map'],
  ['analysis', 'Field analysis'],
  ['detect', 'Detect in field'],
  ['images', 'Detect on images'],
];

// ─────────────────────────────────────────────────── tabs ──
function renderTabs() {
  const host = clear($('#tabs'));
  for (const [id, label] of TABS) {
    host.append(el('button', {
      class: `tab ${state.tab === id ? 'is-on' : ''}`,
      text: label,
      onClick: () => selectTab(id),
    }));
  }
}

function selectTab(id) {
  setState({ tab: id });
  renderTabs();
  document.querySelectorAll('[data-pane]').forEach((pane) => {
    pane.classList.toggle('is-hidden', pane.dataset.pane !== id);
  });
  // Leaflet measures its container on add; a hidden pane has no size.
  if (id === 'map') requestAnimationFrame(invalidate);
  if (id === 'analysis') refreshAnalysis();
}

// ───────────────────────────────────────────── status bar ──
function updateStatusBar({ center, zoom, scale }) {
  if (center) {
    const ns = center.lat >= 0 ? 'N' : 'S';
    const ew = center.lng >= 0 ? 'E' : 'W';
    $('#status-coords').textContent =
      `${Math.abs(center.lat).toFixed(5)} ${ns}   ${Math.abs(center.lng).toFixed(5)} ${ew}`;
  }
  if (Number.isFinite(zoom)) $('#status-zoom').textContent = `zoom ${zoom.toFixed(0)}`;
  if (Number.isFinite(scale) && scale > 0) {
    // Screen scale denominator at a nominal 96 dpi.
    $('#status-scale').textContent = `1:${fmt(scale * 96 / 0.0254)}`;
  }
  $('#status-tilenote').textContent = state.dataset || state.basemap !== 'Orthomosaic'
    ? TILE_NOTES[state.basemap]
    : 'No raster layer';
  $('#status-crs').textContent = state.dataset
    ? `${state.dataset.crs} → EPSG:4326`
    : 'EPSG:4326';
}

function updateTopbar() {
  const name = $('#topbar-file');
  const sub = $('#topbar-file-sub');
  if (!state.dataset) {
    name.textContent = 'No orthomosaic loaded';
    name.title = 'No orthomosaic loaded';
    sub.textContent = 'Load a GeoTIFF to begin';
    return;
  }
  const d = state.dataset;
  name.textContent = d.name;
  name.title = d.path || d.name;
  sub.textContent = `${bytes(d.size_bytes)} · ${d.crs} · ${d.gsd_cm.toFixed(2)} cm/px`
    + `${d.has_overviews ? '' : ' · no overviews'}`;
}

// ───────────────────────────────────── shared refreshers ──

/** Fetch the point cloud for the active detection set and repaint everything. */
async function loadPoints(detectionId) {
  const buffer = await api.points(detectionId, 0);
  setState({ points: decodePoints(buffer) });
  redraw();
  renderHeadline();
  updateLayerCounts();
  updateMapEmptyState();
}

async function onDetectionsLoaded(detectionId, summary) {
  setState({ detectionId, summary: summary || state.summary, aoiCounts: null });
  await loadPoints(detectionId);
  if (!state.dataset) fitPoints();
  await refreshAoiCounts();
  await refreshAnalysis();
  renderStandStatus();
  syncRunButton();
}

async function onDatasetLoaded(dataset) {
  setState({ dataset, basemap: 'Orthomosaic' });
  setOrthomosaic(dataset);
  setBasemap('Orthomosaic');
  updateTopbar();
  reflectOverviews(dataset);
  updateMapEmptyState();
  updateStatusBar({ zoom: getMap()?.getZoom() });
  syncRunButton();
  renderSidebar();
  toast(
    `${dataset.name} loaded — ${fmt(dataset.width)}×${fmt(dataset.height)} px at ${dataset.gsd_cm.toFixed(2)} cm/px.`,
    'ok',
  );
}

const sidebarActions = {
  onLayersChanged() {
    setAoiVisible(state.layers.aoi);
    redraw();
  },
  onBasemapChanged(name) {
    setBasemap(name);
    updateMapEmptyState();
    updateStatusBar({ zoom: getMap()?.getZoom() });
  },
  onFilterChanged() {
    redraw();
    renderHeadline();
    updateLayerCounts();
    scheduleFilterRefresh();
  },
  onTargetChanged() {
    scheduleFilterRefresh();
  },
  onDatasetLoaded,
  onDetectionsLoaded,
  onSourceChanged() {
    renderMethods({ applyPreset: true });
  },
  onOverviewsBuilt() {
    reflectOverviews(state.dataset);
    reloadOrtho();
    updateTopbar();
  },
};

// Sliders fire continuously; recomputing metrics per pixel of travel would
// swamp the backend, so trailing-edge debounce the server round trips.
let filterTimer = null;
function scheduleFilterRefresh() {
  clearTimeout(filterTimer);
  filterTimer = setTimeout(async () => {
    await refreshAoiCounts();
    await refreshAnalysis();
    renderStandStatus();
  }, 260);
}

// ──────────────────────────────────────────────── toolbar ──
function wireToolbar() {
  document.body.addEventListener('click', async (event) => {
    const button = event.target.closest('[data-action]');
    if (!button) return;
    const action = button.dataset.action;

    if (action === 'home') { goHome(); return; }

    if (action === 'restart-tiles') {
      if (!state.dataset) { toast('No orthomosaic is loaded.', 'warn'); return; }
      reloadOrtho();
      toast('Raster tiles reloaded.', 'info');
      return;
    }

    if (action === 'heatmap') {
      state.layers.heat = !state.layers.heat;
      renderSidebar();
      redraw();
      return;
    }

    if (action === 'pdf') {
      if (!state.detectionId) { toast('Load detections before exporting a report.', 'warn'); return; }
      try {
        await api.downloadReport(state.detectionId, {
          ...analysisBody(),
          dataset_name: state.dataset?.name || '',
        });
      } catch (err) {
        notifyError(err, 'Could not generate the report');
      }
      return;
    }

    if (action === 'export-csv') {
      if (!state.detectionId) { toast('Load detections first.', 'warn'); return; }
      try {
        await api.downloadDetections(state.detectionId, state.minConf);
      } catch (err) {
        notifyError(err, 'Export failed');
      }
      return;
    }

    if (action === 'export-aoi') {
      if (!state.detectionId || !state.aoi) { toast('Draw an AOI on the map first.', 'warn'); return; }
      try {
        await api.downloadAoi(state.detectionId, state.aoi, state.minConf);
      } catch (err) {
        notifyError(err, 'Export failed');
      }
      return;
    }

    if (action === 'clear-aoi') {
      clearAoi();
    }
  });

  // Reflect the heatmap toggle in the toolbar button's own state.
  const heatButton = document.querySelector('[data-action="heatmap"]');
  const sync = () => heatButton.classList.toggle('is-on', state.layers.heat);
  heatButton.addEventListener('click', () => setTimeout(sync, 0));
  sync();
}

// ────────────────────────────────────────────── lifecycle ──
async function boot() {
  renderTabs();
  selectTab('map');

  initMap($('#map'), {
    onAoi: async (aoi) => {
      setState({ aoi, aoiCounts: null });
      renderAoiPanel();
      syncRunButton();
      await refreshAoiCounts();
      if (state.scope === 'aoi') await refreshAnalysis();
      renderStandStatus();
    },
    onView: updateStatusBar,
  });

  initSidebar(sidebarActions);
  initAnalysisTab({
    onUpdated() {
      renderStandStatus();
      syncScopeControls();
    },
  });
  initDetectTab({
    async onComplete(job) {
      if (!job.detection_id) return;
      setState({ detectionSource: job.mode === 'aoi' ? 'quick' : 'field' });
      await onDetectionsLoaded(job.detection_id, {
        total: job.stats.kept, plants: job.stats.plants, weeds: job.stats.weeds,
      });
      selectTab('map');
    },
  });
  initImagesTab();
  wireToolbar();

  renderHeadline();
  renderAoiPanel();
  renderStandStatus();
  renderAnalysis();
  updateTopbar();
  updateMapEmptyState();
  updateStatusBar({ zoom: getMap()?.getZoom() });

  try {
    const [config, health] = await Promise.all([api.config(), api.health()]);
    setState({
      online: true,
      config,
      minConf: config.conf,
      standTarget: config.stand_target,
      alertPct: config.stand_alert_pct,
    });
    applyDefaults(config);
    $('#ortho-upload-hint').textContent = `up to ${bytes(config.max_upload_mb * 1e6)}`;
    renderSidebar();
    reflectModel(health.model);
    renderMethods();
  } catch (err) {
    toast(
      'The detection service is not responding. Start the API and reload this page.',
      'error', 'Service offline',
    );
    notifyError(err, 'Startup check failed');
  }

  syncRunButton();
}

document.addEventListener('DOMContentLoaded', boot);
