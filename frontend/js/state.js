/**
 * One mutable store with a subscribe/notify loop.
 *
 * The console has four tabs reading the same detections, AOI and filters;
 * a single store is what keeps the KPI strip, the map and the analysis tab
 * from drifting apart.
 */

const listeners = new Set();

export const state = {
  // service
  online: false,
  config: null,
  model: { loaded: false, name: '', error: '' },

  // raster
  dataset: null,
  orthoMode: 'path',
  pendingOrtho: null,

  // detections
  detectionId: null,
  detectionSource: 'csv',
  summary: { total: 0, plants: 0, weeds: 0, weed_pressure_pct: 0 },
  points: { count: 0, lon: new Float64Array(0), lat: new Float64Array(0), conf: new Float32Array(0), cls: new Uint8Array(0) },
  pendingCsv: null,

  // map
  basemap: 'Orthomosaic',
  layers: { plants: true, weeds: true, heat: false, aoi: true },
  aoi: null,          // GeoJSON FeatureCollection of the latest shape
  aoiCounts: null,
  homeZoom: 24,

  // filters
  minConf: 0.25,
  standTarget: 150000,
  alertPct: 85,

  // analysis
  scope: 'aoi',
  basis: 'rows',
  plotId: 'plot_001',
  geometry: { rows: 4, row_length_ft: 15, row_spacing_in: 15 },
  analysis: null,

  // detection run
  method: 'field',
  detect: { tile: 1280, conf: 0.25, dedup: 0.1, side: 0.3, front: 0.3 },
  job: null,
  jobDetectionId: null,

  // images tab
  imgConf: 0.25,
  imageFiles: [],
  batch: [],

  tab: 'map',
};

export function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

/** Shallow-merge a patch and notify every listener with the changed keys. */
export function setState(patch) {
  Object.assign(state, patch);
  const keys = new Set(Object.keys(patch));
  for (const fn of listeners) fn(state, keys);
}

/** Notify without changing anything — for in-place mutations of nested objects. */
export function touch(...keys) {
  const changed = new Set(keys);
  for (const fn of listeners) fn(state, changed);
}

/** The confidence filter applied client-side, so the slider stays instant. */
export function visibleIndices() {
  const { points, minConf, layers } = state;
  const out = [];
  for (let i = 0; i < points.count; i += 1) {
    if (points.conf[i] < minConf) continue;
    const isWeed = points.cls[i] === 1;
    if (isWeed ? !layers.weeds : !layers.plants) continue;
    out.push(i);
  }
  return out;
}

/** Counts after the confidence filter, ignoring layer visibility. */
export function filteredSummary() {
  const { points, minConf } = state;
  let plants = 0;
  let weeds = 0;
  for (let i = 0; i < points.count; i += 1) {
    if (points.conf[i] < minConf) continue;
    if (points.cls[i] === 1) weeds += 1;
    else plants += 1;
  }
  const total = plants + weeds;
  return {
    total,
    plants,
    weeds,
    weed_pressure_pct: total ? (weeds / total) * 100 : 0,
  };
}

/** The body every analysis / export / report call sends. */
export function analysisBody() {
  return {
    scope: state.scope,
    aoi: state.scope === 'aoi' ? state.aoi : null,
    min_conf: state.minConf,
    basis: state.basis,
    geometry: state.geometry,
    target_plants_acre: state.standTarget,
    alert_pct: state.alertPct,
    plot_id: state.plotId,
  };
}
