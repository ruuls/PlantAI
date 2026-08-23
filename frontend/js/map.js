/**
 * The field map: basemaps, the orthomosaic tile layer, AOI drawing, and a
 * canvas layer that renders the detections.
 *
 * Detections are drawn onto one canvas rather than added as Leaflet markers.
 * A field run routinely returns several hundred thousand points; that many DOM
 * nodes locks the browser up, while a viewport-culled canvas pass stays
 * interactive.
 */

import { state, setState, touch, visibleIndices } from './state.js';
import { api } from './api.js';

const PLANT = '#45c46a';
const WEED = '#ef5350';
const AOI_COLOR = '#a78bfa';
const AOI_STYLE = { color: AOI_COLOR, weight: 2, fillOpacity: 0.1, fillColor: AOI_COLOR };
const MARKER_RADIUS = 3.2;

export const BASEMAPS = ['Orthomosaic', 'Satellite', 'Terrain', 'Street'];

export const TILE_NOTES = {
  Orthomosaic: 'rio-tiler · bands 1,2,3 · 2–98% stretch',
  Satellite: 'Esri World Imagery',
  Terrain: 'Esri World Topo',
  Street: 'OpenStreetMap',
};

let map = null;
let orthoLayer = null;
let drawnItems = null;
let detectionLayer = null;
let heatLayer = null;
let onAoiChange = () => {};
const basemapLayers = {};

// ───────────────────────────────────────────── canvas layers ──

/** Shared plumbing: a full-pane canvas that repaints on every view change. */
const CanvasOverlay = L.Layer.extend({
  initialize(draw) {
    this._draw = draw;
  },

  onAdd(leafletMap) {
    this._map = leafletMap;
    this._canvas = L.DomUtil.create('canvas', 'detection-canvas leaflet-zoom-animated');
    const size = leafletMap.getSize();
    this._canvas.width = size.x;
    this._canvas.height = size.y;
    leafletMap.getPanes().overlayPane.appendChild(this._canvas);

    leafletMap.on('moveend zoomend resize', this._reset, this);
    if (leafletMap.options.zoomAnimation && L.Browser.any3d) {
      leafletMap.on('zoomanim', this._animateZoom, this);
    }
    this._reset();
  },

  onRemove(leafletMap) {
    leafletMap.getPanes().overlayPane.removeChild(this._canvas);
    leafletMap.off('moveend zoomend resize', this._reset, this);
    leafletMap.off('zoomanim', this._animateZoom, this);
  },

  _animateZoom(e) {
    // Slide the canvas with the zoom animation so it doesn't tear mid-flight.
    const scale = this._map.getZoomScale(e.zoom);
    const offset = this._map._latLngBoundsToNewLayerBounds(
      this._map.getBounds(), e.zoom, e.center,
    ).min;
    L.DomUtil.setTransform(this._canvas, offset, scale);
  },

  _reset() {
    if (!this._map) return;
    const size = this._map.getSize();
    if (this._canvas.width !== size.x || this._canvas.height !== size.y) {
      this._canvas.width = size.x;
      this._canvas.height = size.y;
    }
    L.DomUtil.setTransform(this._canvas, this._map.containerPointToLayerPoint([0, 0]), 1);
    this.redraw();
  },

  redraw() {
    if (!this._map || !this._canvas) return;
    const ctx = this._canvas.getContext('2d');
    ctx.clearRect(0, 0, this._canvas.width, this._canvas.height);
    this._draw(ctx, this._map, this._canvas);
  },
});

/** One pre-rendered dot per class, blitted per point — far cheaper than arcs. */
function makeSprite(color, radius) {
  const size = Math.ceil((radius + 1) * 2);
  const sprite = document.createElement('canvas');
  sprite.width = size;
  sprite.height = size;
  const ctx = sprite.getContext('2d');
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, radius, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.lineWidth = 1;
  ctx.strokeStyle = 'rgba(0,0,0,0.55)';
  ctx.stroke();
  return sprite;
}

const SPRITES = { plant: null, weed: null };

function drawDetections(ctx, leafletMap) {
  const { points, minConf, layers } = state;
  if (!points.count || (!layers.plants && !layers.weeds)) return;

  if (!SPRITES.plant) {
    SPRITES.plant = makeSprite(PLANT, MARKER_RADIUS);
    SPRITES.weed = makeSprite(WEED, MARKER_RADIUS);
  }
  const half = SPRITES.plant.width / 2;

  // Cull to the viewport with a small margin so panning stays smooth.
  const bounds = leafletMap.getBounds().pad(0.15);
  const west = bounds.getWest();
  const east = bounds.getEast();
  const south = bounds.getSouth();
  const north = bounds.getNorth();

  const origin = leafletMap.containerPointToLayerPoint([0, 0]);
  let drawn = 0;

  for (let i = 0; i < points.count; i += 1) {
    if (points.conf[i] < minConf) continue;
    const isWeed = points.cls[i] === 1;
    if (isWeed ? !layers.weeds : !layers.plants) continue;

    const lon = points.lon[i];
    const lat = points.lat[i];
    if (lon < west || lon > east || lat < south || lat > north) continue;

    const p = leafletMap.latLngToLayerPoint([lat, lon]);
    ctx.drawImage(SPRITES[isWeed ? 'weed' : 'plant'], p.x - origin.x - half, p.y - origin.y - half);
    drawn += 1;
    // A hard ceiling keeps a pathological zoom-out from stalling the frame.
    if (drawn > 250000) break;
  }
}

// QGIS-style density ramp, matching the reference dashboard's heatmap.
const HEAT_STOPS = [
  [0.00, [44, 0, 100]], [0.15, [31, 0, 255]], [0.30, [0, 180, 255]],
  [0.45, [0, 255, 180]], [0.60, [128, 255, 0]], [0.75, [255, 238, 0]],
  [0.88, [255, 102, 0]], [1.00, [204, 0, 0]],
];

function heatColor(t) {
  for (let i = 1; i < HEAT_STOPS.length; i += 1) {
    const [stop, color] = HEAT_STOPS[i];
    if (t <= stop) {
      const [prevStop, prevColor] = HEAT_STOPS[i - 1];
      const f = (t - prevStop) / Math.max(stop - prevStop, 1e-6);
      return [
        Math.round(prevColor[0] + (color[0] - prevColor[0]) * f),
        Math.round(prevColor[1] + (color[1] - prevColor[1]) * f),
        Math.round(prevColor[2] + (color[2] - prevColor[2]) * f),
      ];
    }
  }
  return HEAT_STOPS[HEAT_STOPS.length - 1][1];
}

/** Three box-blur passes approximate a gaussian closely enough and stay cheap. */
function boxBlur(grid, w, h, radius) {
  const tmp = new Float32Array(grid.length);
  for (let pass = 0; pass < 3; pass += 1) {
    for (let y = 0; y < h; y += 1) {
      let sum = 0;
      const row = y * w;
      for (let x = -radius; x <= radius; x += 1) sum += grid[row + Math.min(Math.max(x, 0), w - 1)];
      for (let x = 0; x < w; x += 1) {
        tmp[row + x] = sum / (radius * 2 + 1);
        sum -= grid[row + Math.min(Math.max(x - radius, 0), w - 1)];
        sum += grid[row + Math.min(Math.max(x + radius + 1, 0), w - 1)];
      }
    }
    for (let x = 0; x < w; x += 1) {
      let sum = 0;
      for (let y = -radius; y <= radius; y += 1) sum += tmp[Math.min(Math.max(y, 0), h - 1) * w + x];
      for (let y = 0; y < h; y += 1) {
        grid[y * w + x] = sum / (radius * 2 + 1);
        sum -= tmp[Math.min(Math.max(y - radius, 0), h - 1) * w + x];
        sum += tmp[Math.min(Math.max(y + radius + 1, 0), h - 1) * w + x];
      }
    }
  }
}

function drawHeat(ctx, leafletMap, canvas) {
  const { points, minConf, layers } = state;
  if (!layers.heat || !points.count) return;

  // Accumulate on a coarse grid, then upscale — a full-resolution blur of the
  // viewport would cost more than the rest of the frame combined.
  const cell = 6;
  const w = Math.max(2, Math.ceil(canvas.width / cell));
  const h = Math.max(2, Math.ceil(canvas.height / cell));
  const grid = new Float32Array(w * h);

  const bounds = leafletMap.getBounds().pad(0.1);
  const origin = leafletMap.containerPointToLayerPoint([0, 0]);
  let hits = 0;

  for (let i = 0; i < points.count; i += 1) {
    if (points.cls[i] !== 1) continue;          // weed density only
    if (points.conf[i] < minConf) continue;
    const lon = points.lon[i];
    const lat = points.lat[i];
    if (lon < bounds.getWest() || lon > bounds.getEast()) continue;
    if (lat < bounds.getSouth() || lat > bounds.getNorth()) continue;

    const p = leafletMap.latLngToLayerPoint([lat, lon]);
    const gx = Math.floor((p.x - origin.x) / cell);
    const gy = Math.floor((p.y - origin.y) / cell);
    if (gx < 0 || gx >= w || gy < 0 || gy >= h) continue;
    grid[gy * w + gx] += 1;
    hits += 1;
  }
  if (!hits) return;

  boxBlur(grid, w, h, 3);

  let peak = 0;
  for (let i = 0; i < grid.length; i += 1) if (grid[i] > peak) peak = grid[i];
  if (peak <= 0) return;

  const field = document.createElement('canvas');
  field.width = w;
  field.height = h;
  const fieldCtx = field.getContext('2d');
  const image = fieldCtx.createImageData(w, h);

  for (let i = 0; i < grid.length; i += 1) {
    const t = grid[i] / peak;
    if (t < 0.04) continue;                      // let the imagery show through
    const [r, g, b] = heatColor(Math.min(t, 1));
    image.data[i * 4] = r;
    image.data[i * 4 + 1] = g;
    image.data[i * 4 + 2] = b;
    image.data[i * 4 + 3] = Math.round(Math.min(t * 1.6, 1) * 205);
  }
  fieldCtx.putImageData(image, 0, 0);

  ctx.save();
  ctx.imageSmoothingEnabled = true;
  ctx.globalCompositeOperation = 'screen';
  ctx.drawImage(field, 0, 0, canvas.width, canvas.height);
  ctx.restore();
}

// ─────────────────────────────────────────────────── public ──

export function initMap(container, { onAoi, onView }) {
  onAoiChange = onAoi || (() => {});

  map = L.map(container, {
    center: [36.7284, -76.5837],
    zoom: 4,
    zoomControl: true,
    attributionControl: true,
    preferCanvas: true,
    maxZoom: 24,
  });

  basemapLayers.Satellite = L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    { attribution: 'Esri World Imagery', maxNativeZoom: 19, maxZoom: 24 },
  );
  basemapLayers.Terrain = L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
    { attribution: 'Esri World Topo', maxNativeZoom: 19, maxZoom: 24 },
  );
  basemapLayers.Street = L.tileLayer(
    'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    { attribution: '© OpenStreetMap contributors', maxNativeZoom: 19, maxZoom: 24 },
  );

  L.control.scale({ position: 'bottomleft', imperial: true, metric: true }).addTo(map);

  drawnItems = new L.FeatureGroup().addTo(map);
  const drawControl = new L.Control.Draw({
    position: 'topleft',
    draw: {
      polygon: { allowIntersection: false, showArea: false, shapeOptions: AOI_STYLE },
      rectangle: { shapeOptions: AOI_STYLE },
      polyline: false, circle: false, marker: false, circlemarker: false,
    },
    edit: { featureGroup: drawnItems, remove: true },
  });
  map.addControl(drawControl);

  map.on(L.Draw.Event.CREATED, (e) => {
    // Only the newest shape counts. Keeping older ones is how a forgotten
    // polygon silently inflates the next plot's numbers.
    drawnItems.clearLayers();
    // Restyle rather than trust the draw handler's options, so a shape added
    // any other way still reads as an AOI.
    e.layer.setStyle?.(AOI_STYLE);
    drawnItems.addLayer(e.layer);
    publishAoi();
  });
  map.on(L.Draw.Event.EDITED, publishAoi);
  map.on(L.Draw.Event.DELETED, publishAoi);

  heatLayer = new CanvasOverlay(drawHeat).addTo(map);
  detectionLayer = new CanvasOverlay(drawDetections).addTo(map);

  const report = () => onView && onView({
    center: map.getCenter(),
    zoom: map.getZoom(),
    scale: metresPerPixel(),
  });
  map.on('move zoom', report);
  map.on('mousemove', (e) => onView && onView({
    center: e.latlng, zoom: map.getZoom(), scale: metresPerPixel(), hover: true,
  }));
  report();

  return map;
}

function metresPerPixel() {
  if (!map) return 0;
  const centre = map.getCenter();
  return 156543.03392 * Math.cos((centre.lat * Math.PI) / 180) / 2 ** map.getZoom();
}

function publishAoi() {
  const layers = drawnItems.getLayers();
  if (!layers.length) {
    onAoiChange(null);
    return;
  }
  const latest = layers[layers.length - 1];
  onAoiChange({ type: 'FeatureCollection', features: [latest.toGeoJSON()] });
}

export function clearAoi() {
  if (drawnItems) drawnItems.clearLayers();
  onAoiChange(null);
}

export function setAoiVisible(visible) {
  if (!map || !drawnItems) return;
  if (visible && !map.hasLayer(drawnItems)) map.addLayer(drawnItems);
  if (!visible && map.hasLayer(drawnItems)) map.removeLayer(drawnItems);
}

export function setBasemap(name) {
  if (!map) return;
  for (const [key, layer] of Object.entries(basemapLayers)) {
    if (key === name) { if (!map.hasLayer(layer)) layer.addTo(map); }
    else if (map.hasLayer(layer)) map.removeLayer(layer);
  }
  if (orthoLayer) {
    if (name === 'Orthomosaic') { if (!map.hasLayer(orthoLayer)) orthoLayer.addTo(map); }
    else if (map.hasLayer(orthoLayer)) map.removeLayer(orthoLayer);
  }
  // Overlays live above the basemap; re-adding a tile layer would cover them.
  if (heatLayer) heatLayer.bringToFront?.();
  redraw();
}

export function setOrthomosaic(dataset) {
  if (!map) return;
  if (orthoLayer) { map.removeLayer(orthoLayer); orthoLayer = null; }
  if (!dataset) return;

  orthoLayer = L.tileLayer(api.tileUrl(dataset.id), {
    attribution: `Orthomosaic · ${dataset.name}`,
    maxNativeZoom: dataset.max_zoom ?? 22,
    maxZoom: 24,
    minZoom: 0,
    bounds: L.latLngBounds(dataset.leaflet_bounds),
    errorTileUrl: 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7',
  });
  if (state.basemap === 'Orthomosaic') orthoLayer.addTo(map);
  fitDataset(dataset);
}

export function reloadOrtho() {
  if (!orthoLayer) return;
  // Cache-bust so a rebuilt pyramid or replaced file is picked up.
  orthoLayer.setUrl(`${api.tileUrl(state.dataset.id)}?v=${Date.now()}`, false);
  orthoLayer.redraw();
}

export function fitDataset(dataset) {
  if (!map || !dataset) return;
  map.fitBounds(L.latLngBounds(dataset.leaflet_bounds), { padding: [24, 24], maxZoom: state.homeZoom });
}

export function goHome() {
  if (state.dataset) fitDataset(state.dataset);
  else if (state.points.count) fitPoints();
}

export function fitPoints() {
  if (!map || !state.points.count) return;
  const { lon, lat } = state.points;
  let west = Infinity; let east = -Infinity; let south = Infinity; let north = -Infinity;
  for (let i = 0; i < state.points.count; i += 1) {
    if (lon[i] < west) west = lon[i];
    if (lon[i] > east) east = lon[i];
    if (lat[i] < south) south = lat[i];
    if (lat[i] > north) north = lat[i];
  }
  if (!Number.isFinite(west)) return;
  map.fitBounds(L.latLngBounds([[south, west], [north, east]]), { padding: [24, 24], maxZoom: 22 });
}

export function redraw() {
  detectionLayer?.redraw();
  heatLayer?.redraw();
}

export function invalidate() {
  map?.invalidateSize();
}

export function getMap() {
  return map;
}
