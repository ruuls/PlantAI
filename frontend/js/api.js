/**
 * Thin wrapper over the PlantAI HTTP API.
 *
 * Every call funnels through `request` so a backend error surfaces as a real
 * Error carrying the server's own message rather than an opaque status code.
 */

const BASE = (window.PLANTAI_API_BASE || '').replace(/\/$/, '');

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${BASE}${path}`, options);
  } catch (err) {
    throw new ApiError(`Cannot reach the detection service: ${err.message}`, 0);
  }
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body && body.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
    } catch { /* non-JSON error body — keep the status line */ }
    throw new ApiError(detail, response.status);
  }
  return response;
}

const json = async (path, options) => (await request(path, options)).json();

const postJson = (path, body) =>
  json(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body ?? {}),
  });

/** Download a blob endpoint straight to the user's disk. */
async function download(path, filename, options) {
  const response = await request(path, options);
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const anchor = Object.assign(document.createElement('a'), { href: url, download: filename });
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  // Revoking immediately can cancel the download in some browsers.
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

export const api = {
  base: BASE,

  health: () => json('/api/health'),
  config: () => json('/api/config'),

  modelStatus: () => json('/api/model'),
  loadModel: (path) => postJson('/api/model', { path }),
  library: (kind) => json(`/api/library?kind=${encodeURIComponent(kind)}`),

  datasetByPath: (path) => postJson('/api/datasets/path', { path }),
  dataset: (id) => json(`/api/datasets/${id}`),
  buildOverviews: (id) => postJson(`/api/datasets/${id}/overviews`),
  tileUrl: (id) => `${BASE}/api/datasets/${id}/tiles/{z}/{x}/{y}.png`,

  /** Uploads report progress, so this one uses XHR rather than fetch. */
  uploadDataset(file, onProgress) {
    return new Promise((resolve, reject) => {
      const form = new FormData();
      form.append('file', file);
      const xhr = new XMLHttpRequest();
      xhr.open('POST', `${BASE}/api/datasets/upload`);
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
      };
      xhr.onload = () => {
        let body = {};
        try { body = JSON.parse(xhr.responseText); } catch { /* keep {} */ }
        if (xhr.status >= 200 && xhr.status < 300) resolve(body);
        else reject(new ApiError(body.detail || `Upload failed (${xhr.status})`, xhr.status));
      };
      xhr.onerror = () => reject(new ApiError('Upload failed: the connection dropped.', 0));
      xhr.send(form);
    });
  },

  uploadDetectionCsv(file, sourceCrs, datasetId) {
    const form = new FormData();
    form.append('file', file);
    form.append('source_crs', sourceCrs || 'EPSG:32618');
    if (datasetId) form.append('dataset_id', datasetId);
    return json('/api/detections/csv', { method: 'POST', body: form });
  },

  detections: (id, minConf = 0) => json(`/api/detections/${id}?min_conf=${minConf}`),

  /** The compact binary point payload the map layer decodes. */
  async points(id, minConf = 0) {
    const response = await request(`/api/detections/${id}/points?min_conf=${minConf}`);
    return response.arrayBuffer();
  },

  aoiCounts: (id, aoi, minConf) => postJson(`/api/detections/${id}/aoi`, { aoi, min_conf: minConf }),
  analysis: (id, body) => postJson(`/api/detections/${id}/analysis`, body),

  startDetection: (body) => postJson('/api/detect', body),
  job: (id) => json(`/api/jobs/${id}`),
  cancelJob: (id) => postJson(`/api/jobs/${id}/cancel`),
  jobEventsUrl: (id) => `${BASE}/api/jobs/${id}/events`,

  detectImages(files, conf) {
    const form = new FormData();
    for (const file of files) form.append('files', file);
    form.append('conf', String(conf));
    return json('/api/images/detect', { method: 'POST', body: form });
  },

  downloadDetections: (id, minConf) =>
    download(`/api/detections/${id}/export.csv?min_conf=${minConf}`, 'detections.csv'),

  downloadAoi: (id, aoi, minConf) =>
    download(`/api/detections/${id}/aoi/export.csv`, 'aoi_detections.csv', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ aoi, min_conf: minConf }),
    }),

  downloadAnalysis: (id, kind, body) =>
    download(`/api/detections/${id}/exports/${kind}`, `${kind === 'stand' ? 'stand_summary' : kind}.csv`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),

  downloadReport: (id, body) =>
    download(`/api/detections/${id}/report.pdf`, 'plantai_report.pdf', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
};

/**
 * Decode the binary detection payload into typed arrays.
 *
 * Coordinates travel as float32 offsets from a float64 reference point, which
 * keeps them under a millimetre of error while halving the payload.
 */
export function decodePoints(buffer) {
  const empty = { count: 0, lon: new Float64Array(0), lat: new Float64Array(0), conf: new Float32Array(0), cls: new Uint8Array(0) };
  if (!buffer || buffer.byteLength < 24) return empty;

  const view = new DataView(buffer);
  const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  if (magic !== 'PAI1') return empty;

  const count = view.getUint32(4, true);
  if (count === 0) return empty;

  const refLon = view.getFloat64(8, true);
  const refLat = view.getFloat64(16, true);

  let offset = 24;
  const lonOff = new Float32Array(buffer, offset, count); offset += count * 4;
  const latOff = new Float32Array(buffer, offset, count); offset += count * 4;
  const conf = new Float32Array(buffer.slice(offset, offset + count * 4)); offset += count * 4;
  const cls = new Uint8Array(buffer.slice(offset, offset + count));

  const lon = new Float64Array(count);
  const lat = new Float64Array(count);
  for (let i = 0; i < count; i += 1) {
    lon[i] = refLon + lonOff[i];
    lat[i] = refLat + latOff[i];
  }
  return { count, lon, lat, conf, cls };
}
