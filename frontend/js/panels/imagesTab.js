/** The detect-on-images tab: batch inference on loose JPG/PNG tiles. */

import { api } from '../api.js';
import { state, setState } from '../state.js';
import { $, clear, dropzone, el, fillTable, fmt, notifyError, pct, toast } from '../ui.js';

export function initImagesTab() {
  const zone = $('#images-drop');
  const input = $('#images-file');
  const run = $('#run-images');

  const setZone = dropzone(zone, input, (files) => {
    const images = files.filter((f) => /\.(jpe?g|png)$/i.test(f.name));
    setState({ imageFiles: images });
    setZone(
      images.length
        ? `<b>${images.length} image${images.length === 1 ? '' : 's'} ready</b><br>`
          + `<span class="faint">${images.slice(0, 3).map((f) => f.name).join(', ')}`
          + `${images.length > 3 ? ` +${images.length - 3} more` : ''}</span>`
        : 'Drop images here or <span class="link">browse files</span>',
      images.length > 0,
    );
    run.disabled = images.length === 0;
  }, { multiple: true });

  const confInput = $('#img-conf');
  confInput.addEventListener('input', () => {
    const value = Number(confInput.value);
    setState({ imgConf: value });
    $('#img-conf-label').textContent = value.toFixed(2);
  });

  run.addEventListener('click', runBatch);
  $('#download-batch').addEventListener('click', downloadBatch);
  renderBatch([]);
}

async function runBatch() {
  const run = $('#run-images');
  if (!state.model.loaded) {
    toast('Load a model in the sidebar before running image detection.', 'warn');
    return;
  }
  run.disabled = true;
  run.textContent = 'Running…';
  $('#images-note').textContent = `Scoring ${state.imageFiles.length} image(s)…`;

  try {
    const { results } = await api.detectImages(state.imageFiles, state.imgConf);
    setState({ batch: results });
    renderBatch(results);
    renderPreviews(results);
    const failures = results.filter((r) => r.error).length;
    $('#images-note').textContent = failures
      ? `${results.length - failures} scored, ${failures} could not be read.`
      : `${results.length} image(s) scored at confidence ${state.imgConf.toFixed(2)}.`;
  } catch (err) {
    notifyError(err, 'Image detection failed');
  } finally {
    run.disabled = state.imageFiles.length === 0;
    run.textContent = 'Run detection on images';
  }
}

function renderBatch(results) {
  const rows = results
    .filter((r) => !r.error)
    .map((r) => [r.name, fmt(r.plants), fmt(r.weeds), fmt(r.total), pct(r.weed_pressure_pct)]);
  fillTable($('#batch-table'),
    ['Image', 'Plants', 'Weeds', 'Total', 'Weed pressure'],
    rows,
    'Upload image tiles and run detection to populate this table.');
  $('#download-batch').disabled = rows.length === 0;
}

function renderPreviews(results) {
  const host = clear($('#image-previews'));
  for (const result of results) {
    if (result.error) {
      host.append(el('div', { class: 'card' },
        el('div', { class: 'card-head', text: result.name }),
        el('div', { class: 'card-note', text: result.error }),
      ));
      continue;
    }
    host.append(el('div', { class: 'card' },
      el('div', { class: 'card-head', text: `${result.name} — ${result.plants} plants · ${result.weeds} weeds` }),
      el('img', { class: 'preview-img', src: result.preview, alt: `Annotated detections for ${result.name}` }),
    ));
  }
}

function downloadBatch() {
  const rows = state.batch.filter((r) => !r.error);
  if (!rows.length) return;
  const header = 'image,plants,weeds,total,weed_pressure_pct,avg_confidence_pct';
  const body = rows.map((r) =>
    [r.name, r.plants, r.weeds, r.total, r.weed_pressure_pct, r.avg_confidence].join(','));
  const blob = new Blob([[header, ...body].join('\n')], { type: 'text/csv' });

  const url = URL.createObjectURL(blob);
  const anchor = Object.assign(document.createElement('a'), { href: url, download: 'batch_results.csv' });
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}
