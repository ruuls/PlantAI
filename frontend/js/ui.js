/** DOM helpers and small shared widgets. */

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'class') node.className = value;
    else if (key === 'html') node.innerHTML = value;
    else if (key === 'text') node.textContent = value;
    else if (key === 'style' && typeof value === 'object') Object.assign(node.style, value);
    else if (key.startsWith('on') && typeof value === 'function') {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else node.setAttribute(key, value === true ? '' : String(value));
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
  return node;
}

// ─────────────────────────────────────────────── formatting ──
export const fmt = (n) => Number.isFinite(n) ? Math.round(n).toLocaleString('en-US') : '—';
export const fmt1 = (n) => Number.isFinite(n) ? n.toFixed(1) : '—';
export const fmt2 = (n) => Number.isFinite(n) ? n.toFixed(2) : '—';
export const pct = (n) => Number.isFinite(n) ? `${n.toFixed(1)}%` : '—';

/**
 * Acreage with enough precision to be useful at both ends of the range.
 * A 75 ft² research plot is 0.0017 acres; fixed 4 decimals leaves it with two
 * significant digits, so small values get more places instead.
 */
export function acres(n) {
  if (!Number.isFinite(n)) return '—';
  if (n === 0) return '0';
  if (n >= 100) return n.toFixed(1);
  if (n >= 1) return n.toFixed(3);
  const places = Math.min(8, 3 - Math.floor(Math.log10(Math.abs(n))));
  return n.toFixed(places);
}

export function bytes(n) {
  if (!Number.isFinite(n)) return '—';
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  return `${(n / 1e3).toFixed(0)} KB`;
}

// ────────────────────────────────────────────────── widgets ──
export function statCell(label, value) {
  return el('div', { class: 'stat-cell' },
    el('span', { class: 'field-label', text: label }),
    el('span', { class: 'stat-value', text: value }),
  );
}

export function headlineCell(label, value) {
  return el('div', { class: 'headline-cell' },
    el('span', { class: 'field-label', text: label }),
    el('span', { class: 'headline-value', text: value }),
  );
}

export function slider({ label, min, max, step, value, display, onInput, width }) {
  const readout = el('span', { class: 'num', text: display });
  const input = el('input', {
    type: 'range', min, max, step, value,
    onInput: (e) => {
      const next = Number(e.target.value);
      readout.textContent = onInput(next) ?? String(next);
    },
  });
  const block = el('div', { class: 'slider-block' },
    el('div', { class: 'row-between' }, el('span', { class: 'field-label', text: label }), readout),
    input,
  );
  if (width) block.style.width = width;
  return block;
}

export function radioRow(label, isOn, onClick, extraClass = '') {
  return el('label', {
    class: `radio-row ${isOn ? 'is-on' : ''} ${extraClass}`.trim(),
    onClick,
  },
    el('span', { class: 'radio' }),
    el('span', { class: 'radio-text', text: label }),
  );
}

export function setBanner(node, tone, title, detail) {
  node.className = `banner banner-${tone || 'amber'}`;
  clear(node);
  node.append(
    el('div', { class: 'banner-title', text: title }),
    el('div', { class: 'banner-detail', text: detail }),
  );
}

export function table(headers, rows, emptyMessage) {
  const thead = el('thead', {}, el('tr', {}, headers.map((h) => el('th', { text: h }))));
  const body = el('tbody');
  if (!rows.length) {
    body.append(el('tr', { class: 'empty-row' },
      el('td', { colspan: headers.length, text: emptyMessage || 'Nothing to show yet.' })));
  } else {
    for (const row of rows) {
      body.append(el('tr', {}, row.map((cell) =>
        cell instanceof Node ? el('td', {}, cell) : el('td', { text: String(cell) }))));
    }
  }
  return [thead, body];
}

export function fillTable(tableNode, headers, rows, emptyMessage) {
  clear(tableNode).append(...table(headers, rows, emptyMessage));
}

// ──────────────────────────────────────────────────── toasts ──
const TOAST_MS = { info: 4200, ok: 3600, warn: 6000, error: 9000 };

export function toast(message, kind = 'info', title = '') {
  const stack = $('#toasts');
  if (!stack) return;
  const node = el('div', { class: `toast toast-${kind}` },
    title ? el('div', { class: 'toast-title', text: title }) : null,
    el('div', { text: message }),
  );
  stack.append(node);
  setTimeout(() => {
    node.style.transition = 'opacity .2s';
    node.style.opacity = '0';
    setTimeout(() => node.remove(), 220);
  }, TOAST_MS[kind] ?? 4200);
}

export const notifyError = (err, title = 'Something went wrong') =>
  toast(err instanceof Error ? err.message : String(err), 'error', title);

/**
 * Wire a label-wrapped file input up as a drop target.
 * Returns a setter used to reflect the chosen file back in the zone.
 */
export function dropzone(zone, input, onFiles, { multiple = false } = {}) {
  const stop = (e) => { e.preventDefault(); e.stopPropagation(); };

  ['dragenter', 'dragover'].forEach((evt) =>
    zone.addEventListener(evt, (e) => { stop(e); zone.classList.add('is-hover'); }));
  ['dragleave', 'drop'].forEach((evt) =>
    zone.addEventListener(evt, (e) => { stop(e); zone.classList.remove('is-hover'); }));

  zone.addEventListener('drop', (e) => {
    const files = Array.from(e.dataTransfer?.files || []);
    if (files.length) onFiles(multiple ? files : [files[0]]);
  });
  input.addEventListener('change', () => {
    const files = Array.from(input.files || []);
    if (files.length) onFiles(multiple ? files : [files[0]]);
  });

  return (text, isSet) => {
    zone.innerHTML = '';
    zone.append(input, document.createTextNode(''));
    zone.insertAdjacentHTML('beforeend', text);
    zone.classList.toggle('is-set', Boolean(isSet));
  };
}
