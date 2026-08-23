# Vendored map libraries

Committed rather than pulled from a CDN so the console works on a field laptop
with no internet — the same machine that has the orthomosaic on it.

| Library | Version | License | Files |
|---|---|---|---|
| [Leaflet](https://leafletjs.com) | 1.9.4 | BSD-2-Clause (`LICENSE-leaflet.txt`) | `leaflet.js`, `leaflet.css`, `images/` |
| [Leaflet.draw](https://github.com/Leaflet/Leaflet.draw) | 1.0.4 | MIT | `leaflet.draw.js`, `leaflet.draw.css`, `images/spritesheet*` |

`images/spritesheet-dark.svg` is Leaflet.draw's own spritesheet with its two
fill colours swapped for lighter ones — the stock icons are drawn in `#464646`,
which is invisible on this console's dark toolbar. `css/app.css` points the
draw toolbar at it.

To refresh either library:

```bash
npm pack leaflet@1.9.4 leaflet-draw@1.0.4
tar xzf leaflet-1.9.4.tgz && cp package/dist/leaflet.{js,css} frontend/vendor/
tar xzf leaflet-draw-1.0.4.tgz && cp package/dist/leaflet.draw.{js,css} frontend/vendor/
# then regenerate the recoloured sprite:
sed -e 's/fill:#464646/fill:#b6c0cc/g' -e 's/fill:#bbbbbb/fill:#5c6672/g' \
    frontend/vendor/images/spritesheet.svg > frontend/vendor/images/spritesheet-dark.svg
```
