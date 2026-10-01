"""The HTML report's pieces (private; ``html_report``): the page's style and script, the
report's plain text as HTML, and the status chips.

The text of every answer is the same as ``Report.summary``: ``text_html`` turns its lines
into paragraphs (a wrapped sentence joined again), status lines (the marker as a chip with
its word), list items, muted assumptions and tables (lines whose columns are aligned by two
or more spaces become a sortable table). Nothing is reworded.
"""

from __future__ import annotations

import html
import re

from .._style import Status
from ..glossary import GLOSSARY, PHRASES, Term, slug

STATUS_WORDS = {
    Status.OK: ("ok", "Answered"),
    Status.CAUTION: ("caution", "Caution"),
    Status.PROBLEM: ("problem", "Check the data first"),
    Status.NOT_ASSESSED: ("na", "Not assessed"),
}
_MARKERS = {status.marker: status for status in Status}
_MARKER_LINE = re.compile(r"^(✅|⚠️|❌|➖)\s+(.*)$")
_COLUMNS = re.compile(r"\S {2,}\S")
_SPLIT = re.compile(r" {2,}")
_NUMBER = re.compile(r"^[−+-]?[<>≤≥]?\d[\d.,]*%?$")


class Terms:
    """The glossary's phrases in the report's text, each linked to its entry with its meaning
    as a tooltip: only the first mention of a term in each section (``annotate``), never inside
    a link or a table, so the text stays readable."""

    def __init__(self) -> None:
        by_name = {term.name: term for term in GLOSSARY}
        self.lookup: dict[str, Term] = {}
        for name, phrases in PHRASES.items():
            for phrase in phrases:
                self.lookup[html.escape(phrase).lower()] = by_name[name]
        alternatives = sorted(self.lookup, key=len, reverse=True)
        self.pattern = re.compile(
            r"(?<![\w-])(" + "|".join(re.escape(a) for a in alternatives) + r")(?![\w-])",
            re.IGNORECASE,
        )

    def _link(self, match: re.Match, seen: set[str]) -> str:
        term = self.lookup[match.group(1).lower()]
        if term.name in seen:
            return match.group(0)
        seen.add(term.name)
        return (
            f'<a class="term" href="#{slug(term)}">{match.group(0)}'
            f'<span class="tip" role="tooltip">{html.escape(term.meaning)}</span></a>'
        )

    def annotate(self, markup: str) -> str:
        """``markup`` with the first mention of each term linked (its text, not its tags)."""
        seen: set[str] = set()
        out, in_link, in_table = [], 0, 0
        for piece in re.split(r"(<[^>]+>)", markup):
            if piece.startswith("<"):
                if re.match(r"<a[\s>]", piece):
                    in_link += 1
                elif piece.startswith("</a"):
                    in_link = max(0, in_link - 1)
                elif re.match(r"<table[\s>]", piece):
                    in_table += 1
                elif piece.startswith("</table"):
                    in_table = max(0, in_table - 1)
                out.append(piece)
            # A table scrolls sideways, which would clip a tooltip; its words are explained
            # around it.
            elif in_link or in_table or not piece.strip():
                out.append(piece)
            else:
                out.append(self.pattern.sub(lambda m: self._link(m, seen), piece))
        return "".join(out)


def chip(status: Status | None) -> str:
    """The status as a coloured chip with its icon and word (never colour alone)."""
    if status is None:
        return ""
    kind, word = STATUS_WORDS[status]
    icon = f'<span aria-hidden="true">{status.marker}</span>'
    return f'<span class="chip {kind}">{icon} {word}</span>'


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _table(rows: list[str]) -> str:
    cells = [_SPLIT.split(row.strip()) for row in rows]
    width = len(cells[0])
    if width < 2 or any(len(row) != width for row in cells):
        return "<pre>" + html.escape("\n".join(r.strip() for r in rows)) + "</pre>"
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cells[0])
    body = []
    for row in cells[1:]:
        tds = "".join(
            f'<td class="{"num" if _NUMBER.match(c) else ""}">{html.escape(c)}</td>' for c in row
        )
        body.append(f"<tr>{tds}</tr>")
    return (
        '<div class="table-wrap"><table class="sortable"><thead><tr>'
        + head
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def _paragraph(text: str, level: int) -> str:
    style = f' style="margin-left:{min(level, 4) * 1.1:.1f}em"' if level else ""
    marker = _MARKER_LINE.match(text)
    if marker:
        status = _MARKERS[marker.group(1)]
        kind, word = STATUS_WORDS[status]
        # Inline, the sentence says what the marker found; the icon keeps its word for readers
        # that cannot see it.
        icon = f'<span class="mark {kind}" role="img" aria-label="{word}">{status.marker}</span>'
        return f'<p class="status-line"{style}>{icon} {html.escape(marker.group(2))}</p>'
    if text.startswith("- "):
        return f'<p class="item"{style}>{html.escape(text[2:])}</p>'
    if text.startswith("Assumptions:"):
        body = html.escape(text[len("Assumptions:") :].strip())
        return f'<p class="assumptions"{style}><strong>Assumptions</strong> {body}</p>'
    return f"<p{style}>{html.escape(text)}</p>"


def text_html(text: str) -> str:
    """The report's text as HTML (see the module)."""
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    base = min((_indent(line) for line in lines if line.strip()), default=0)
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if _COLUMNS.search(line.strip()):
            block = [line]
            while (
                i + 1 < len(lines)
                and lines[i + 1].strip()
                and _COLUMNS.search(lines[i + 1].strip())
            ):
                i += 1
                block.append(lines[i])
            if len(block) >= 2:
                out.append(_table(block))
                i += 1
                continue
        first = _indent(line)
        parts = [line.strip()]
        while (
            i + 1 < len(lines)
            and lines[i + 1].strip()
            and _indent(lines[i + 1]) > first
            and not _MARKER_LINE.match(lines[i + 1].strip())
            and not lines[i + 1].strip().startswith("- ")
            and not _COLUMNS.search(lines[i + 1].strip())
        ):
            i += 1
            parts.append(lines[i].strip())
        out.append(_paragraph(" ".join(parts), (first - base) // 2))
        i += 1
    return "\n".join(out)


CSS = """
:root {
  color-scheme: light;
  --bg: #f6f6f4; --surface: #fcfcfb; --surface-2: #f0efec; --border: #dddcd7;
  --text: #0b0b0b; --text-2: #52514e; --text-3: #6f6e69; --link: #1c5cab;
  --ok: #0a7d0a; --ok-bg: #e3f3e3; --caution: #8a5a00; --caution-bg: #fdf1d6;
  --problem: #b02a2a; --problem-bg: #fbe3e3; --na: #52514e; --na-bg: #ebeae6;
  --grid: #e6e5e1;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --bg: #121211; --surface: #1a1a19; --surface-2: #232321; --border: #383835;
    --text: #ffffff; --text-2: #c3c2b7; --text-3: #9b9a92; --link: #86b6ef;
    --ok: #5fd35f; --ok-bg: #173117; --caution: #f5c04a; --caution-bg: #3a2e12;
    --problem: #f08a8a; --problem-bg: #3d1a1a; --na: #c3c2b7; --na-bg: #2b2b29;
    --grid: #2e2e2c;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #121211; --surface: #1a1a19; --surface-2: #232321; --border: #383835;
  --text: #ffffff; --text-2: #c3c2b7; --text-3: #9b9a92; --link: #86b6ef;
  --ok: #5fd35f; --ok-bg: #173117; --caution: #f5c04a; --caution-bg: #3a2e12;
  --problem: #f08a8a; --problem-bg: #3d1a1a; --na: #c3c2b7; --na-bg: #2b2b29;
  --grid: #2e2e2c;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
a { color: var(--link); }
.layout { display: grid; grid-template-columns: 270px minmax(0, 1fr); max-width: 1320px;
  margin: 0 auto; gap: 28px; padding: 24px 20px 64px; }
nav.toc { position: sticky; top: 16px; align-self: start; max-height: calc(100vh - 32px);
  overflow: auto; font-size: 13.5px; }
nav.toc h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .06em;
  color: var(--text-3); margin: 0 0 8px; }
nav.toc a { display: flex; gap: 8px; align-items: baseline; padding: 5px 8px;
  border-radius: 6px; color: var(--text); text-decoration: none; }
nav.toc a:hover { background: var(--surface-2); }
nav.toc .icon { width: 1.3em; flex: none; }
header.top { margin-bottom: 18px; }
header.top h1 { font-size: 26px; margin: 0 0 4px; letter-spacing: -.01em; }
header.top .sub { color: var(--text-2); margin: 0; }
header.top .meta { color: var(--text-3); font-size: 13px; margin-top: 6px; }
.theme-toggle { float: right; border: 1px solid var(--border); background: var(--surface);
  color: var(--text-2); border-radius: 8px; padding: 6px 10px; font: inherit; font-size: 13px;
  cursor: pointer; }
section.card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
  padding: 18px 22px; margin: 0 0 18px; scroll-margin-top: 16px; }
section.card > h2 { font-size: 18px; margin: 0 0 10px; display: flex; gap: 10px;
  align-items: center; flex-wrap: wrap; }
.glance-row { display: grid; grid-template-columns: 190px minmax(0, 1fr); gap: 12px;
  padding: 9px 0; border-top: 1px solid var(--grid); text-decoration: none; color: inherit; }
.glance-row:first-of-type { border-top: 0; }
.glance-row .q { font-weight: 600; }
.glance-row .a { color: var(--text-2); }
.glance-row:hover .q { color: var(--link); }
.chip { display: inline-flex; gap: 4px; align-items: center; font-size: 12px; font-weight: 600;
  padding: 2px 8px; border-radius: 999px; white-space: nowrap; }
.chip.ok { color: var(--ok); background: var(--ok-bg); }
.chip.caution { color: var(--caution); background: var(--caution-bg); }
.chip.problem { color: var(--problem); background: var(--problem-bg); }
.chip.na { color: var(--na); background: var(--na-bg); }
p { margin: 6px 0; }
p.item::before { content: "•"; color: var(--text-3); margin-right: .5em; }
p.assumptions { color: var(--text-3); font-size: 13.5px; }
p.status-line { padding-left: 1.6em; text-indent: -1.6em; }
p.status-line .mark { display: inline-block; width: 1.6em; text-indent: 0; }
.key { color: var(--text-3); font-size: 13px; margin: 0 0 8px; }
.table-wrap { overflow-x: auto; margin: 8px 0 12px; }
table { border-collapse: collapse; font-size: 13.5px; min-width: 50%; }
th, td { text-align: left; padding: 5px 12px 5px 0; border-bottom: 1px solid var(--grid);
  vertical-align: top; white-space: nowrap; }
th { color: var(--text-2); font-weight: 600; cursor: pointer; user-select: none; }
th[aria-sort="ascending"]::after { content: " ▲"; font-size: 10px; }
th[aria-sort="descending"]::after { content: " ▼"; font-size: 10px; }
td.num { font-variant-numeric: tabular-nums; text-align: right; }
pre { background: var(--surface-2); padding: 10px 12px; border-radius: 8px; overflow-x: auto;
  font-size: 12.5px; }
details { margin-top: 12px; border-top: 1px solid var(--grid); padding-top: 10px; }
details > summary { cursor: pointer; color: var(--link); font-weight: 600; font-size: 14px; }
.figures { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 14px; margin-top: 10px; }
figure { margin: 0; background: #ffffff; border: 1px solid var(--border); border-radius: 10px;
  padding: 8px; }
figure img { width: 100%; height: auto; display: block; }
figcaption { font-size: 12.5px; color: #52514e; padding: 6px 4px 2px; }
figcaption a { color: #1c5cab; }
.viz { margin-top: 12px; }
.controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 8px 0; }
.seg { display: inline-flex; border: 1px solid var(--border); border-radius: 8px;
  overflow: hidden; }
.seg button { border: 0; background: var(--surface); color: var(--text-2); padding: 6px 12px;
  font: inherit; font-size: 13px; cursor: pointer; }
.seg button[aria-pressed="true"] { background: var(--surface-2); color: var(--text);
  font-weight: 600; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 14px; font-size: 13px; margin: 6px 0; }
.legend button { display: inline-flex; gap: 6px; align-items: center; border: 0;
  background: none; color: var(--text); font: inherit; font-size: 13px; cursor: pointer;
  padding: 2px 0; }
.legend button[aria-pressed="false"] { opacity: .4; }
.swatch { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.plot-box { position: relative; width: 100%; max-width: 720px; }
.plot-box svg { width: 100%; height: auto; display: block; background: var(--surface);
  color: var(--text);
  border: 1px solid var(--grid); border-radius: 8px; }
.plot-box .axis { fill: var(--text-3); font-size: 11px; }
.plot-box .arrow { stroke: var(--text); stroke-width: 1.5; }
.plot-box .arrow-label { fill: var(--text); font-size: 11px; font-weight: 600;
  paint-order: stroke; stroke: var(--surface); stroke-width: 3px; }
.tooltip { position: absolute; pointer-events: none; background: var(--surface);
  color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 6px 8px;
  font-size: 12px; box-shadow: 0 2px 8px rgba(0,0,0,.15); display: none; white-space: nowrap; }
.note { color: var(--text-3); font-size: 13px; }
a.term { color: inherit; text-decoration: underline dotted var(--text-3);
  text-underline-offset: 3px; position: relative; }
a.term .tip { display: none; position: absolute; left: 0; top: 1.5em; z-index: 10;
  width: max-content; max-width: min(340px, 80vw); white-space: normal;
  background: var(--surface); color: var(--text);
  border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; font-size: 13px;
  font-weight: 400; line-height: 1.45; box-shadow: 0 4px 14px rgba(0,0,0,.18); text-align: left; }
a.term:hover .tip, a.term:focus .tip { display: block; }
dl.glossary dt:target { background: var(--caution-bg); border-radius: 4px; }
button.dl { border: 1px solid var(--border); background: var(--surface); color: var(--link);
  border-radius: 6px; padding: 1px 8px; font: inherit; font-size: 12px; cursor: pointer; }
figcaption button.dl { background: #ffffff; color: #1c5cab; border-color: #dddcd7; }
button.dl:hover { background: var(--surface-2); }
dl.glossary { margin: 6px 0 14px; }
dl.glossary dt { font-weight: 600; margin-top: 10px; }
dl.glossary dd { margin: 2px 0 0; }
dl.glossary .where, dl.glossary .refs { display: block; color: var(--text-3); font-size: 13px; }
section.card h3 { font-size: 15px; margin: 18px 0 6px; }
footer { color: var(--text-3); font-size: 13px; margin-top: 24px; }
@media (max-width: 900px) {
  .layout { grid-template-columns: minmax(0, 1fr); padding: 16px; gap: 12px; }
  nav.toc { position: static; max-height: none; }
  .glance-row { grid-template-columns: minmax(0, 1fr); gap: 2px; }
  section.card { padding: 14px 16px; }
}
@media print {
  nav.toc, .theme-toggle, .controls, button.dl, #files { display: none; }
  .layout { display: block; }
  details { display: block; }
  section.card { break-inside: avoid-page; }
}
"""

SCRIPT = """
(function () {
  // Sortable tables: click a header to sort by it.
  document.querySelectorAll('table.sortable').forEach(function (table) {
    table.querySelectorAll('th').forEach(function (th, column) {
      th.addEventListener('click', function () {
        var body = table.tBodies[0], rows = Array.from(body.rows);
        var down = th.getAttribute('aria-sort') !== 'descending';
        table.querySelectorAll('th').forEach(function (h) { h.removeAttribute('aria-sort'); });
        th.setAttribute('aria-sort', down ? 'descending' : 'ascending');
        function key(row) {
          var text = row.cells[column].textContent.trim();
          var number = parseFloat(text.replace(/[<>≤≥%+]/g, '').replace('−', '-'));
          return isNaN(number) ? text.toLowerCase() : number;
        }
        rows.sort(function (a, b) {
          var x = key(a), y = key(b);
          if (x < y) return down ? 1 : -1;
          if (x > y) return down ? -1 : 1;
          return 0;
        });
        rows.forEach(function (row) { body.appendChild(row); });
      });
    });
  });
  // Light and dark: the viewer's choice wins over the system's.
  var toggle = document.querySelector('.theme-toggle');
  if (toggle) toggle.addEventListener('click', function () {
    var root = document.documentElement;
    var dark = root.getAttribute('data-theme') === 'dark' ||
      (!root.getAttribute('data-theme') && matchMedia('(prefers-color-scheme: dark)').matches);
    root.setAttribute('data-theme', dark ? 'light' : 'dark');
    try { localStorage.setItem('mc-theme', dark ? 'light' : 'dark'); } catch (e) {}
    document.querySelectorAll('.plot-box').forEach(function (box) { box.redraw && box.redraw(); });
  });
  try {
    var saved = localStorage.getItem('mc-theme');
    if (saved) document.documentElement.setAttribute('data-theme', saved);
  } catch (e) {}
  // Downloads: every file the page carries, saved from the data inside it.
  var store = document.getElementById('files-data');
  var files = store ? JSON.parse(store.textContent) : {};
  document.addEventListener('click', function (event) {
    var button = event.target.closest('button.dl');
    if (!button) return;
    var entry = files[button.dataset.file];
    if (!entry) return;
    var bytes = atob(entry[1]), buffer = new Uint8Array(bytes.length);
    for (var i = 0; i < bytes.length; i++) buffer[i] = bytes.charCodeAt(i);
    var url = URL.createObjectURL(new Blob([buffer], { type: entry[0] }));
    var link = document.createElement('a');
    link.href = url; link.download = button.dataset.file.split('/').pop();
    document.body.appendChild(link); link.click(); link.remove();
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  });
  // The interactive embeddings.
  var source = document.getElementById('embedding-data');
  if (!source) return;
  var data = JSON.parse(source.textContent);
  document.querySelectorAll('.viz[data-viz]').forEach(function (viz) { scatter(viz, data); });

  function scatter(viz, data) {
    var state = { method: data.methods[0], values: data.values[0], hidden: {} };
    var box = viz.querySelector('.plot-box'), tip = box.querySelector('.tooltip');
    var legend = viz.querySelector('.legend');
    data.conditions.forEach(function (name, i) {
      var b = document.createElement('button');
      b.type = 'button'; b.setAttribute('aria-pressed', 'true');
      b.innerHTML = '<span class="swatch" style="background:' + data.colours[i] + '"></span>' +
        escape(name) + ' <span class="note">(' + data.counts[i] + ')</span>';
      b.addEventListener('click', function () {
        state.hidden[i] = !state.hidden[i];
        b.setAttribute('aria-pressed', state.hidden[i] ? 'false' : 'true');
        draw();
      });
      legend.appendChild(b);
    });
    viz.querySelectorAll('.seg button').forEach(function (b) {
      b.addEventListener('click', function () {
        state[b.dataset.key] = b.dataset.value;
        viz.querySelectorAll('.seg button[data-key="' + b.dataset.key + '"]').forEach(
          function (o) { o.setAttribute('aria-pressed', o === b ? 'true' : 'false'); });
        draw();
      });
    });
    var W = 760, H = 560, M = 44, placed = [];
    function draw() {
      var set = data.sets[state.values][state.method];
      var caption = viz.querySelector('.caption');
      caption.textContent = set.caption;
      var xs = set.x, ys = set.y, arrows = set.arrows || [];
      // The range holds every point and every arrow (from the origin), with a margin.
      var ax = arrows.map(function (a) { return a[1]; });
      var ay = arrows.map(function (a) { return a[2]; });
      var all = [xs.concat(ax, arrows.length ? [0] : []), ys.concat(ay, arrows.length ? [0] : [])];
      var lo = [Math.min.apply(null, all[0]), Math.min.apply(null, all[1])];
      var hi = [Math.max.apply(null, all[0]), Math.max.apply(null, all[1])];
      [0, 1].forEach(function (k) {
        var pad = (hi[k] - lo[k]) * 0.04 || 1; lo[k] -= pad; hi[k] += pad;
      });
      function sx(v) { return M + (v - lo[0]) / ((hi[0] - lo[0]) || 1) * (W - 2 * M); }
      function sy(v) { return H - M - (v - lo[1]) / ((hi[1] - lo[1]) || 1) * (H - 2 * M); }
      var svg = ['<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="' +
        escape(set.caption) + '">'];
      svg.push('<text class="axis" x="' + W / 2 + '" y="' + (H - 10) +
        '" text-anchor="middle">' + escape(set.xlabel) + '</text>');
      svg.push('<text class="axis" x="14" y="' + H / 2 +
        '" text-anchor="middle" transform="rotate(-90 14 ' + H / 2 + ')">' +
        escape(set.ylabel) + '</text>');
      placed = [];
      // The control first, so the conditions sit on top of it.
      for (var i = 0; i < xs.length; i++) {
        var c = set.cond[i];
        if (state.hidden[c]) continue;
        var px = sx(xs[i]), py = sy(ys[i]);
        placed.push([px, py, i]);
        svg.push('<circle cx="' + px.toFixed(1) + '" cy="' + py.toFixed(1) +
          '" r="3" fill="' + data.colours[c] + '" fill-opacity="0.6"/>');
      }
      // Labels of near-parallel arrows would overlap: each moves down past the ones placed.
      var labels = [];
      arrows.slice().sort(function (a, b) { return sy(a[2]) - sy(b[2]); }).forEach(function (a) {
        var x0 = sx(0), y0 = sy(0), x1 = sx(a[1]), y1 = sy(a[2]);
        var lx = x1 + 4, ly = y1 - 4, wide = 6.5 * a[0].length;
        if (lx + wide > W - 4) lx = x1 - 4 - wide;
        labels.forEach(function (b) {
          var near = Math.abs(b[0] - lx) < Math.max(wide, b[2]);
          if (near && Math.abs(b[1] - ly) < 13) ly = b[1] + 13;
        });
        labels.push([lx, ly, wide]);
        svg.push('<line class="arrow" x1="' + x0 + '" y1="' + y0 + '" x2="' + x1 +
          '" y2="' + y1 + '"/>');
        svg.push('<circle cx="' + x1 + '" cy="' + y1 + '" r="2.5" fill="currentColor"/>');
        svg.push('<text class="arrow-label" x="' + lx + '" y="' + ly + '">' +
          escape(a[0]) + '</text>');
      });
      svg.push('</svg>');
      box.querySelector('.svg-slot').innerHTML = svg.join('');
    }
    box.redraw = draw;
    box.addEventListener('mousemove', function (event) {
      var svg = box.querySelector('svg');
      if (!svg) return;
      var r = svg.getBoundingClientRect(), k = W / r.width;
      var mx = (event.clientX - r.left) * k, my = (event.clientY - r.top) * k, best = null, d = 144;
      placed.forEach(function (p) {
        var e = (p[0] - mx) * (p[0] - mx) + (p[1] - my) * (p[1] - my);
        if (e < d) { d = e; best = p; }
      });
      if (!best) { tip.style.display = 'none'; return; }
      var i = best[2];
      var set = data.sets[state.values][state.method];
      tip.innerHTML = '<strong>' + escape(data.conditions[set.cond[i]]) + '</strong><br>' +
        escape(set.labels[i]);
      tip.style.display = 'block';
      var left = best[0] / k + 12, top = best[1] / k + 12;
      if (left > r.width - 180) left = best[0] / k - 180;
      tip.style.left = left + 'px'; tip.style.top = top + 'px';
    });
    box.addEventListener('mouseleave', function () { tip.style.display = 'none'; });
    draw();
  }
  function escape(text) {
    return String(text).replace(/[&<>"]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
    });
  }
})();
"""

__all__ = ["CSS", "SCRIPT", "STATUS_WORDS", "Terms", "chip", "text_html"]
