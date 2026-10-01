"""The HTML report's style sheet (private; ``html_report``): light and dark from the same
variables, the sidebar, the cards, the tables, the figures and their enlarged view, the
interactive embeddings and the glossary drawer."""

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
.layout { display: grid; grid-template-columns: 270px minmax(0, 1fr); max-width: 2200px;
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
.swatch { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.note { color: var(--text-3); font-size: 13px; }
a.term { color: inherit; text-decoration: underline dotted var(--text-3);
  text-underline-offset: 3px; position: relative; }
a.term .tip { display: none; position: absolute; left: 0; top: 1.5em; z-index: 10;
  width: max-content; max-width: min(340px, 80vw); white-space: normal;
  background: var(--surface); color: var(--text);
  border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; font-size: 13px;
  font-weight: 400; line-height: 1.45; box-shadow: 0 4px 14px rgba(0,0,0,.18); text-align: left; }
a.term:hover .tip, a.term:focus .tip { display: block; }
p.intro { color: var(--text-2); margin: 0 0 8px; }
p.answer { background: var(--surface-2); border-radius: 8px; padding: 8px 12px;
  margin: 8px 0 12px; }
.reading { border-left: 3px solid var(--link); padding: 2px 0 2px 12px; margin: 10px 0 14px; }
ul.checked { margin: 6px 0 4px; padding-left: 1.2em; }
ul.checked li { margin: 3px 0; }
svg.heatmap { display: block; max-width: 100%; height: auto; }
svg.heatmap text { font: 12px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
svg.heatmap .hm-col, svg.heatmap .hm-row { fill: var(--text); }
svg.heatmap .hm-cell { font-size: 11px; font-variant-numeric: tabular-nums; }
.hm-key { display: inline-block; width: 12px; height: 12px; border-radius: 3px;
  vertical-align: -1px; }
figure img, svg.heatmap { cursor: zoom-in; }
.lightbox { position: fixed; inset: 0; z-index: 60; background: rgba(0,0,0,.8);
  display: none; align-items: center; justify-content: center; padding: 3vh 3vw;
  cursor: zoom-out; }
.lightbox.open { display: flex; }
.lightbox .frame { background: #ffffff; border-radius: 10px; padding: 10px; max-width: 94vw;
  max-height: 94vh; overflow: auto; }
.lightbox .frame.vector { background: var(--surface); }
.lightbox img { display: block; max-width: calc(94vw - 20px); max-height: calc(94vh - 60px);
  width: auto; height: auto; }
.lightbox svg.heatmap { max-width: none; height: auto; cursor: zoom-out; }
.lightbox .cap { color: #52514e; font-size: 13px; margin: 6px 2px 0; }
.lightbox .frame.vector .cap { color: var(--text-3); }
.lightbox .close { position: fixed; top: 12px; right: 18px; font-size: 28px; line-height: 1;
  border: 0; background: none; color: #ffffff; cursor: pointer; }
.viz-body { display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start; }
.viz-body .selection { flex: 1 1 380px; min-width: 0; font-size: 13px; }
.selection th, .selection td { padding-right: 8px; }
.selection table { min-width: 0; }
.selection .bar { display: inline-block; height: 9px; border-radius: 2px; min-width: 1px;
  max-width: 44px; vertical-align: 0; }
.selection td:nth-child(3) { min-width: 84px; }
.viz.full { position: fixed; inset: 0; z-index: 50; margin: 0; padding: 16px 24px;
  background: var(--bg); overflow: auto; }
body.locked { overflow: hidden; }
.glossary-tab { position: fixed; right: 0; top: 42%; z-index: 30; writing-mode: vertical-rl;
  transform: rotate(180deg); border: 1px solid var(--border); border-right: 0;
  border-radius: 0 8px 8px 0; background: var(--surface); color: var(--link); padding: 12px 7px;
  font: inherit; font-size: 13px; font-weight: 600; cursor: pointer;
  box-shadow: 0 2px 8px rgba(0,0,0,.12); }
.drawer { position: fixed; top: 0; right: 0; bottom: 0; width: min(460px, 92vw); z-index: 45;
  background: var(--surface); border-left: 1px solid var(--border);
  box-shadow: -8px 0 28px rgba(0,0,0,.2); transform: translateX(105%);
  transition: transform .2s ease; display: flex; flex-direction: column; visibility: hidden; }
.drawer.open { transform: none; visibility: visible; }
.drawer header { padding: 14px 18px 10px; border-bottom: 1px solid var(--grid); }
.drawer header h2 { font-size: 18px; margin: 0 0 8px; }
.drawer input[type="search"] { width: 100%; padding: 7px 10px; border-radius: 8px;
  border: 1px solid var(--border); background: var(--bg); color: var(--text); font: inherit;
  font-size: 14px; }
.drawer .close { float: right; font-size: 24px; line-height: 1; border: 0; background: none;
  color: var(--text-2); cursor: pointer; }
.drawer-body { overflow: auto; padding: 0 18px 28px; font-size: 14px; }
.drawer-body h3 { font-size: 13px; text-transform: uppercase; letter-spacing: .05em;
  color: var(--text-3); margin: 18px 0 4px; }
dl.glossary dt.hit { background: var(--caution-bg); border-radius: 4px; }
.card > p, .reading p, ul.checked li, .card > details > p { max-width: 130ch; }
.feature-grid { display: grid; gap: 18px; margin-top: 10px;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 820px), 1fr)); }
.feature { background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 12px 16px; min-width: 0; }
.feature-head { display: flex; justify-content: space-between; align-items: center; gap: 10px; }
.feature-head h4 { margin: 0; font-size: 16px; }
.feature-head .tools { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.pl-plot { width: 100%; min-height: 460px; }
.pair-row { display: flex; flex-wrap: wrap; gap: 8px 22px; align-items: flex-start;
  margin-top: 4px; }
.pair-row .panel { min-width: 0; }
.panel-title { margin: 6px 0 0; color: var(--text-2); font-size: 14px; }
.panel-title b { font-size: 18px; color: var(--text); margin-right: 6px; }
.figure-legend { color: var(--text-2); font-size: 13.5px; margin-top: 8px; max-width: 130ch; }
.figure-legend p { margin: 4px 0; }
.feature .tools { margin: 4px 0; }
.feature.full { position: fixed; inset: 0; z-index: 50; margin: 0; padding: 16px 28px;
  background: var(--bg); overflow: auto; border-radius: 0; border: 0; }
.pl-embed { flex: 3 1 620px; min-width: 0; min-height: 600px; }
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
  nav.toc, .theme-toggle, .controls, button.dl, #files, .glossary-tab, .drawer header input,
  .drawer .close, .lightbox { display: none; }
  .drawer { position: static; transform: none; visibility: visible; box-shadow: none;
    border: 0; width: auto; display: block; }
  .layout { display: block; }
  details { display: block; }
  section.card { break-inside: avoid-page; }
}
"""

__all__ = ["CSS"]
