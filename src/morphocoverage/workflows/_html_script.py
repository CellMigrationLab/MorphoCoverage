"""The HTML report's script (private; ``html_report``), plain JavaScript with no library:
sortable tables, the light and dark switch, downloads of the files the page carries, figures
enlarged on a click, the glossary drawer, and the interactive PCA and UMAP (hover a point,
hide a condition, draw around points to see which conditions they come from)."""

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
  // Figures enlarged on a click (or Enter); a click anywhere or Escape closes.
  var lightbox = document.createElement('div');
  lightbox.className = 'lightbox';
  lightbox.setAttribute('role', 'dialog');
  lightbox.setAttribute('aria-modal', 'true');
  lightbox.setAttribute('aria-label', 'Enlarged figure');
  lightbox.innerHTML = '<button type="button" class="close" aria-label="Close">×</button>' +
    '<div class="frame"></div>';
  document.body.appendChild(lightbox);
  var opener = null;
  function enlarge(node) {
    var frame = lightbox.querySelector('.frame');
    var copy = node.cloneNode(true);
    ['width', 'loading', 'tabindex'].forEach(function (name) { copy.removeAttribute(name); });
    frame.className = 'frame' + (node.tagName.toLowerCase() === 'svg' ? ' vector' : '');
    frame.innerHTML = '';
    frame.appendChild(copy);
    var view = node.viewBox && node.viewBox.baseVal;
    if (view && view.width) {
      // As large as the window holds, the whole figure in view, and at most 3 times its size.
      var wide = Math.min(innerWidth * 0.9 - 20, (innerHeight * 0.88 - 40) * view.width /
        view.height, view.width * 3);
      copy.style.width = Math.max(wide, 200) + 'px';
    }
    var text = node.getAttribute('alt') || node.getAttribute('aria-label') || '';
    if (text) {
      var caption = document.createElement('p');
      caption.className = 'cap'; caption.textContent = text;
      frame.appendChild(caption);
    }
    opener = node;
    lightbox.classList.add('open');
    lightbox.querySelector('.close').focus();
  }
  function shut() {
    if (!lightbox.classList.contains('open')) return;
    lightbox.classList.remove('open');
    if (opener) opener.focus();
  }
  lightbox.addEventListener('click', shut);
  var zoomable = 'figure img, svg.heatmap';
  document.querySelectorAll(zoomable).forEach(function (node) {
    node.setAttribute('tabindex', '0');
    node.setAttribute('title', 'Click to enlarge');
  });
  document.addEventListener('click', function (event) {
    var node = event.target.closest(zoomable);
    if (node && !lightbox.contains(node)) enlarge(node);
  });
  document.addEventListener('keydown', function (event) {
    if (event.key === 'Escape') { shut(); closeGlossary(); }
    var node = event.target.closest && event.target.closest(zoomable);
    if (node && event.key === 'Enter' && !lightbox.contains(node)) enlarge(node);
  });
  // The glossary drawer: the side tab, the sidebar link and every underlined term open it at
  // the entry; its search keeps the entries whose words match.
  var drawer = document.getElementById('glossary');
  function openGlossary(id) {
    if (!drawer) return;
    drawer.classList.add('open');
    drawer.setAttribute('aria-hidden', 'false');
    drawer.querySelectorAll('.hit').forEach(function (n) { n.classList.remove('hit'); });
    var entry = id && document.getElementById(id);
    if (entry) {
      entry.classList.add('hit');
      entry.scrollIntoView({ block: 'start' });
    } else {
      var search = drawer.querySelector('input');
      if (search) search.focus();
    }
  }
  function closeGlossary() {
    if (!drawer) return;
    drawer.classList.remove('open');
    drawer.setAttribute('aria-hidden', 'true');
  }
  document.addEventListener('click', function (event) {
    if (event.target.closest('.drawer .close')) { closeGlossary(); return; }
    var link = event.target.closest('a[href^="#g-"], a[href="#glossary"], button.glossary-tab');
    if (!link || !drawer) return;
    event.preventDefault();
    var target = (link.getAttribute('href') || '').slice(1);
    openGlossary(target === 'glossary' ? '' : target);
  });
  var search = drawer && drawer.querySelector('input');
  if (search) search.addEventListener('input', function () {
    var words = search.value.trim().toLowerCase();
    drawer.querySelectorAll('dt').forEach(function (dt) {
      var dd = dt.nextElementSibling;
      var text = (dt.textContent + ' ' + dd.textContent).toLowerCase();
      var shown = !words || text.indexOf(words) >= 0;
      dt.hidden = !shown; dd.hidden = !shown;
    });
    drawer.querySelectorAll('h3').forEach(function (h) {
      var list = h.nextElementSibling;
      h.hidden = list && !list.querySelector('dt:not([hidden])');
    });
  });
  // The interactive embeddings.
  var source = document.getElementById('embedding-data');
  if (!source) return;
  var data = JSON.parse(source.textContent);
  document.querySelectorAll('.viz[data-viz]').forEach(function (viz) { scatter(viz, data); });

  function scatter(viz, data) {
    // lasso: the outline drawn, in the plot's coordinates; kept while conditions are hidden
    // or shown (the axes do not move), cleared on a switch of method or values.
    var state = { method: data.methods[0], values: data.values[0], hidden: {}, lasso: null };
    var box = viz.querySelector('.plot-box'), tip = box.querySelector('.tooltip');
    var legend = viz.querySelector('.legend'), panel = viz.querySelector('.selection');
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
        state.lasso = null;
        viz.querySelectorAll('.seg button[data-key="' + b.dataset.key + '"]').forEach(
          function (o) { o.setAttribute('aria-pressed', o === b ? 'true' : 'false'); });
        draw();
      });
    });
    var full = viz.querySelector('button.full');
    if (full) full.addEventListener('click', function () {
      var on = !viz.classList.contains('full');
      viz.classList.toggle('full', on);
      full.textContent = on ? 'Close the large view' : 'Enlarge';
      document.body.classList.toggle('locked', on);
    });
    document.addEventListener('keydown', function (event) {
      if (event.key === 'Escape' && viz.classList.contains('full')) full.click();
    });
    var W = 760, H = 560, M = 44, placed = [];
    function inside(x, y, polygon) {
      var hit = false;
      for (var i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
        var a = polygon[i], b = polygon[j];
        if ((a[1] > y) !== (b[1] > y) && x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]) {
          hit = !hit;
        }
      }
      return hit;
    }
    function outline(polygon) {
      return '<path class="lasso" d="M' + polygon.map(function (p) {
        return p[0].toFixed(1) + ',' + p[1].toFixed(1);
      }).join('L') + 'Z"/>';
    }
    function describe(set, chosen) {
      if (!panel) return;
      if (!chosen) {
        panel.innerHTML = '<p class="note">Draw around points with the mouse (or a finger) to ' +
          'see which conditions they come from.</p>';
        return;
      }
      var total = {}, picked = {}, fields = {}, shown = 0;
      placed.forEach(function (p) {
        var c = set.cond[p[2]];
        total[c] = (total[c] || 0) + 1; shown++;
      });
      chosen.forEach(function (i) {
        var c = set.cond[i];
        picked[c] = (picked[c] || 0) + 1;
        fields[c + '|' + set.labels[i]] = true;
      });
      var n = chosen.length;
      var rows = data.conditions.map(function (name, c) {
        if (!total[c]) return '';
        var mine = picked[c] || 0, share = n ? mine / n : 0, base = total[c] / shown;
        var ratio = base ? share / base : 0;
        return '<tr><td><span class="swatch" style="background:' + data.colours[c] +
          '"></span> ' + escape(name) + '</td><td class="num">' + mine + '</td>' +
          '<td><span class="bar" style="width:' + (share * 100).toFixed(1) + '%;background:' +
          data.colours[c] + '"></span> ' + Math.round(share * 100) + '%</td>' +
          '<td class="num">' + Math.round(base * 100) + '%</td>' +
          '<td class="num">' + (mine ? '×' + ratio.toFixed(1) : '–') + '</td></tr>';
      }).join('');
      panel.innerHTML = '<p><strong>' + n + ' point' + (n === 1 ? '' : 's') +
        '</strong> drawn around, from ' + Object.keys(fields).length + ' field' +
        (Object.keys(fields).length === 1 ? '' : 's') + '. <button type="button" ' +
        'class="dl clear">Clear</button></p>' +
        '<div class="table-wrap"><table><thead><tr><th>condition</th><th>points</th>' +
        '<th>share</th><th>by chance</th><th>ratio</th></tr></thead><tbody>' +
        rows + '</tbody></table></div><p class="note">Share: of the points drawn around. ' +
        'By chance: the condition\\'s share of every point shown, what the selection would ' +
        'hold by chance. A ratio above 1: the condition is more common here than in the whole ' +
        'plot. A pattern to read with the tests, not a test.</p>';
      panel.querySelector('.clear').addEventListener('click', function () {
        state.lasso = null; draw();
      });
    }
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
      for (var i = 0; i < xs.length; i++) {
        if (!state.hidden[set.cond[i]]) placed.push([sx(xs[i]), sy(ys[i]), i]);
      }
      var chosen = null, picked = {};
      if (state.lasso) {
        chosen = [];
        placed.forEach(function (p) {
          if (inside(p[0], p[1], state.lasso)) { chosen.push(p[2]); picked[p[2]] = true; }
        });
      }
      // The control first, so the conditions sit on top of it.
      placed.forEach(function (p) {
        var faded = chosen && !picked[p[2]];
        svg.push('<circle cx="' + p[0].toFixed(1) + '" cy="' + p[1].toFixed(1) +
          '" r="3" fill="' + data.colours[set.cond[p[2]]] + '" fill-opacity="' +
          (faded ? '0.12' : (chosen ? '0.9' : '0.6')) + '"/>');
      });
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
      if (state.lasso) svg.push(outline(state.lasso));
      svg.push('</svg>');
      box.querySelector('.svg-slot').innerHTML = svg.join('');
      describe(set, chosen);
    }
    box.redraw = draw;
    function local(event) {
      var svg = box.querySelector('svg'), r = svg.getBoundingClientRect(), k = W / r.width;
      return [(event.clientX - r.left) * k, (event.clientY - r.top) * k, k, r];
    }
    var drawing = null;
    box.addEventListener('pointerdown', function (event) {
      if (!box.querySelector('svg') || event.button > 0) return;
      var at = local(event);
      drawing = [[at[0], at[1]]];
      box.setPointerCapture(event.pointerId);
      tip.style.display = 'none';
      event.preventDefault();
    });
    box.addEventListener('pointerup', function () {
      if (!drawing) return;
      var polygon = drawing;
      drawing = null;
      var xs = polygon.map(function (p) { return p[0]; });
      var ys = polygon.map(function (p) { return p[1]; });
      var wide = Math.max.apply(null, xs) - Math.min.apply(null, xs);
      var tall = Math.max.apply(null, ys) - Math.min.apply(null, ys);
      // A click, not an outline: the selection is cleared.
      state.lasso = polygon.length > 4 && wide + tall > 12 ? polygon : null;
      draw();
    });
    box.addEventListener('pointermove', function (event) {
      if (!box.querySelector('svg')) return;
      var at = local(event), mx = at[0], my = at[1], k = at[2], r = at[3];
      if (drawing) {
        drawing.push([mx, my]);
        var old = box.querySelector('path.lasso');
        if (old) old.remove();
        box.querySelector('svg').insertAdjacentHTML('beforeend', outline(drawing));
        return;
      }
      var best = null, d = 144;
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

__all__ = ["SCRIPT"]
