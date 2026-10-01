"""The HTML report's Plotly figures (private; ``html_report``): the script that draws, from
the page's data, every feature figure (its observations, each repeat's summary and interval,
and the matrix of every pair of conditions under it) and the PCA and UMAP (with a lasso
that says which conditions the points drawn around come from). It follows the page's light
or dark theme; without plotly.js (``_html_plotly.plotly_js``) the static figures stay."""

PLOTLY_SCRIPT = r"""
(function () {
  if (typeof Plotly === 'undefined') return;
  var FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
  // The technical repeats' shapes, by rank, as matplotlib's markers in the static figures.
  var SHAPES = ['circle', 'square', 'triangle-up', 'diamond', 'triangle-down', 'cross', 'x',
    'star'];
  function css(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }
  function theme() {
    return { text: css('--text'), muted: css('--text-3'), grid: css('--grid'),
      surface: css('--surface') };
  }
  function base(t) {
    return { font: { family: FONT, size: 14, color: t.text }, paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)', hoverlabel: { font: { family: FONT, size: 13 } } };
  }
  var config = { responsive: true, displaylogo: false,
    modeBarButtonsToRemove: ['autoScale2d', 'select2d'],
    toImageButtonOptions: { format: 'png', scale: 2 } };
  var redraws = [];
  window.mcRedraw = function () { redraws.forEach(function (f) { f(); }); };
  function escape(text) {
    return String(text).replace(/[&<>"]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
    });
  }
  function merge(a, b) { for (var k in b) a[k] = b[k]; return a; }
  // Enlarge: a figure fills the window, its plots taller; Escape or the button closes it.
  function enlarger(block, nodes, heights) {
    var button = block.querySelector('button.full');
    if (!button) return;
    function size(on) {
      nodes.forEach(function (node, i) {
        if (!node || !node.data) return;
        var tall = on ? Math.max(heights[i], (innerHeight - 200) / (i === 0 ? 1 : 1.6)) :
          heights[i];
        Plotly.relayout(node, { height: tall });
      });
    }
    button.addEventListener('click', function () {
      var on = !block.classList.contains('full');
      block.classList.toggle('full', on);
      document.body.classList.toggle('locked', on);
      button.textContent = on ? 'Close the large view' : 'Enlarge';
      size(on);
    });
    document.addEventListener('keydown', function (event) {
      if (event.key === 'Escape' && block.classList.contains('full')) button.click();
    });
  }

  // Feature figures: the observations, each repeat's summary and interval, every pair.
  var store = document.getElementById('feature-data');
  var specs = store ? JSON.parse(store.textContent) : [];
  specs.forEach(function (entry) {
    var block = document.querySelector('.feature[data-spec="' + entry.id + '"]');
    if (!block) return;
    var plot = block.querySelector('.pl-plot'), pair = block.querySelector('.pl-pair');
    // The values shown (raw, or normalised to each repeat's control): the switch redraws the
    // points, the summaries, the intervals and the matrix's d from that view's numbers.
    var state = entry.current, height = 460;
    block.querySelectorAll('.seg button').forEach(function (button) {
      button.addEventListener('click', function () {
        state = button.dataset.value;
        block.querySelectorAll('.seg button').forEach(function (other) {
          other.setAttribute('aria-pressed', other === button ? 'true' : 'false');
        });
        draw();
      });
    });
    function draw() {
      var spec = entry.views[state], points = spec.points, many = spec.conditions.length > 5;
      block.querySelector('.caption').textContent = spec.caption;
      block.querySelector('.sampled').textContent = points.shown < points.total ?
        points.shown + ' of ' + points.total + ' observations drawn (a seeded sample of ' +
        'each cluster); the summaries, intervals and tests use every one.' : '';
      var t = theme(), traces = [];
      spec.conditions.forEach(function (c, ci) {
        var x = [], y = [], text = [], symbol = [];
        points.cluster.forEach(function (k, i) {
          if (spec.clusters[k].condition !== ci) return;
          x.push(points.x[i]); y.push(points.y[i]); text.push(points.label[i]);
          symbol.push(SHAPES[points.shape[i]] || 'circle-dot');
        });
        traces.push({ type: 'scatter', mode: 'markers', x: x, y: y, text: text,
          name: c.name, legendgroup: c.name,
          marker: { color: c.colour, size: 7, opacity: 0.55, symbol: symbol },
          hovertemplate: '<b>' + escape(c.name) + '</b> · %{text}<br>' + escape(spec.feature) +
            ' %{y}<extra></extra>' });
      });
      spec.conditions.forEach(function (c, ci) {
        var mine = spec.clusters.filter(function (k) {
          return k.condition === ci && k.summary !== null;
        });
        traces.push({ type: 'scatter', mode: 'markers', name: c.name, legendgroup: c.name,
          showlegend: false,
          x: mine.map(function (k) { return k.x; }),
          y: mine.map(function (k) { return k.summary; }),
          customdata: mine.map(function (k) {
            return [k.repeat || 'repeats pooled', k.low, k.high];
          }),
          marker: { symbol: 'line-ew', size: 30, line: { width: 4, color: t.text } },
          error_y: { type: 'data', symmetric: false, thickness: 2.5, width: 9, color: t.text,
            array: mine.map(function (k) { return k.high === null ? 0 : k.high - k.summary; }),
            arrayminus: mine.map(function (k) {
              return k.low === null ? 0 : k.summary - k.low;
            }) },
          hovertemplate: '<b>' + escape(c.name) + '</b> · %{customdata[0]}<br>median %{y}' +
            '<br>95% interval %{customdata[1]} to %{customdata[2]}<extra></extra>' });
      });
      var xs = spec.clusters.map(function (k) { return k.x; });
      var span = Math.max.apply(null, xs) - Math.min.apply(null, xs) + 1.4;
      var longest = Math.max.apply(null, spec.clusters.map(function (k) {
        return (k.repeat || '').length;
      }));
      var slant = (plot.clientWidth || 900) / span < 8 * longest + 6;
      var layout = merge(base(t), {
        height: height, margin: { l: 70, r: 16, t: slant ? 30 + 6 * longest : 34,
          b: many ? 110 : 70 },
        hovermode: 'closest', showlegend: false, dragmode: 'zoom',
        xaxis: { tickvals: spec.conditions.map(function (c) { return c.centre; }),
          ticktext: spec.conditions.map(function (c) { return escape(c.name); }),
          tickangle: many ? -30 : 0, tickfont: { size: 15 }, showgrid: false, zeroline: false,
          range: [Math.min.apply(null, xs) - 0.7, Math.max.apply(null, xs) + 0.7] },
        yaxis: { title: { text: escape(spec.ylabel), font: { size: 15 } }, gridcolor: t.grid,
          zeroline: false },
        annotations: spec.clusters.filter(function (k) { return k.repeat; }).map(function (k) {
          // Slanted when the slots are narrow, so neighbouring repeats' names never overlap.
          return { x: k.x, y: 1, yref: 'paper', yanchor: 'bottom', xanchor: slant ? 'left' :
            'center', showarrow: false, textangle: slant ? -40 : 0, text: escape(k.repeat),
            font: { size: 12, color: t.muted } };
        })
      });
      Plotly.react(plot, traces, layout, config);
      if (pair && spec.pairwise) matrix(pair, spec.pairwise, t);
    }
    draw();
    redraws.push(draw);
    enlarger(block, [plot, pair], [height, pair ? pair.offsetHeight || 420 : 0]);
  });

  function matrix(node, pw, t) {
    var n = pw.names.length, longest = Math.max.apply(null, pw.names.map(function (s) {
      return s.length;
    }));
    var lower = pw.text.map(function (row, i) {
      return row.map(function (v, j) { return i > j ? v : ''; });
    });
    var upper = pw.text.map(function (row, i) {
      return row.map(function (v, j) { return i < j ? v : ''; });
    });
    var common = { type: 'heatmap', x: pw.names, y: pw.names, showscale: false, xgap: 4,
      ygap: 4, hoverongaps: false, customdata: pw.hover, texttemplate: '%{text}',
      textfont: { size: n > 8 ? 12 : 14 }, hovertemplate: '%{customdata}<extra></extra>' };
    var traces = [
      merge(merge({}, common), { z: pw.d, text: lower, zmin: -pw.cap, zmax: pw.cap,
        colorscale: [[0, 'rgb(42,120,214)'], [0.5, 'rgb(240,239,236)'], [1, 'rgb(227,73,72)']]
      }),
      merge(merge({}, common), { z: pw.p, text: upper, zmin: 0, zmax: 1,
        colorscale: [[0, 'rgb(240,239,236)'], [1, 'rgb(46,125,50)']] })
    ];
    var cell = n > 8 ? 58 : 80, left = 24 + 8.5 * longest;
    var side = cell * n;
    var layout = merge(base(t), {
      width: left + side + 20, height: side + 20 + 8.5 * longest * 0.7 + 30,
      margin: { l: left, r: 20, t: 20, b: 30 + 8.5 * longest * 0.7 },
      xaxis: { tickangle: -35, showgrid: false, zeroline: false, tickfont: { size: 14 },
        constrain: 'domain', ticks: '', showline: false },
      yaxis: { autorange: 'reversed', showgrid: false, zeroline: false, tickfont: { size: 14 },
        scaleanchor: 'x', constrain: 'domain', ticks: '', showline: false }
    });
    Plotly.react(node, traces, layout, merge(merge({}, config), { responsive: false }));
  }

  // The PCA and UMAP: a trace per condition, the loadings as arrows, and the lasso.
  var source = document.getElementById('embedding-data');
  var data = source ? JSON.parse(source.textContent) : null;
  document.querySelectorAll('.viz[data-viz]').forEach(function (viz) {
    if (!data) return;
    var state = { method: data.methods[0], values: data.values[0] };
    var node = viz.querySelector('.pl-embed'), panel = viz.querySelector('.selection');
    var height = 600;
    function current() { return data.sets[state.values][state.method]; }
    function describe(selected) {
      if (!selected) {
        panel.innerHTML = '<p class="note">Draw around points with the mouse (or a finger) ' +
          'to see which conditions they come from; double-click the plot to clear.</p>';
        return;
      }
      var set = current(), shown = 0, total = {}, picked = {}, fields = {};
      node.data.forEach(function (trace, c) {
        if (trace.visible === 'legendonly') return;
        total[c] = trace.x.length; shown += trace.x.length;
      });
      selected.forEach(function (p) {
        picked[p.curveNumber] = (picked[p.curveNumber] || 0) + 1;
        fields[p.curveNumber + '|' + set.labels[p.customdata]] = true;
      });
      var n = selected.length, nf = Object.keys(fields).length;
      var rows = data.conditions.map(function (name, c) {
        if (!total[c]) return '';
        var mine = picked[c] || 0, share = n ? mine / n : 0, chance = total[c] / shown;
        return '<tr><td><span class="swatch" style="background:' + data.colours[c] +
          '"></span> ' + escape(name) + '</td><td class="num">' + mine + '</td><td>' +
          '<span class="bar" style="width:' + (share * 100).toFixed(1) + '%;background:' +
          data.colours[c] + '"></span> ' + Math.round(share * 100) + '%</td><td class="num">' +
          Math.round(chance * 100) + '%</td><td class="num">' +
          (mine ? '×' + (share / chance).toFixed(1) : '–') + '</td></tr>';
      }).join('');
      panel.innerHTML = '<p><strong>' + n + ' point' + (n === 1 ? '' : 's') +
        '</strong> drawn around, from ' + nf + ' field' + (nf === 1 ? '' : 's') + '.</p>' +
        '<div class="table-wrap"><table><thead><tr><th>condition</th><th>points</th>' +
        '<th>share</th><th>by chance</th><th>ratio</th></tr></thead><tbody>' + rows +
        '</tbody></table></div><p class="note">Share: of the points drawn around. By ' +
        'chance: the condition\'s share of every point shown, what the selection would hold ' +
        'by chance. A ratio above 1: the condition is more common here than in the whole ' +
        'plot. A pattern to read with the tests, not a test.</p>';
    }
    function draw() {
      var t = theme(), set = current();
      var traces = data.conditions.map(function (name, c) {
        var index = [];
        set.cond.forEach(function (k, i) { if (k === c) index.push(i); });
        return { type: 'scatter', mode: 'markers', name: escape(name) + ' (' + index.length +
          ')', x: index.map(function (i) { return set.x[i]; }),
          y: index.map(function (i) { return set.y[i]; }),
          text: index.map(function (i) { return set.labels[i]; }), customdata: index,
          marker: { color: data.colours[c], size: 7, opacity: 0.7 },
          selected: { marker: { opacity: 1 } }, unselected: { marker: { opacity: 0.12 } },
          hovertemplate: '<b>' + escape(name) + '</b><br>%{text}<extra></extra>' };
      });
      var annotations = [];
      (set.arrows || []).forEach(function (a) {
        annotations.push({ x: a[1], y: a[2], ax: 0, ay: 0, axref: 'x', ayref: 'y', text: '',
          showarrow: true, arrowhead: 2, arrowwidth: 1.8, arrowcolor: t.text });
        annotations.push({ x: a[1], y: a[2], text: '<b>' + escape(a[0]) + '</b>',
          showarrow: false, xanchor: a[1] >= 0 ? 'left' : 'right', yanchor: 'bottom',
          font: { size: 13, color: t.text }, bgcolor: t.surface, opacity: 0.9 });
      });
      var layout = merge(base(t), {
        height: viz.classList.contains('full') ? Math.max(height, innerHeight - 200) : height,
        margin: { l: 70, r: 16, t: 16, b: 60 }, dragmode: 'lasso', hovermode: 'closest',
        legend: { orientation: 'h', y: -0.12, font: { size: 14 } },
        xaxis: { title: { text: escape(set.xlabel) }, gridcolor: t.grid, zeroline: false },
        yaxis: { title: { text: escape(set.ylabel) }, gridcolor: t.grid, zeroline: false },
        annotations: annotations
      });
      viz.querySelector('.caption').textContent = set.caption;
      Plotly.react(node, traces, layout, config);
      describe(null);
    }
    viz.querySelectorAll('.seg button').forEach(function (b) {
      b.addEventListener('click', function () {
        state[b.dataset.key] = b.dataset.value;
        viz.querySelectorAll('.seg button[data-key="' + b.dataset.key + '"]').forEach(
          function (o) { o.setAttribute('aria-pressed', o === b ? 'true' : 'false'); });
        draw();
      });
    });
    draw();
    node.on('plotly_selected', function (event) {
      describe(event && event.points && event.points.length ? event.points : null);
    });
    node.on('plotly_deselect', function () { describe(null); });
    redraws.push(draw);
    enlarger(viz, [node], [height]);
  });
})();
"""

__all__ = ["PLOTLY_SCRIPT"]
