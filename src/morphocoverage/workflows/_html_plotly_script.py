"""The HTML report's Plotly figures (private; ``html_report``): the script that draws, from
the page's data, every feature figure (its observations, each repeat's summary and interval,
and the matrix of every pair of conditions under it) and the PCA and UMAP (with a lasso
that says which conditions the points drawn around come from). It follows the page's light
or dark theme; without plotly.js (``_html_plotly.plotly_js``) the static figures stay."""

PLOTLY_SCRIPT = r"""
(function () {
  if (typeof Plotly === 'undefined') return;
  var FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
  // The matrices' colours: d diverging from the middle, p and q green when they pass.
  var DIVERGING = [[0, 'rgb(42,120,214)'], [0.5, 'rgb(240,239,236)'], [1, 'rgb(227,73,72)']];
  var GREENS = [[0, 'rgb(240,239,236)'], [1, 'rgb(46,125,50)']];
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
  // Plotly reads its strings as HTML, so & < > are escaped; a quote is not (Plotly would
  // show &quot; as written), and no string here goes into an attribute.
  function escape(text) {
    return String(text).replace(/[&<>]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c];
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
    var plot = block.querySelector('.pl-plot');
    var pairs = Array.prototype.slice.call(block.querySelectorAll('.pl-pair'));
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
          x.push(points.x[i]); y.push(points.y[i]); text.push(escape(points.label[i]));
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
      // The slanted condition names, cut at 30 characters, take this much room below.
      var names = Math.max.apply(null, spec.conditions.map(function (c) {
        return Math.min(c.name.length, 30);
      }));
      var layout = merge(base(t), {
        // Taller by the room the long names take, so the plot keeps its height.
        height: height + (many ? Math.max(0, 7 * names - 80) : 0),
        margin: { l: 70, r: 16, t: slant ? 30 + 6 * longest : 34,
          b: many ? Math.max(110, 30 + 7 * names) : 70 },
        hovermode: 'closest', showlegend: false, dragmode: 'zoom',
        xaxis: { tickvals: spec.conditions.map(function (c) { return c.centre; }),
          ticktext: spec.conditions.map(function (c) { return escape(cutName(c.name, 30)); }),
          tickangle: many ? -30 : 0, tickfont: { size: 15 }, showgrid: false, zeroline: false,
          automargin: true,
          range: [Math.min.apply(null, xs) - 0.7, Math.max.apply(null, xs) + 0.7] },
        yaxis: { title: { text: escape(spec.ylabel), font: { size: 15 } }, gridcolor: t.grid,
          zeroline: false },
        annotations: spec.clusters.filter(function (k) {
          return k.repeat && k.repeat !== 'single';  // CellPatcher's word for no such level
        }).map(function (k) {
          // Slanted when the slots are narrow, so neighbouring repeats' names never overlap.
          return { x: k.x, y: 1, yref: 'paper', yanchor: 'bottom', xanchor: slant ? 'left' :
            'center', showarrow: false, textangle: slant ? -40 : 0, text: escape(k.repeat),
            font: { size: 12, color: t.muted } };
        })
      });
      Plotly.react(plot, traces, layout, config);
      if (spec.pairwise) {
        pairs.forEach(function (node, k) {
          matrix(node, spec.pairwise, spec.pairwise[node.dataset.panel], t, k === 0 ||
            node.parentNode.offsetTop !== pairs[0].parentNode.offsetTop);
        });
      }
    }
    draw();
    redraws.push(draw);
    enlarger(block, [plot], [height]);
  });

  // A matrix of every pair (panel B or C): its lower-left and upper-right triangles drawn as
  // two heatmaps with their own colours, the text and the hover of every cell.
  // A matrix writes a name of up to 24 characters in full; a longer one ends in an ellipsis,
  // the full name in the hover (as in the static figure). Functions, not variables: the
  // figures above are drawn before a variable here would be set.
  // Cut in the middle, as _wording.shortened: names of a series often differ at their end.
  function cutName(s, most) {
    if (s.length <= most) return s;
    var head = Math.floor((most - 1) * 3 / 5), tail = most - 1 - head;
    return s.slice(0, head).replace(/\s+$/, '') + '\u2026' +
      s.slice(s.length - tail).replace(/^\s+/, '');
  }
  function shortName(s) { return cutName(s, 24); }
  function matrix(node, pw, panel, t, labels) {
    var n = pw.names.length, longest = Math.max.apply(null, pw.names.map(function (s) {
      return shortName(s).length;
    }));
    function half(below) {
      return panel.text.map(function (row, i) {
        return row.map(function (v, j) { return (below ? i > j : i < j) ? v : ''; });
      });
    }
    // Plotly reads names and hovers as HTML: the conditions' names are escaped. The cells sit
    // at 0..n-1 so two names cut alike stay two rows.
    var names = pw.names.map(function (s) { return escape(shortName(s)); });
    var at = pw.names.map(function (_, i) { return i; });
    var hovers = panel.hover.map(function (row) { return row.map(escape); });
    var common = { type: 'heatmap', x: at, y: at, showscale: false, xgap: 4,
      ygap: 4, hoverongaps: false, customdata: hovers, texttemplate: '%{text}',
      textfont: { size: n > 8 ? 12 : 14 }, hovertemplate: '%{customdata}<extra></extra>' };
    function scale(stops) { return stops.map(function (stop) { return stop.slice(); }); }
    var lowerScale = panel.lower_scale === 'd' ?
      { zmin: -pw.cap, zmax: pw.cap, colorscale: scale(DIVERGING) } :
      { zmin: 0, zmax: 1, colorscale: scale(GREENS) };
    var traces = [
      merge(merge(merge({}, common), lowerScale), { z: panel.lower, text: half(true) }),
      merge(merge({}, common), { z: panel.upper, text: half(false), zmin: 0, zmax: 1,
        colorscale: scale(GREENS) })
    ];
    // Each cell reads its row's condition against its column's: the axes say so.
    var cell = n > 8 ? 58 : 80, left = labels ? Math.min(48 + 8.5 * longest, 250) : 20;
    var side = cell * n, below = 54 + 8.5 * longest * 0.7;
    var title = { font: { size: 13 }, standoff: 10 };
    var layout = merge(base(t), {
      width: left + side + 20, height: side + 20 + below,
      margin: { l: left, r: 20, t: 20, b: below },
      xaxis: { tickangle: -35, showgrid: false, zeroline: false, tickfont: { size: 14 },
        constrain: 'domain', ticks: '', showline: false, tickvals: at, ticktext: names,
        title: merge({ text: 'against' }, title) },
      yaxis: { autorange: 'reversed', showgrid: false, zeroline: false, tickfont: { size: 14 },
        scaleanchor: 'x', constrain: 'domain', ticks: '', showline: false, tickvals: at,
        ticktext: names,
        showticklabels: labels, title: merge({ text: labels ? 'condition' : '' }, title) }
    });
    Plotly.react(node, traces, layout, merge(merge({}, config), { responsive: false }));
  }

  // The PCA and UMAP: coloured by condition, population or a feature; the populations and
  // the labelled cells ringed; the lasso says which conditions, populations and features the
  // points drawn around come from.
  var source = document.getElementById('embedding-data');
  var data = source ? JSON.parse(source.textContent) : null;
  var RINGS = ['#111111', '#7a1fa2', '#b8860b', '#00695c', '#ad1457', '#37474f'];
  var GROUPS = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d'];
  function ringColour(k) { return RINGS[k % RINGS.length]; }
  function quantile(sorted, q) {
    if (!sorted.length) return 0;
    var i = Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))));
    return sorted[i];
  }
  function popName(k) {
    var p = data.populations[k];
    return p.name + (p.kind === 'group' ? ', a group to validate' : '') + ' (' +
      Math.round(p.share * 1000) / 10 + '%' + (p.defined ? ': ' +
      p.defined.split(' (d')[0] : '') + ')';
  }
  document.querySelectorAll('.viz[data-viz]').forEach(function (viz) {
    if (!data) return;
    var state = { method: data.methods[0], values: data.values[0], colour: 'condition',
      pick: null };
    var node = viz.querySelector('.pl-embed'), panel = viz.querySelector('.selection');
    var menu = viz.querySelector('select.colour-by'), find = viz.querySelector('input.find');
    // Past 12 conditions most are grey: the legend names the coloured ones, and the search
    // highlights the condition picked.
    var many = data.colours.indexOf(data.grey) >= 0;
    var height = 600;
    function current() { return data.sets[state.values][state.method]; }
    function hiddenConditions() {
      var hidden = {};
      if (state.colour !== 'condition') return hidden;
      (node.data || []).forEach(function (trace) {
        if (trace.meta && trace.meta.condition !== undefined && trace.visible === 'legendonly')
          hidden[trace.meta.condition] = true;
      });
      return hidden;
    }
    function featureRows(set, picked) {
      if (!set.z || !data.features.length) return '';
      var mine = {}, rows = [];
      picked.forEach(function (i) { mine[i] = true; });
      data.features.forEach(function (f) {
        var z = set.z[f]; if (!z) return;
        var a = 0, b = 0, na = 0, nb = 0, va = 0, vb = 0;
        z.forEach(function (v, i) { if (mine[i]) { a += v; na++; } else { b += v; nb++; } });
        if (na < 2 || nb < 2) return;
        a /= na; b /= nb;
        z.forEach(function (v, i) {
          if (mine[i]) va += (v - a) * (v - a); else vb += (v - b) * (v - b);
        });
        var sd = Math.sqrt((va / (na - 1) + vb / (nb - 1)) / 2);
        if (sd > 0) rows.push([f, (a - b) / sd]);
      });
      rows.sort(function (x, y) { return Math.abs(y[1]) - Math.abs(x[1]); });
      rows = rows.slice(0, 6);
      if (!rows.length) return '';
      var widest = Math.max.apply(null, rows.map(function (r) { return Math.abs(r[1]); })) || 1;
      return '<p><strong>Features that set them apart</strong> (Cohen\'s d, the points drawn ' +
        'around against the other points shown):</p><div class="table-wrap"><table><thead>' +
        '<tr><th>feature</th><th>d</th><th></th></tr></thead><tbody>' + rows.map(function (r) {
          return '<tr><td>' + escape(r[0]) + '</td><td class="num">' + (r[1] > 0 ? '+' : '') +
            r[1].toFixed(2) + '</td><td><span class="bar" style="width:' +
            (Math.abs(r[1]) / widest * 100).toFixed(0) + '%;background:' +
            (r[1] > 0 ? '#c2185b' : '#1565c0') + '"></span> ' +
            (r[1] > 0 ? 'higher' : 'lower') + '</td></tr>';
        }).join('') + '</tbody></table></div>';
    }
    function describe(selected) {
      if (!selected) {
        panel.innerHTML = '<p class="note">Draw around points with the mouse (or a finger) ' +
          'to see which conditions and populations they come from and which features set ' +
          'them apart; double-click the plot to clear.</p>';
        return;
      }
      var set = current(), hidden = hiddenConditions(), seen = {}, picked = [];
      selected.forEach(function (p) {
        var i = p.customdata;
        if (i === undefined || seen[i]) return;
        seen[i] = true; picked.push(i);
      });
      var total = {}, shown = 0, mine = {}, fields = {};
      set.cond.forEach(function (c) {
        if (!hidden[c]) { total[c] = (total[c] || 0) + 1; shown++; }
      });
      picked.forEach(function (i) {
        mine[set.cond[i]] = (mine[set.cond[i]] || 0) + 1;
        fields[set.cond[i] + '|' + set.labels[i]] = true;
      });
      var n = picked.length, nf = Object.keys(fields).length;
      var rows = data.conditions.map(function (name, c) {
        if (!total[c]) return '';
        var k = mine[c] || 0, share = n ? k / n : 0, chance = total[c] / shown;
        if (many && !k) return '';
        return '<tr><td><span class="swatch" style="background:' + data.colours[c] +
          '"></span> ' + escape(name) + '</td><td class="num">' + k + '</td><td>' +
          '<span class="bar" style="width:' + (share * 100).toFixed(1) + '%;background:' +
          data.colours[c] + '"></span> ' + Math.round(share * 100) + '%</td><td class="num">' +
          Math.round(chance * 100) + '%</td><td class="num">' +
          (k ? '×' + (share / chance).toFixed(1) : '–') + '</td></tr>';
      }).join('');
      var pops = '';
      if (set.pop && data.populations.length) {
        var count = {};
        picked.forEach(function (i) { count[set.pop[i]] = (count[set.pop[i]] || 0) + 1; });
        pops = '<p>Populations: ' + data.populations.map(function (p, k) {
          return count[k] ? escape(p.name) + ' ' + Math.round(count[k] / n * 100) + '%' : '';
        }).filter(Boolean).concat(count[-1] ? ['between them ' +
          Math.round(count[-1] / n * 100) + '%'] : []).join(' · ') + '.</p>';
      }
      panel.innerHTML = '<p><strong>' + n + ' point' + (n === 1 ? '' : 's') +
        '</strong> drawn around, from ' + nf + ' field' + (nf === 1 ? '' : 's') + '.</p>' +
        '<div class="table-wrap"><table><thead><tr><th>condition</th><th>points</th>' +
        '<th>share</th><th>by chance</th><th>ratio</th></tr></thead><tbody>' + rows +
        '</tbody></table></div>' + pops + featureRows(set, picked) + '<p class="note">' +
        'Share: of the points drawn around. By chance: the condition\'s share of every point ' +
        'shown. A ratio above 1: the condition is more common here than in the whole plot. ' +
        'A pattern to read with the tests, not a test.</p>';
    }
    function points(index) {
      var set = current();
      return { x: index.map(function (i) { return set.x[i]; }),
        y: index.map(function (i) { return set.y[i]; }),
        text: index.map(function (i) { return escape(set.labels[i]); }), customdata: index };
    }
    function scatter(name, index, marker, extra) {
      return Object.assign({ type: 'scatter', mode: 'markers', name: name, marker: marker,
        selected: { marker: { opacity: 1 } }, unselected: { marker: { opacity: 0.12 } },
        hovertemplate: '<b>' + name + '</b><br>%{text}<extra></extra>' }, points(index),
        extra || {});
    }
    function rings(set) {
      var traces = [];
      // In population colours the populations are the colours: no rings over them.
      if (set.pop && state.colour !== 'population') {
        data.populations.forEach(function (p, k) {
          var index = [];
          set.pop.forEach(function (v, i) { if (v === k) index.push(i); });
          if (!index.length) return;
          traces.push(scatter(escape(popName(k)), index, { symbol: 'circle-open',
            size: 10, color: ringColour(k), line: { width: 1.6 } },
            { visible: p.share < 0.5 ? true : 'legendonly' }));
        });
      }
      if (set.lab && data.label) {
        var marked = [];
        set.lab.forEach(function (v, i) { if (v) marked.push(i); });
        if (marked.length) traces.push(scatter('labelled ' + escape(data.label), marked,
          { symbol: 'x-thin-open', size: 8, color: '#000000', line: { width: 1.2 } }));
      }
      return traces;
    }
    function base_traces(set) {
      if (state.colour === 'population' && set.pop) {
        var groups = data.populations.map(function (p, k) {
          var index = [];
          set.pop.forEach(function (v, i) { if (v === k) index.push(i); });
          return scatter(escape(popName(k)), index,
            { color: GROUPS[k % GROUPS.length], size: 7, opacity: 0.75 });
        });
        var between = [];
        set.pop.forEach(function (v, i) { if (v < 0) between.push(i); });
        return [scatter('between populations (' + between.length + ')', between,
          { color: '#9e9e9e', size: 6, opacity: 0.35 })].concat(groups);
      }
      if (state.colour !== 'condition' && set.z && set.z[state.colour]) {
        var all = set.x.map(function (_, i) { return i; }), z = set.z[state.colour];
        var sorted = z.slice().sort(function (a, b) { return a - b; });
        return [scatter(escape(state.colour), all, { color: z, colorscale: 'Viridis',
          cmin: quantile(sorted, 0.02), cmax: quantile(sorted, 0.98), size: 7, opacity: 0.8,
          colorbar: { title: { text: escape(state.colour) + ' (SD)' }, thickness: 14 } },
          { showlegend: false, hovertemplate: '%{text}<br>' + escape(state.colour) +
            ' %{marker.color:.2f} SD<extra></extra>' })];
      }
      var greyed = data.colours.filter(function (c, k) {
        return c === data.grey && k !== state.pick;
      }).length;
      var first = true, picked = null, traces = [];
      data.conditions.forEach(function (name, c) {
        var index = [], grey = data.colours[c] === data.grey;
        set.cond.forEach(function (k, i) { if (k === c) index.push(i); });
        var marker = { color: data.colours[c], size: grey ? 6 : 7,
          opacity: state.pick === null ? (grey ? 0.45 : 0.7) : 0.2 };
        var extra = { meta: { condition: c } };
        // A long name is cut in the legend, whole in the hover.
        var label = escape(cutName(name, 40)) + ' (' + index.length + ')';
        extra.hovertemplate = '<b>' + escape(name) + '</b><br>%{text}<extra></extra>';
        if (c === state.pick) {
          // The condition picked: on top, in its colour (ink when grey), larger.
          picked = scatter(label, index,
            { color: grey ? theme().text : data.colours[c], size: 9, opacity: 1 }, extra);
          return;
        }
        if (grey) {
          extra.legendgroup = 'grey';
          extra.showlegend = first;
          first = false;
          traces.unshift(scatter('the other ' + greyed + ' conditions, grey', index, marker,
            extra));
          return;
        }
        traces.push(scatter(label, index, marker, extra));
      });
      return picked ? traces.concat([picked]) : traces;
    }
    function draw() {
      var t = theme(), set = current();
      var traces = base_traces(set).concat(rings(set));
      var annotations = [];
      (set.arrows || []).forEach(function (a) {
        annotations.push({ x: a[1], y: a[2], ax: 0, ay: 0, axref: 'x', ayref: 'y', text: '',
          showarrow: true, arrowhead: 2, arrowwidth: 1.8, arrowcolor: t.text });
        annotations.push({ x: a[1], y: a[2], text: '<b>' + escape(cutName(a[0], 30)) + '</b>',
          hovertext: escape(a[0]),
          showarrow: false, xanchor: a[1] >= 0 ? 'left' : 'right', yanchor: 'bottom',
          font: { size: 13, color: t.text }, bgcolor: t.surface, opacity: 0.9 });
      });
      var layout = merge(base(t), {
        height: viz.classList.contains('full') ? Math.max(height, innerHeight - 200) : height,
        margin: { l: 70, r: 16, t: 16, b: 60 }, dragmode: 'lasso', hovermode: 'closest',
        // Under the x axis's title, never over it.
        legend: { orientation: 'h', yanchor: 'top', y: -0.2, font: { size: 14 } },
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
    if (menu) menu.addEventListener('change', function () { state.colour = menu.value; draw(); });
    if (find) find.addEventListener('input', function () {
      var index = data.conditions.indexOf(find.value.trim());
      var pick = index >= 0 ? index : null;
      if (pick === state.pick) return;
      state.pick = pick;
      if (pick !== null && menu && state.colour !== 'condition') {
        state.colour = 'condition'; menu.value = 'condition';
      }
      draw();
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
