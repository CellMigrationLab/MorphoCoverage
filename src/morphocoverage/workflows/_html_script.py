"""The HTML report's script (private; ``html_report``), plain JavaScript with no library:
sortable tables, the light and dark switch, downloads of the files the page carries, figures
enlarged on a click and the glossary drawer. The Plotly figures are ``_html_plotly_script``'s."""

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
  // Tabs: one feature at a time; the arrow keys move between them.
  document.querySelectorAll('.tabs').forEach(function (tabs) {
    var buttons = Array.from(tabs.querySelectorAll('[role="tab"]'));
    var panels = Array.from(tabs.parentNode.querySelectorAll('.tab-panel'));
    function show(index) {
      buttons.forEach(function (b, i) {
        b.setAttribute('aria-selected', String(i === index));
        b.tabIndex = i === index ? 0 : -1;
      });
      panels.forEach(function (p, i) { p.hidden = i !== index; });
    }
    buttons.forEach(function (button, index) {
      button.addEventListener('click', function () { show(index); });
      button.addEventListener('keydown', function (event) {
        var step = event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0;
        if (!step) return;
        var next = (index + step + buttons.length) % buttons.length;
        show(next);
        buttons[next].focus();
      });
    });
  });
  // Many features: the search box filters the list, the one picked is shown.
  document.querySelectorAll('.picker').forEach(function (picker) {
    var find = picker.querySelector('input[type="search"]');
    var pick = picker.querySelector('select');
    var panels = Array.from(picker.querySelectorAll('.tab-panel'));
    function show() {
      panels.forEach(function (p, i) { p.hidden = String(i) !== pick.value; });
    }
    pick.addEventListener('change', show);
    find.addEventListener('input', function () {
      var word = find.value.trim().toLowerCase(), first = null;
      Array.from(pick.options).forEach(function (option) {
        option.hidden = word !== '' && option.textContent.toLowerCase().indexOf(word) < 0;
        if (!option.hidden && first === null) first = option;
      });
      if (first && pick.selectedOptions[0] && pick.selectedOptions[0].hidden) {
        pick.value = first.value; show();
      }
    });
  });
  // The plan's choices: the panel of the option picked.
  document.querySelectorAll('.choose').forEach(function (choose) {
    var pick = choose.querySelector('select');
    var panels = Array.from(choose.querySelectorAll('.choice-panel'));
    pick.addEventListener('change', function () {
      panels.forEach(function (p, i) { p.hidden = String(i) !== pick.value; });
    });
  });
  // Long tables: the first rows, and a button to show them all (sorting sorts them all).
  document.querySelectorAll('.long-table > button.show-all').forEach(function (button) {
    var shut = button.textContent;
    button.addEventListener('click', function () {
      var open = button.parentNode.classList.toggle('open');
      button.textContent = open ? 'Show the first ' + button.dataset.rows : shut;
      button.setAttribute('aria-expanded', String(open));
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
    if (window.mcRedraw) window.mcRedraw();
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
})();
"""

__all__ = ["SCRIPT"]
