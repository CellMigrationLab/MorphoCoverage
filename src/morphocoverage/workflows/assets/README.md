# Assets embedded in the HTML report

- `plotly-cartesian.min.js`: plotly.js 4.1.1, the cartesian partial bundle
  (`plotly.js-cartesian-dist-min` on npm: scatter, box, heatmap and the other
  cartesian traces), MIT licence (`PLOTLY_LICENSE`). `html_report` writes it
  into every page once, so the interactive figures work offline. To update:
  `npm pack plotly.js-cartesian-dist-min`, copy `plotly-cartesian.min.js` and
  `LICENSE` here, and change the version above.
