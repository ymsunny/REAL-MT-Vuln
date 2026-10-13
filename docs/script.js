(() => {
  'use strict';
  const data = window.REAL_MT;
  if (!data) return;
  const get = id => document.getElementById(id);
  const format = value => value.toFixed(1);
  const setPressed = (selector, selected) => document.querySelectorAll(selector).forEach(button => button.setAttribute('aria-pressed', String(button === selected)));

  document.querySelectorAll('[data-context]').forEach(button => {
    button.addEventListener('click', () => {
      const noisy = button.dataset.context === 'noisy';
      get('retrieval-demo').dataset.state = noisy ? 'noisy' : 'gold';
      get('demo-hint').textContent = noisy ? '“obeying”' : '“talking back”';
      get('demo-output').innerHTML = noisy ? 'Are you <strong>obeying an opening</strong> to me?' : 'Are you <strong>talking back</strong> to me?';
      get('demo-status').textContent = noisy ? 'Misleading hint adopted' : 'Meaning preserved';
      setPressed('[data-context]', button);
    });
  });

  document.querySelectorAll('[data-noise]').forEach(button => {
    button.addEventListener('click', () => {
      const condition = data.conditions.find(item => item.key === button.dataset.noise);
      get('noise-reading').dataset.noiseKind = condition.key;
      get('noise-label').textContent = condition.hint.toUpperCase();
      get('noise-text').textContent = condition.text;
      get('noise-note').textContent = condition.note;
      setPressed('[data-noise]', button);
    });
  });

  function renderResults() {
    const model = data.models[Number(get('model-select').value)];
    const direction = get('direction-select').value;
    const values = direction === 'avg' ? model.averages : model.values.map(row => row[Number(direction)]);
    document.querySelectorAll('.result-row').forEach((row, index) => {
      row.querySelector('.bar-fill').style.width = (values[index] / 3 * 100) + '%';
      row.querySelector('.bar-score').textContent = format(values[index]);
    });
    get('results-status').textContent = model.name + ', ' + (direction === 'avg' ? 'average across 10 directions' : data.fullDirections[Number(direction)]) + '. Fidelity scores: ' + data.conditions.map((c, i) => c.label + ' ' + format(values[i])).join('; ') + '.';
    get('direction-table-caption').textContent = 'Table 3 · ' + model.name + '.';
    get('direction-table').innerHTML = data.directions.map((name, i) => '<tr><th scope="row">' + name + '</th>' + model.values.map(row => '<td>' + format(row[i]) + '</td>').join('') + '</tr>').join('');
  }
  get('model-select').addEventListener('change', renderResults);
  get('direction-select').addEventListener('change', renderResults);

  const strategies = {
    ALI: 'ALI trains on two parts opposite-meaning context and one part no context, paired with correct target translations.',
    CDA: 'CDA trains on one part opposite-meaning context, one part gold context and one part no context, paired with correct target translations.',
    'CK-PLUG': 'CK-PLUG is training-free: it uses token-level confidence gain to blend context-aware and internal predictions, or suppress external knowledge.',
    Vanilla: 'Vanilla fine-tuning uses three parts no-context training data, matching the total training size of ALI and CDA.'
  };
  function renderMitigation() {
    const entry = data.mitigation[Number(get('mitigation-direction').value)];
    const baseline = entry.scores[0];
    const selected = entry.scores.find(item => item.name === get('strategy-select').value);
    get('strategy-note').textContent = strategies[selected.name];
    document.querySelectorAll('.selected-strategy').forEach(label => { label.textContent = selected.name; });
    ['gold', 'opposite'].forEach((condition, i) => {
      get(condition + '-baseline').textContent = format(baseline[condition]);
      get(condition + '-score').textContent = format(selected[condition]);
      get(condition + '-fill').style.width = (selected[condition] / 3 * 100) + '%';
      document.querySelectorAll('.tradeoff-chart')[i].querySelector('.mini-row:not(.selected) .mini-track i').style.width = (baseline[condition] / 3 * 100) + '%';
    });
    get('car-comparison').textContent = baseline.oppositeCar + '% → ' + selected.oppositeCar + '%';
    get('mitigation-table-caption').textContent = 'Table 4 · ' + entry.direction + ' · selected conditions.';
    get('mitigation-table').innerHTML = entry.scores.map(score => '<tr><th scope="row">' + score.name + '</th><td>' + format(score.gold) + '</td><td>' + format(score.opposite) + '</td><td>' + score.oppositeCar + '%</td></tr>').join('');
    get('mitigation-status').textContent = entry.direction + ', ' + selected.name + '. Gold Fidelity: ' + format(baseline.gold) + ' to ' + format(selected.gold) + '. Opposite-meaning Fidelity: ' + format(baseline.opposite) + ' to ' + format(selected.opposite) + '. Opposite-meaning CAR: ' + baseline.oppositeCar + ' to ' + selected.oppositeCar + ' percent.';
  }
  get('mitigation-direction').addEventListener('change', renderMitigation);
  get('strategy-select').addEventListener('change', renderMitigation);

  get('copy-citation').addEventListener('click', async () => {
    const citation = get('bibtex').textContent.trim();
    let copied = false;
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(citation);
        copied = true;
      }
    } catch (_) { /* Try a local-file-compatible clipboard fallback below. */ }
    if (!copied) {
      const textarea = document.createElement('textarea');
      textarea.value = citation;
      textarea.setAttribute('readonly', '');
      textarea.style.cssText = 'position:fixed;left:-9999px;top:0';
      document.body.appendChild(textarea);
      textarea.select();
      try { copied = document.execCommand('copy'); } catch (_) { copied = false; }
      textarea.remove();
      get('copy-citation').focus({ preventScroll: true });
    }
    get('copy-citation').querySelector('.copy-label').textContent = copied ? 'Copied!' : 'Copy citation';
    get('copy-status').textContent = copied ? 'BibTeX copied to clipboard.' : 'Select the citation text or download the BibTeX file.';
  });

  const dialog = get('figure-dialog');
  let opener;
  document.querySelectorAll('[data-figure]').forEach(button => {
    button.addEventListener('click', () => {
      opener = button;
      get('figure-image').src = button.dataset.figure;
      get('figure-image').alt = button.dataset.caption;
      get('figure-caption').textContent = button.dataset.caption;
      dialog.showModal();
      document.body.classList.add('dialog-open');
    });
  });
  get('close-figure').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => {
    if (event.target !== dialog) return;
    const bounds = dialog.getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
  });
  dialog.addEventListener('close', () => {
    document.body.classList.remove('dialog-open');
    if (opener) opener.focus({ preventScroll: true });
  });
})();
