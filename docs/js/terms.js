// Explain terms where they first appear in prose. Native buttons support touch and keyboard.
(() => {
  const definitions = [
    [/Gaussian[ -]process(?: models)?/i, 'A model that predicts a function from measurements and quantifies uncertainty in those predictions.'],
    [/acquisition (?:function|rule)/i, 'A rule that scores possible measurement locations to choose where to measure next.'],
    [/geometric equalizer/i, 'A location-dependent weight that reduces the average boundary inflation in the acquisition score. It does not use measured values.'],
    [/reconstruction warp/i, 'A transformation of the acquisition coordinates, built from the predicted function, that changes where new measurements are selected.'],
    [/posterior mean/i, 'The model’s predicted function after accounting for the measurements.'],
    [/posterior variance/i, 'The uncertainty remaining in the function value after accounting for the measurements, expressed as a squared standard deviation.'],
    [/posterior covariance/i, 'How uncertainty at two locations is related after accounting for the measurements.'],
    [/gradient(?:s)?/i, 'The direction and rate of local change in a function. A large gradient magnitude means the function changes rapidly.'],
    [/lengthscale/i, 'A kernel parameter that sets how quickly correlation decreases with distance.'],
    [/kernel/i, 'The covariance function: it describes how function values at two locations are related in the model.'],
    [/Regret/i, 'The difference between the best possible function value and the best value found so far.'],
  ];
  const tip = document.createElement('div');
  tip.id = 'term-definition'; tip.className = 'term-tooltip'; tip.role = 'tooltip'; tip.hidden = true;
  document.body.append(tip);
  let active, dismissTimer;
  function close() {
    clearTimeout(dismissTimer);
    if (active) { active.removeAttribute('aria-describedby'); active.setAttribute('aria-expanded', 'false'); }
    active = null; tip.hidden = true;
  }
  function show(button, text) {
    close(); active = button; tip.textContent = text; tip.hidden = false;
    button.setAttribute('aria-describedby', tip.id); button.setAttribute('aria-expanded', 'true');
    const r = button.getBoundingClientRect(), t = tip.getBoundingClientRect();
    tip.style.left = `${Math.max(12, Math.min(r.left, innerWidth - t.width - 12))}px`;
    tip.style.top = `${r.bottom + t.height + 12 < innerHeight ? r.bottom + 8 : Math.max(12, r.top - t.height - 8)}px`;
  }
  for (const [pattern, definition] of definitions) {
    const paragraphs = document.querySelectorAll('p.lead, p.lead-sub, .explain p, p.caption, p.demo-text, p.note, details.math p');
    let found = false;
    for (const p of paragraphs) {
      const walker = document.createTreeWalker(p, NodeFilter.SHOW_TEXT);
      let node;
      while ((node = walker.nextNode())) {
        if (node.parentElement.closest('button, a, .eq')) continue;
        const match = pattern.exec(node.textContent);
        if (!match) continue;
        const rest = node.splitText(match.index);
        rest.splitText(match[0].length);
        const button = document.createElement('button');
        button.type = 'button'; button.className = 'term'; button.textContent = match[0];
        button.setAttribute('aria-expanded', 'false'); button.setAttribute('aria-controls', tip.id);
        rest.replaceWith(button);
        button.addEventListener('pointerenter', e => { if (e.pointerType !== 'touch') show(button, definition); });
        button.addEventListener('focus', () => show(button, definition));
        button.addEventListener('click', () => show(button, definition));
        button.addEventListener('blur', close);
        button.addEventListener('pointerleave', e => {
          if (e.pointerType !== 'touch' && e.relatedTarget !== tip && document.activeElement !== button) dismissTimer = setTimeout(close, 180);
        });
        found = true; break;
      }
      if (found) break;
    }
  }
  tip.addEventListener('pointerenter', () => clearTimeout(dismissTimer));
  tip.addEventListener('pointerleave', () => { if (document.activeElement !== active) close(); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') close(); });
  document.addEventListener('pointerdown', e => { if (!e.target.closest('.term, .term-tooltip')) close(); });
  window.addEventListener('resize', close);
  window.addEventListener('scroll', () => { if (active) show(active, tip.textContent); }, { passive: true });
})();
