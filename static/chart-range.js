(() => {
  const storageKey = 'ollama-chart-range';
  const validPeriods = new Set(['24h', '7d', '30d']);

  document.querySelectorAll('[data-range-switcher]').forEach((switcher) => {
    const scope = switcher.closest('main') || document;
    const buttons = Array.from(switcher.querySelectorAll('[data-range]'));
    const panels = Array.from(scope.querySelectorAll('[data-chart-period]'));

    const savedPeriod = (() => {
      try {
        return window.localStorage.getItem(storageKey);
      } catch (_) {
        return null;
      }
    })();
    const hashPeriod = window.location.hash.match(/^#range-(24h|7d|30d)$/)?.[1];
    const initialPeriod = hashPeriod || (validPeriods.has(savedPeriod) ? savedPeriod : '24h');

    const activate = (period, updateLocation = true) => {
      if (!validPeriods.has(period)) return;
      buttons.forEach((button) => {
        const selected = button.dataset.range === period;
        button.classList.toggle('active', selected);
        button.setAttribute('aria-selected', String(selected));
        button.tabIndex = selected ? 0 : -1;
      });
      panels.forEach((panel) => {
        panel.hidden = panel.dataset.chartPeriod !== period;
      });
      try {
        window.localStorage.setItem(storageKey, period);
      } catch (_) {
        // The selector still works when private browsing blocks local storage.
      }
      if (updateLocation) {
        window.history.replaceState(null, '', `#range-${period}`);
      }
    };

    buttons.forEach((button, index) => {
      button.addEventListener('click', () => activate(button.dataset.range));
      button.addEventListener('keydown', (event) => {
        if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
        event.preventDefault();
        let nextIndex = index;
        if (event.key === 'ArrowLeft') nextIndex = (index - 1 + buttons.length) % buttons.length;
        if (event.key === 'ArrowRight') nextIndex = (index + 1) % buttons.length;
        if (event.key === 'Home') nextIndex = 0;
        if (event.key === 'End') nextIndex = buttons.length - 1;
        buttons[nextIndex].focus();
        activate(buttons[nextIndex].dataset.range);
      });
    });

    activate(initialPeriod, false);
  });
})();
