(function () {
  'use strict';
  var root = document.documentElement;
  var bar = document.getElementById('nv-bar');
  if (!bar) return;
  var pending = false;
  function measure() {
    pending = false;
    if (!root.classList.contains('nv-js')) {
      root.style.removeProperty('--nv-bar-space');
      return;
    }
    var style = window.getComputedStyle(bar);
    if (style.display === 'none') return;
    var height = bar.getBoundingClientRect().height;
    if (height > 0) {
      var reserve = Math.ceil(height + Math.max(0, parseFloat(style.bottom) || 0) + 8);
      if (root.style.getPropertyValue('--nv-bar-space') !== reserve + 'px') {
        root.style.setProperty('--nv-bar-space', reserve + 'px');
      }
    }
  }
  function queue() {
    if (pending) return;
    pending = true;
    window.requestAnimationFrame(measure);
  }
  window.addEventListener('resize', queue, { passive: true });
  window.addEventListener('load', queue);
  window.addEventListener('afterprint', queue);
  if (window.visualViewport) window.visualViewport.addEventListener('resize', queue, { passive: true });
  if ('ResizeObserver' in window) new window.ResizeObserver(queue).observe(bar);
  measure();
}());
