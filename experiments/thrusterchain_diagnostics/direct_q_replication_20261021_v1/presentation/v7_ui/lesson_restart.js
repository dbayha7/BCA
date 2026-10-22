(function () {
  'use strict';
  // Capture the recap's existing link before the lesson navigation saves progress.
  // Reloading clears the controllers' private state as well as the saved answers.
  document.addEventListener('click', function (event) {
    var target = event.target;
    var control = target && target.closest && target.closest(
      '[data-lesson-restart], #learn-recap .ls-again[href="#learn-world"]'
    );
    if (!control) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    ['tc-learn-last', 'tc-learn-seen', 'tc-learn-quiz'].forEach(function (key) {
      try { window.localStorage.removeItem(key); } catch (_) { /* Storage can be disabled for local files. */ }
    });
    window.history.replaceState(null, '', '#learn-world');
    window.location.reload();
  }, true);
}());
