/* ===========================================
   Sana Documentation - Shared Theme Script
   =========================================== */
(function () {
  /* ---------- Centralized Theme Icons ---------- */
  var ICONS = {
    sun: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M6.76 4.84l-1.8-1.79-1.41 1.41 1.79 1.79zM4 10.5H1v2h3zm9-9.95h-2V3.5h2zm7.45 3.91l-1.41-1.41-1.79 1.79 1.41 1.41zM17.24 18.16l1.79 1.8 1.41-1.41-1.8-1.79zM20 10.5v2h3v-2zm-8-5c-3.31 0-6 2.69-6 6s2.69 6 6 6 6-2.69 6-6-2.69-6-6-6zm-1 16.95h2V19.5h-2zm-7.45-3.91l1.41 1.41 1.79-1.8-1.41-1.41z"/></svg>',
    moon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M9 2c-1.05 0-2.05.16-3 .46 4.06 1.27 7 5.06 7 9.54 0 4.48-2.94 8.27-7 9.54.95.3 1.95.46 3 .46 5.52 0 10-4.48 10-10S14.52 2 9 2z"/></svg>'
  };

  /* ---------- Inject Icons into Palette Toggle ---------- */
  var labels = document.querySelectorAll(".md-header__option label.md-header__button");
  labels.forEach(function (label) {
    if (label.getAttribute("for") === "__palette_1") {
      label.innerHTML = ICONS.sun;
    } else if (label.getAttribute("for") === "__palette_0") {
      label.innerHTML = ICONS.moon;
    }
  });

  /* ---------- Auto-detect System Theme (First Visit) ---------- */
  if (typeof __md_get === "function") {
    var saved = __md_get("__palette");
    if (!saved) {
      var prefersDark = window.matchMedia("(prefers-color-scheme: dark)");
      var selector = prefersDark.matches
        ? "[data-md-color-media='(prefers-color-scheme: dark)']"
        : "[data-md-color-media='(prefers-color-scheme: light)']";
      var input = document.querySelector(selector);
      if (input) {
        input.click();
      }
    }
  }
})();
