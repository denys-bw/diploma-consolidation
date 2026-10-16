"use strict";

const POLL_INTERVAL_MS = 1500;
const SIZE_UNITS = ["Б", "КБ", "МБ", "ГБ"];

function decimal(value, digits) {
  return value.toFixed(digits).replace(".", ",");
}

function fileSize(bytes) {
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < SIZE_UNITS.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return unit === 0 ? `${value} ${SIZE_UNITS[0]}` : `${decimal(value, 1)} ${SIZE_UNITS[unit]}`;
}

async function fetchJson(url) {
  const response = await fetch(url, { headers: { Accept: "application/json" }, cache: "no-store" });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  return response.json();
}

function initTabs(root) {
  const tabs = Array.from(root.querySelectorAll(".tabs__tab"));
  const panels = Array.from(root.querySelectorAll(".tabs__panel"));
  if (tabs.length === 0) {
    return;
  }
  const activate = (name, remember) => {
    tabs.forEach((tab) => {
      const active = tab.dataset.tab === name;
      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
    });
    panels.forEach((panel) => panel.classList.toggle("is-active", panel.id === `tab-${name}`));
    if (remember) {
      history.replaceState(null, "", `#${name}`);
    }
  };
  tabs.forEach((tab, position) => {
    tab.addEventListener("click", () => activate(tab.dataset.tab, true));
    tab.addEventListener("keydown", (event) => {
      const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
      if (step === undefined) {
        return;
      }
      event.preventDefault();
      const next = tabs[(position + step + tabs.length) % tabs.length];
      next.focus();
      activate(next.dataset.tab, true);
    });
  });
  const requested = location.hash.slice(1);
  activate(tabs.some((tab) => tab.dataset.tab === requested) ? requested : tabs[0].dataset.tab, false);
}

function initDropzone(zone) {
  const input = zone.querySelector('input[type="file"]');
  const list = zone.querySelector("[data-file-list]");
  if (!input || !list) {
    return;
  }
  const render = () => {
    const files = Array.from(input.files);
    list.replaceChildren(
      ...files.map((file) => {
        const item = document.createElement("li");
        item.textContent = `${file.name} · ${fileSize(file.size)}`;
        return item;
      }),
    );
    list.hidden = files.length === 0;
  };
  let depth = 0;
  zone.addEventListener("dragenter", (event) => {
    event.preventDefault();
    depth += 1;
    zone.classList.add("is-over");
  });
  zone.addEventListener("dragover", (event) => event.preventDefault());
  zone.addEventListener("dragleave", () => {
    depth = Math.max(depth - 1, 0);
    zone.classList.toggle("is-over", depth > 0);
  });
  zone.addEventListener("drop", (event) => {
    event.preventDefault();
    depth = 0;
    zone.classList.remove("is-over");
    if (event.dataTransfer && event.dataTransfer.files.length > 0) {
      input.files = event.dataTransfer.files;
      render();
    }
  });
  input.addEventListener("change", render);
}

function initAhp(root) {
  const url = root.dataset.url;
  const selects = Array.from(root.querySelectorAll("select[data-ahp-pair]"));
  const lambda = root.querySelector("[data-ahp-lambda]");
  const index = root.querySelector("[data-ahp-ci]");
  const ratio = root.querySelector("[data-ahp-cr]");
  const verdict = root.querySelector("[data-ahp-verdict]");
  let latest = 0;
  const render = (result) => {
    Object.entries(result.weights).forEach(([criterion, weight]) => {
      const fill = root.querySelector(`[data-weight-fill="${criterion}"]`);
      const value = root.querySelector(`[data-weight-value="${criterion}"]`);
      if (fill) {
        fill.style.width = `${(weight * 100).toFixed(1)}%`;
      }
      if (value) {
        value.textContent = decimal(weight, 3);
      }
    });
    lambda.textContent = decimal(result.lambda_max, 3);
    index.textContent = decimal(result.consistency_index, 3);
    ratio.textContent = decimal(result.consistency_ratio, 3);
    verdict.textContent = result.consistent ? "узгоджено" : "неузгоджено";
    verdict.classList.toggle("badge--success", result.consistent);
    verdict.classList.toggle("badge--danger", !result.consistent);
  };
  const update = async () => {
    latest += 1;
    const request = latest;
    const query = new URLSearchParams(selects.map((select) => [select.name, select.value]));
    try {
      const result = await fetchJson(`${url}?${query}`);
      if (request === latest) {
        render(result);
      }
    } catch (error) {
      console.warn("Не вдалося перерахувати ваги МАІ", error);
    }
  };
  selects.forEach((select) => select.addEventListener("change", update));
  root.querySelectorAll("[data-preset]").forEach((button) => {
    button.addEventListener("click", () => {
      const values = JSON.parse(button.dataset.preset);
      selects.forEach((select, position) => {
        select.value = values[position];
      });
      update();
    });
  });
  update();
}

function initRunStatus(root) {
  const url = root.dataset.runStatus;
  const label = root.querySelector("[data-stage-label]");
  const bar = root.querySelector("[data-progress-bar]");
  const stages = Array.from(root.querySelectorAll("[data-stage]"));
  const render = (state) => {
    label.textContent = state.stage_label;
    bar.style.width = `${Math.round(state.progress * 100)}%`;
    const current = stages.findIndex((item) => item.dataset.stage === state.stage);
    stages.forEach((item, position) => {
      item.classList.toggle("is-done", current >= 0 && position < current);
      item.classList.toggle("is-active", position === current);
    });
  };
  const poll = async () => {
    try {
      const state = await fetchJson(url);
      if (state.done) {
        window.location.reload();
        return;
      }
      render(state);
    } catch (error) {
      console.warn("Не вдалося отримати стан обробки", error);
    }
    window.setTimeout(poll, POLL_INTERVAL_MS);
  };
  window.setTimeout(poll, POLL_INTERVAL_MS);
}

function initRoleSelects() {
  document.querySelectorAll("[data-role-select]").forEach((select) => {
    select.addEventListener("change", () => {
      const row = select.closest("tr");
      if (row) {
        row.classList.toggle("is-off", select.value !== "attribute");
      }
    });
  });
}

function initConditional(element) {
  const [name, expected] = element.dataset.showWhen.split("=");
  const inputs = Array.from(document.querySelectorAll(`input[name="${name}"]`));
  const update = () => {
    const chosen = inputs.find((input) => input.checked);
    element.hidden = !chosen || chosen.value !== expected;
  };
  inputs.forEach((input) => input.addEventListener("change", update));
  update();
}

function initForms() {
  document.addEventListener("submit", (event) => {
    const form = event.target;
    if (!(form instanceof HTMLFormElement)) {
      return;
    }
    const message = form.dataset.confirm;
    if (message && !window.confirm(message)) {
      event.preventDefault();
      return;
    }
    if (form.dataset.busy === "true") {
      event.preventDefault();
      return;
    }
    form.dataset.busy = "true";
    window.setTimeout(() => {
      form.querySelectorAll("button").forEach((button) => {
        if (button.type === "submit") {
          button.disabled = true;
        }
      });
    }, 0);
  });
  window.addEventListener("pageshow", (event) => {
    if (!event.persisted) {
      return;
    }
    document.querySelectorAll('form[data-busy="true"]').forEach((form) => {
      delete form.dataset.busy;
      form.querySelectorAll("button").forEach((button) => {
        button.disabled = false;
      });
    });
  });
}

function init() {
  document.querySelectorAll("[data-tabs]").forEach(initTabs);
  document.querySelectorAll("[data-dropzone]").forEach(initDropzone);
  document.querySelectorAll("[data-ahp]").forEach(initAhp);
  document.querySelectorAll("[data-run-status]").forEach(initRunStatus);
  document.querySelectorAll("[data-show-when]").forEach(initConditional);
  initRoleSelects();
  initForms();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
