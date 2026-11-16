"use strict";

const byId = (id) => document.getElementById(id);
const names = {
  evidence_support: "证据支持",
  entity_role: "实体角色",
  relation_support: "关系支持",
  temporal_scope: "时间范围",
  temporal_kind: "时间类型",
  evidence_novelty: "证据新颖性",
  replacement_suitability: "替换适合度",
};
const state = {
  fields: [],
  ids: [],
  annotator: "",
  caseId: "",
  completed: new Set(),
  filter: "all",
  busy: false,
  saved: "",
};
const annotatorKey = "hoprag-human-review-annotator";
const review = byId("review-form");
const message = byId("status");
let messageTimer;

function create(tag, content, className) {
  const node = document.createElement(tag);
  if (content !== undefined) node.textContent = content;
  if (className) node.className = className;
  return node;
}

function notify(text, isError = false) {
  window.clearTimeout(messageTimer);
  message.hidden = !text;
  message.textContent = text;
  message.classList.toggle("is-error", isError);
  if (text && !isError) messageTimer = window.setTimeout(() => notify(""), 4500);
}

async function request(path, payload) {
  const response = await fetch(path, payload === undefined ? undefined : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error || "操作失败（HTTP " + response.status + "）。");
  }
  return response;
}

async function json(path, payload) {
  return (await request(path, payload)).json();
}

function rememberAnnotator(annotator) {
  try {
    if (annotator) localStorage.setItem(annotatorKey, annotator);
    else localStorage.removeItem(annotatorKey);
  } catch {
    // The server still persists progress when browser storage is unavailable.
  }
}

function populateFields() {
  const container = byId("annotation-fields");
  for (const { name, description, options } of state.fields) {
    const field = create("div", undefined, "field");
    const label = create("label", names[name] || name);
    label.htmlFor = name;
    label.append(" ", create("span", name, "field-name"));
    const select = create("select");
    select.id = name;
    select.name = name;
    select.required = true;
    const empty = create("option", "请选择");
    empty.value = "";
    empty.disabled = true;
    empty.selected = true;
    select.append(empty);
    for (const { value, label: explanation } of options) {
      const option = create("option", explanation ? value + " · " + explanation : value);
      option.value = value;
      select.append(option);
    }
    const help = create("p", description || "", "field-help");
    help.id = name + "-help";
    select.setAttribute("aria-describedby", help.id);
    field.append(label, select, help);
    container.append(field);
  }
}

function values() {
  return Object.fromEntries(new FormData(review));
}

function hasUnsaved() {
  return state.caseId && JSON.stringify(values()) !== state.saved;
}

function mayLeave() {
  return !hasUnsaved() || window.confirm("当前案例有未保存修改。离开后将恢复最近一次保存的记录。确定离开？");
}

window.addEventListener("beforeunload", (event) => {
  if (hasUnsaved()) {
    event.preventDefault();
    event.returnValue = "";
  }
});

function filteredIds() {
  if (state.filter === "all") return state.ids;
  return state.ids.filter((id) => state.completed.has(id) === (state.filter === "complete"));
}

function adjacent(direction) {
  const matching = filteredIds();
  return direction < 0
    ? matching.findLast((id) => id < state.caseId)
    : matching.find((id) => id > state.caseId);
}

function syncNavigation() {
  const matching = filteredIds();
  const list = byId("case-list");
  const current = create("option", matching.length ? "选择案例" : "无符合条件的案例");
  current.value = "";
  current.disabled = true;
  list.replaceChildren(current);
  for (const id of matching) {
    const option = create("option", id + (state.completed.has(id) ? " · 已完成" : " · 未完成"));
    option.value = id;
    list.append(option);
  }
  list.value = matching.includes(state.caseId) ? state.caseId : "";
  list.disabled = state.busy || !matching.length;
  const completed = state.completed.has(state.caseId);
  byId("case-state").textContent = state.caseId ? (completed ? "已完成" : "未完成") : "";
  byId("case-state").classList.toggle("is-complete", completed);
  byId("previous-case").disabled = state.busy || !adjacent(-1);
  byId("next-case").disabled = state.busy || !adjacent(1);
  byId("progress").value = state.completed.size;
  byId("progress").max = state.ids.length;
  byId("progress-text").textContent = "已完成 " + state.completed.size + " / " + state.ids.length;
}

function setBusy(busy) {
  state.busy = busy;
  for (const id of [
    "case-filter", "case-target", "jump-button", "switch-annotator", "export-csv",
    "save-review", "annotator",
  ]) byId(id).disabled = busy;
  byId("open-session").disabled = busy || !state.ids.length;
  // Read-only text areas remain in FormData, so pending writes never look like edits.
  for (const control of review.querySelectorAll("textarea")) control.readOnly = busy;
  // Use inert to lock review controls without changing their form values during writes.
  review.inert = busy;
  syncNavigation();
}

function documentMeta({ source, published, url }) {
  return [
    source ? "媒体 " + source : "",
    published ? "发布 " + published : "",
    url ? "原报道 " + url : "",
  ].filter(Boolean).join(" · ");
}

function renderMaterials(record) {
  byId("case-question").textContent = record.question;
  const container = byId("case-groups");
  const fragment = document.createDocumentFragment();
  for (const group of record.groups) {
    const section = create("section", undefined, "material-group");
    section.dataset.group = group.key;
    section.append(create("h3", group.title));
    if (group.document) {
      const article = create("header", undefined, "document");
      article.append(
        create("h4", group.document.title),
        create("p", documentMeta(group.document), "metadata"),
      );
      section.append(article);
    }
    for (const windowInfo of group.windows) {
      const { ref, title, line, meta, text, raw } = windowInfo;
      const article = create("article", undefined, "window");
      article.dataset.ref = ref;
      const heading = create("h4");
      heading.append(create("span", ref, "ref"), create("span", title));
      article.append(heading);
      const details = [!group.document ? documentMeta(windowInfo) : "", line ? "案例包第 " + line + " 行" : "", meta]
        .filter(Boolean).join("\n");
      if (details) article.append(create("p", details, "metadata"));
      article.append(create("pre", text, "window-text"));
      if (raw !== undefined && raw !== null) {
        const folded = create("details", undefined, "raw-window");
        folded.append(create("summary", "查看完整原始 Chunk（仅供定位）"), create("pre", raw, "window-text"));
        article.append(folded);
      }
      section.append(article);
    }
    fragment.append(section);
  }
  container.replaceChildren(fragment);
  byId("materials").scrollTop = 0;
}

function renderCase(payload) {
  const record = payload.case;
  state.caseId = record.id;
  byId("case-number").textContent = record.id;
  byId("case-target").value = record.id;
  renderMaterials(record);
  for (const control of review.querySelectorAll("select, textarea")) {
    control.value = payload.annotation?.[control.name] ?? "";
  }
  state.saved = JSON.stringify(values());
  document.querySelector(".annotation").scrollTop = 0;
  syncNavigation();
}

function retrieveCase(id) {
  return json("/api/case?" + new URLSearchParams({ id, annotator: state.annotator }));
}

async function loadCase(id) {
  if (!id || id === state.caseId || state.busy) return;
  if (!state.ids.includes(id)) {
    notify("请填写有效的案例编号。", true);
    return;
  }
  if (!mayLeave()) {
    syncNavigation();
    return;
  }
  notify("");
  setBusy(true);
  try {
    const payload = await retrieveCase(id);
    renderCase(payload);
    try {
      await json("/api/position", { annotator: state.annotator, case_id: id });
    } catch (error) {
      notify("已打开此案例。上次查看位置未能保存：" + error.message, true);
    }
  } catch (error) {
    notify(error.message, true);
  } finally {
    setBusy(false);
  }
}

byId("sign-in").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.busy || !state.ids.length) return;
  const annotator = byId("annotator").value.trim();
  if (!/^[a-z0-9][a-z0-9_-]{0,39}$/.test(annotator)) {
    notify("标识须以小写英文字母或数字开头，最多 40 个字符。可用小写字母、数字、短横线和下划线。", true);
    return;
  }
  notify("");
  setBusy(true);
  try {
    const session = await json("/api/session", { annotator });
    state.annotator = session.annotator;
    const id = state.ids.includes(session.last_case) ? session.last_case : state.ids[0];
    const payload = await retrieveCase(id);
    state.completed = new Set(session.completed);
    state.filter = "all";
    byId("case-filter").value = "all";
    renderCase(payload);
    rememberAnnotator(state.annotator);
    byId("annotator-name").textContent = state.annotator;
    byId("welcome").hidden = true;
    byId("header-actions").hidden = false;
    byId("workspace").hidden = false;
  } catch (error) {
    state.annotator = "";
    notify(error.message, true);
  } finally {
    setBusy(false);
  }
});

byId("switch-annotator").addEventListener("click", () => {
  if (state.busy || !mayLeave()) return;
  state.annotator = "";
  state.caseId = "";
  state.completed.clear();
  state.saved = "";
  rememberAnnotator("");
  review.reset();
  byId("welcome").hidden = false;
  byId("header-actions").hidden = true;
  byId("workspace").hidden = true;
  byId("annotator").value = "";
  byId("annotator").focus();
  notify("");
});

byId("case-filter").addEventListener("change", (event) => {
  state.filter = event.target.value;
  syncNavigation();
});
byId("case-list").addEventListener("change", (event) => loadCase(event.target.value));
byId("previous-case").addEventListener("click", () => loadCase(adjacent(-1)));
byId("next-case").addEventListener("click", () => loadCase(adjacent(1)));
byId("jump-case").addEventListener("submit", (event) => {
  event.preventDefault();
  const input = byId("case-target").value.trim().toLowerCase();
  const id = /^\d{1,4}$/.test(input) ? "evb-" + input.padStart(4, "0") : input;
  loadCase(id);
});

review.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.busy || !state.caseId) return;
  const annotation = values();
  notify("");
  setBusy(true);
  try {
    const saved = await json("/api/save", {
      annotator: state.annotator,
      case_id: state.caseId,
      values: annotation,
    });
    state.completed = new Set(saved.completed);
    if (saved.annotation) {
      for (const control of review.querySelectorAll("select, textarea")) {
        control.value = saved.annotation[control.name] ?? "";
      }
    }
    state.saved = JSON.stringify(values());
    notify(state.caseId + " 已保存并标记完成。");
  } catch (error) {
    notify(error.message, true);
  } finally {
    setBusy(false);
  }
});

byId("export-csv").addEventListener("click", async () => {
  if (state.busy || !state.annotator) return;
  if (hasUnsaved() && !window.confirm("CSV 只包含已保存的标注。确定导出？")) return;
  notify("");
  setBusy(true);
  try {
    const response = await request("/api/export", { annotator: state.annotator });
    const url = URL.createObjectURL(await response.blob());
    const link = create("a");
    link.href = url;
    link.download = "HUMAN_ANNOTATION_350.csv";
    document.body.append(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    notify("CSV 已导出。");
  } catch (error) {
    notify(error.message, true);
  } finally {
    setBusy(false);
  }
});

async function start() {
  try {
    const { fields, ids } = await json("/api/bootstrap");
    state.fields = fields;
    state.ids = ids;
    populateFields();
    try {
      byId("annotator").value = localStorage.getItem(annotatorKey) || "";
    } catch {
      // Users can enter their identifier again.
    }
    setBusy(false);
  } catch (error) {
    notify(error.message, true);
  }
}

start();
