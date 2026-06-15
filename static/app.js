const state = {
  cases: [],
  session: null,
  result: null,
  selectedCase: null,
  activeImageId: null,
  secondaryImageId: null,
  layout: "single",
  tool: "pan",
  views: new Map(),
  images: new Map(),
  regions: new Map(),
  runtime: null,
  rating: null,
  onboardingStep: 0,
  windowTimer: null,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const TOOL_LABELS = {
  pan: "Pan tool",
  window: "Window/level drag",
  ruler: "Ruler tool",
};

document.addEventListener("DOMContentLoaded", async () => {
  applySavedTheme();
  bindEvents();
  await Promise.all([loadCases(), loadRuntimeStatus()]);
  refreshIcons();
  updateProgress();
  updateViewerControls();
  if (!localStorage.getItem("radiology-onboarding-complete")) openOnboarding();
});

function bindEvents() {
  $("#blind-read").addEventListener("input", updateCommitState);
  $("#commit-read").addEventListener("click", runAnalysis);
  $("#new-session").addEventListener("click", resetRead);
  $("#file-input").addEventListener("change", createUploadSession);
  $("#theme-button").addEventListener("click", cycleTheme);
  $("#help-button").addEventListener("click", openOnboarding);
  $("#close-onboarding").addEventListener("click", closeOnboarding);
  $("#walkthrough-next").addEventListener("click", advanceOnboarding);
  $("#metadata-button").addEventListener("click", openMetadata);
  $$("[data-close-dialog]").forEach((button) => button.addEventListener("click", () => {
    document.getElementById(button.dataset.closeDialog).close();
  }));
  $$(".inspector-tab").forEach((button) => button.addEventListener("click", () => activatePanel(button.dataset.panel)));
  $$(".subtab").forEach((button) => button.addEventListener("click", () => activateSubpanel(button.dataset.subpanel)));
  $$("[data-layout]").forEach((button) => button.addEventListener("click", () => setLayout(button.dataset.layout)));
  $$("[data-tool]").forEach((button) => button.addEventListener("click", () => setTool(button.dataset.tool)));
  $("#window-preset").addEventListener("change", applyWindowPreset);
  $("#zoom-in").addEventListener("click", () => changeZoom(0.15));
  $("#zoom-out").addEventListener("click", () => changeZoom(-0.15));
  $("#fit-view").addEventListener("click", fitView);
  $("#rotate-view").addEventListener("click", rotateView);
  $("#flip-view").addEventListener("click", flipView);
  $("#invert-view").addEventListener("click", invertView);
  $("#reset-view").addEventListener("click", resetActiveView);
  $("#chat-send").addEventListener("click", sendChat);
  $("#chat-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendChat();
    }
  });
  $$(".chat-suggestions button").forEach((button) => button.addEventListener("click", () => {
    $("#chat-input").value = button.dataset.prompt;
    sendChat();
  }));
  $("#export-markdown").addEventListener("click", exportMarkdown);
  $("#export-json").addEventListener("click", exportJson);
  $$("#rating button").forEach((button) => button.addEventListener("click", () => setRating(Number(button.dataset.rating))));
  $("#submit-feedback").addEventListener("click", submitFeedback);
  bindViewport($("#primary-canvas"), "primary");
  bindViewport($("#secondary-canvas"), "secondary");
  document.addEventListener("keydown", handleShortcut);
}

async function loadCases() {
  try {
    const cases = await api("/api/cases");
    if (!Array.isArray(cases)) throw new Error("Practice studies are temporarily unavailable.");
    state.cases = cases;
    renderCases();
  } catch (error) {
    state.cases = [];
    renderCases();
    showError(error.message);
  }
}

async function loadRuntimeStatus() {
  try {
    state.runtime = await api("/api/status");
    const indicator = $("#runtime-state");
    const ready = ["ready", "demo", "on-demand"].includes(state.runtime.runtime_status);
    indicator.className = `runtime-state ${ready ? "ready" : "error"}`;
    indicator.querySelector("span").textContent = ready
      ? state.runtime.runtime_status === "demo"
        ? "Practice mode"
        : state.runtime.runtime_status === "on-demand"
          ? "GPU on demand"
          : "Models ready"
      : state.runtime.runtime_status === "loading" ? "Models loading" : "Runtime unavailable";
  } catch {
    $("#runtime-state").className = "runtime-state error";
    $("#runtime-state span").textContent = "Status unavailable";
  }
}

function renderCases() {
  const completed = completedCases();
  if (!state.cases.length) {
    $("#case-list").innerHTML = '<div class="case-empty">Practice studies could not be loaded. You can still open a local chest study.</div>';
    return;
  }
  $("#case-list").innerHTML = state.cases.map((item) => `
    <button class="case-item ${completed.includes(item.id) ? "complete" : ""}" data-case="${escapeHtml(item.id)}" ${item.available ? "" : "disabled"}>
      <strong>${escapeHtml(item.title)}</strong>
      <span>${completed.includes(item.id) ? "Completed" : escapeHtml(item.difficulty)}</span>
    </button>
  `).join("");
  $$(".case-item").forEach((button) => button.addEventListener("click", () => createDemoSession(button.dataset.case)));
}

async function createDemoSession(caseId) {
  if (!await clearCurrentSession()) return;
  const form = new FormData();
  form.append("case_id", caseId);
  await createSession(form, state.cases.find((item) => item.id === caseId));
}

async function createUploadSession(event) {
  const files = [...event.target.files];
  if (!files.length) return;
  if (!await clearCurrentSession()) return;
  const form = new FormData();
  files.forEach((file) => form.append("files", file));
  await createSession(form, { title: files[0].name, difficulty: "uploaded" });
  event.target.value = "";
}

async function createSession(form, context) {
  hideError();
  setBusy(true, "Opening study");
  try {
    state.session = await api("/api/sessions", { method: "POST", body: form });
    state.result = null;
    state.selectedCase = state.cases.find((item) => item.id === state.session.study.case_id) || null;
    state.activeImageId = state.session.study.primary_image_id;
    state.secondaryImageId = chooseSecondaryImage();
    state.views.clear();
    state.images.clear();
    state.regions.clear();
    $("#case-kicker").textContent = `${String(context.difficulty || "study").toUpperCase()} · ${state.session.study.source.toUpperCase()}`;
    $("#case-title").textContent = state.session.study.title;
    $$(".case-item").forEach((button) => button.classList.toggle("active", button.dataset.case === state.session.study.case_id));
    resetRead();
    renderStudyList();
    await loadVisibleImages();
    updateViewerControls();
    refreshIcons();
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy(false);
  }
}

async function clearCurrentSession() {
  if (state.session?.status === "complete" && !confirm("Close the current training session?")) return false;
  if (state.session) {
    await fetch(`/api/sessions/${state.session.id}`, { method: "DELETE" }).catch(() => {});
  }
  state.session = null;
  return true;
}

function resetRead() {
  state.result = null;
  state.regions.clear();
  if (state.session) state.session.status = "ready";
  $("#blind-read").value = "";
  $("#blind-read").disabled = false;
  $("#commit-read").classList.remove("hidden");
  $("#new-session").classList.add("hidden");
  $("#analysis-progress").classList.add("hidden");
  $("#evidence-content").classList.add("hidden");
  $("#chat-input").disabled = true;
  $("#chat-send").disabled = true;
  $("#export-markdown").disabled = true;
  $("#export-json").disabled = true;
  $("#chat-messages").innerHTML = '<div class="chat-empty">Complete an analysis to start a grounded conversation.</div>';
  $("#professor-state").textContent = "Waiting for analysis";
  activatePanel("read");
  updateCommitState();
  renderAllViewports();
}

function renderStudyList() {
  const images = state.session?.study.images || [];
  const twoUpButton = $('[data-layout="two-up"]');
  $("#study-heading").classList.toggle("hidden", images.length === 0);
  $("#image-count").textContent = String(images.length);
  twoUpButton.disabled = images.length < 2;
  if (images.length < 2 && state.layout === "two-up") setLayout("single");
  $("#study-list").innerHTML = images.map((image, index) => `
    <button class="study-item ${image.id === state.activeImageId ? "active" : ""}" data-image="${image.id}">
      <strong>${escapeHtml(image.label || `Image ${index + 1}`)}</strong>
      <span>${escapeHtml(image.projection || "Unknown")} · ${index + 1}/${images.length}</span>
    </button>
  `).join("");
  $$(".study-item").forEach((button) => button.addEventListener("click", async () => {
    state.activeImageId = button.dataset.image;
    if (state.layout === "two-up" && state.secondaryImageId === state.activeImageId) {
      state.secondaryImageId = state.session.study.primary_image_id;
    }
    renderStudyList();
    await loadVisibleImages();
    updateViewerControls();
  }));
}

function chooseSecondaryImage() {
  const images = state.session?.study.images || [];
  return images.find((item) => item.id !== state.session.study.primary_image_id && item.projection.includes("LAT"))?.id
    || images.find((item) => item.id !== state.session.study.primary_image_id)?.id
    || state.session?.study.primary_image_id
    || null;
}

function imageById(imageId) {
  return state.session?.study.images.find((item) => item.id === imageId) || null;
}

function viewState(imageId) {
  if (!state.views.has(imageId)) {
    const image = imageById(imageId);
    const original = image?.window_presets?.[0];
    state.views.set(imageId, {
      zoom: 1, panX: 0, panY: 0, rotation: 0, flipX: false, invert: false,
      center: original?.center ?? 127.5,
      width: original?.width ?? 255,
      preset: original?.id || "original",
      ruler: null,
    });
  }
  return state.views.get(imageId);
}

function activeViewState() {
  return state.activeImageId ? viewState(state.activeImageId) : null;
}

async function loadVisibleImages() {
  if (!state.session) return;
  const ids = [state.activeImageId];
  if (state.layout === "two-up" && state.secondaryImageId) ids.push(state.secondaryImageId);
  await Promise.all(ids.filter(Boolean).map(loadStudyImage));
  $("#viewer-empty").classList.add("hidden");
  renderAllViewports();
}

async function loadStudyImage(imageId) {
  const view = viewState(imageId);
  const image = imageById(imageId);
  if (!image) return;
  const url = `${image.image_url}?center=${encodeURIComponent(view.center)}&width=${encodeURIComponent(view.width)}&invert=${view.invert}`;
  if (state.images.get(imageId)?.url === url) return;
  const element = new Image();
  await new Promise((resolve, reject) => {
    element.onload = resolve;
    element.onerror = () => reject(new Error(`Could not render ${image.label}.`));
    element.src = url;
  });
  state.images.set(imageId, { element, url });
}

function renderAllViewports() {
  renderViewport("primary", state.activeImageId, $("#primary-canvas"), $("#primary-label"));
  renderViewport("secondary", state.secondaryImageId, $("#secondary-canvas"), $("#secondary-label"));
}

function renderViewport(slot, imageId, canvas, label) {
  const loaded = state.images.get(imageId);
  const image = imageById(imageId);
  if (!loaded || !image) {
    canvas.style.display = "none";
    return;
  }
  const scale = Math.min(1, 1400 / Math.max(loaded.element.naturalWidth, loaded.element.naturalHeight));
  canvas.width = Math.round(loaded.element.naturalWidth * scale);
  canvas.height = Math.round(loaded.element.naturalHeight * scale);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(loaded.element, 0, 0, canvas.width, canvas.height);
  visibleRegions(imageId).forEach((region) => drawRegion(ctx, canvas, region));
  drawRuler(ctx, canvas, viewState(imageId).ruler, image);
  const view = viewState(imageId);
  applyViewTransform(canvas, view);
  canvas.style.display = "block";
  label.textContent = `${image.projection || "Image"} · ${slot === "primary" ? "Active" : "Comparison"}`;
}

function applyViewTransform(canvas, view) {
  canvas.style.transform = `translate(${view.panX}px, ${view.panY}px) scale(${view.flipX ? -view.zoom : view.zoom}, ${view.zoom}) rotate(${view.rotation}deg)`;
}

function visibleRegions(imageId) {
  return [...state.regions.values()].filter((item) => item.image_id === imageId && item.visible && !item.removed);
}

function drawRegion(ctx, canvas, region) {
  const x = region.x1 * canvas.width;
  const y = region.y1 * canvas.height;
  const width = (region.x2 - region.x1) * canvas.width;
  const height = (region.y2 - region.y1) * canvas.height;
  const selected = region.selected;
  ctx.fillStyle = `rgba(87, 200, 197, ${region.opacity * 0.24})`;
  ctx.strokeStyle = selected ? "#ffffff" : "rgba(87, 200, 197, .95)";
  ctx.lineWidth = selected ? 4 : Math.max(2, canvas.width / 500);
  ctx.fillRect(x, y, width, height);
  ctx.strokeRect(x, y, width, height);
  ctx.font = `${Math.max(11, canvas.width / 66)}px IBM Plex Mono`;
  const label = region.label;
  const labelWidth = ctx.measureText(label).width + 12;
  ctx.fillStyle = "rgba(3, 8, 8, .82)";
  ctx.fillRect(x, Math.max(0, y - 21), labelWidth, 21);
  ctx.fillStyle = "#7ee2dd";
  ctx.fillText(label, x + 6, Math.max(15, y - 6));
}

function drawRuler(ctx, canvas, ruler, image) {
  if (!ruler) return;
  const x1 = ruler.x1 * canvas.width;
  const y1 = ruler.y1 * canvas.height;
  const x2 = ruler.x2 * canvas.width;
  const y2 = ruler.y2 * canvas.height;
  ctx.strokeStyle = "#f3cf82";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.fillStyle = "#f3cf82";
  ctx.beginPath(); ctx.arc(x1, y1, 3, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath(); ctx.arc(x2, y2, 3, 0, Math.PI * 2); ctx.fill();
  $("#measurement-value").textContent = measurementText(ruler, image);
}

function measurementText(ruler, image) {
  const dx = (ruler.x2 - ruler.x1) * image.width;
  const dy = (ruler.y2 - ruler.y1) * image.height;
  const spacing = image.metadata.pixel_spacing_mm;
  if (spacing) {
    return `${Math.sqrt((dx * spacing[1]) ** 2 + (dy * spacing[0]) ** 2).toFixed(1)} mm`;
  }
  return `${Math.round(Math.sqrt(dx ** 2 + dy ** 2))} px`;
}

function bindViewport(canvas, slot) {
  const viewport = canvas.closest(".viewport");
  let drag = null;
  let suppressClick = false;
  const finishDrag = () => {
    if (!drag) return;
    const wasWindowing = drag.windowing;
    suppressClick = suppressClick || drag.moved || wasWindowing || drag.ruler;
    drag = null;
    viewport.classList.remove("dragging");
    if (wasWindowing) scheduleWindowRender(true);
  };
  viewport.addEventListener("pointerdown", (event) => {
    const imageId = slot === "primary" ? state.activeImageId : state.secondaryImageId;
    if (!imageId || !state.images.has(imageId)) return;
    if (slot === "secondary") {
      [state.activeImageId, state.secondaryImageId] = [state.secondaryImageId, state.activeImageId];
      renderStudyList();
      renderAllViewports();
      updateViewerControls();
      return;
    }
    const view = viewState(imageId);
    const point = normalizedPoint(event, canvas);
    if (state.tool === "pan") drag = { x: event.clientX - view.panX, y: event.clientY - view.panY, moved: false };
    if (state.tool === "window") drag = { x: event.clientX, y: event.clientY, center: view.center, width: view.width, windowing: true, moved: false };
    if (state.tool === "ruler") {
      view.ruler = { x1: point.x, y1: point.y, x2: point.x, y2: point.y };
      drag = { ruler: true, moved: false };
      renderAllViewports();
    }
    viewport.classList.add("dragging");
    viewport.setPointerCapture(event.pointerId);
  });
  viewport.addEventListener("pointermove", (event) => {
    if (!drag || !state.activeImageId) return;
    const view = activeViewState();
    drag.moved = drag.moved || Math.abs(event.movementX) + Math.abs(event.movementY) > 2;
    if (drag.ruler) {
      const point = normalizedPoint(event, canvas);
      view.ruler.x2 = point.x;
      view.ruler.y2 = point.y;
      renderAllViewports();
    } else if (state.tool === "pan") {
      view.panX = event.clientX - drag.x;
      view.panY = event.clientY - drag.y;
      applyViewTransform(canvas, view);
    } else if (state.tool === "window") {
      const sensitivity = Math.max(Math.abs(drag.width), 1) / 300;
      view.width = Math.max(1, drag.width + (event.clientX - drag.x) * sensitivity * 2);
      view.center = drag.center - (event.clientY - drag.y) * sensitivity;
      view.preset = "custom";
      updateViewerControls();
      previewWindowLevel(canvas, drag, view);
    }
  });
  viewport.addEventListener("pointerup", (event) => {
    if (!drag && !suppressClick) {
      selectRegionAt(event, canvas);
      return;
    }
    finishDrag();
  });
  viewport.addEventListener("pointercancel", finishDrag);
  viewport.addEventListener("lostpointercapture", finishDrag);
  viewport.addEventListener("wheel", (event) => {
    if (!state.activeImageId || slot !== "primary") return;
    event.preventDefault();
    changeZoom(event.deltaY < 0 ? 0.12 : -0.12);
  }, { passive: false });
  viewport.addEventListener("dblclick", () => {
    if (slot === "primary") fitView();
  });
  canvas.addEventListener("click", (event) => {
    if (suppressClick) {
      suppressClick = false;
      return;
    }
    selectRegionAt(event, canvas);
  });
}

function previewWindowLevel(canvas, baseline, view) {
  const contrast = clamp(baseline.width / Math.max(view.width, 1), 0.35, 3);
  const brightness = clamp(1 + (baseline.center - view.center) / Math.max(baseline.width, 1), 0.45, 1.7);
  canvas.style.filter = `contrast(${contrast}) brightness(${brightness})`;
}

function normalizedPoint(event, canvas) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: clamp((event.clientX - rect.left) / rect.width, 0, 1),
    y: clamp((event.clientY - rect.top) / rect.height, 0, 1),
  };
}

function selectRegionAt(event, canvas) {
  if (!state.activeImageId) return;
  const point = normalizedPoint(event, canvas);
  let selected = null;
  visibleRegions(state.activeImageId).forEach((region) => {
    region.selected = false;
    if (point.x >= region.x1 && point.x <= region.x2 && point.y >= region.y1 && point.y <= region.y2) selected = region;
  });
  if (selected) selected.selected = true;
  renderRegions();
  renderAllViewports();
}

function setLayout(layout) {
  if (layout === "two-up" && (state.session?.study.images.length || 0) < 2) return;
  state.layout = layout;
  $$("[data-layout]").forEach((button) => {
    const active = button.dataset.layout === layout;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  $("#viewer-grid").className = `viewer-grid ${layout}`;
  $('.viewport[data-viewport="secondary"]').classList.toggle("hidden", layout !== "two-up");
  loadVisibleImages().catch((error) => showError(error.message));
}

function setTool(tool) {
  if (!state.activeImageId) return;
  state.tool = tool;
  $$("[data-tool]").forEach((button) => {
    const active = button.dataset.tool === tool;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  $$(".viewport").forEach((viewport) => {
    viewport.classList.toggle("windowing", tool === "window");
    viewport.classList.toggle("measuring", tool === "ruler");
  });
  $("#active-tool").textContent = TOOL_LABELS[tool] || `${tool[0].toUpperCase()}${tool.slice(1)} tool`;
}

function applyWindowPreset() {
  const image = imageById(state.activeImageId);
  const preset = image?.window_presets.find((item) => item.id === $("#window-preset").value);
  if (!preset) return;
  const view = activeViewState();
  view.center = preset.center;
  view.width = preset.width;
  view.preset = preset.id;
  updateViewerControls();
  scheduleWindowRender(true);
}

function scheduleWindowRender(immediate = false) {
  clearTimeout(state.windowTimer);
  state.windowTimer = setTimeout(async () => {
    try {
      await loadStudyImage(state.activeImageId);
      renderAllViewports();
    } catch (error) {
      showError(error.message);
    } finally {
      $("#primary-canvas").style.filter = "";
      $("#secondary-canvas").style.filter = "";
    }
  }, immediate ? 0 : 120);
}

function changeZoom(delta) {
  const view = activeViewState();
  if (!view) return;
  view.zoom = clamp(view.zoom + delta, 0.35, 4);
  updateViewerControls();
  applyViewTransform($("#primary-canvas"), view);
}

function fitView() {
  const view = activeViewState();
  if (!view) return;
  view.zoom = 1; view.panX = 0; view.panY = 0;
  updateViewerControls(); applyViewTransform($("#primary-canvas"), view);
}

function rotateView() {
  const view = activeViewState();
  if (!view) return;
  view.rotation = (view.rotation + 90) % 360;
  applyViewTransform($("#primary-canvas"), view);
}

function flipView() {
  const view = activeViewState();
  if (!view) return;
  view.flipX = !view.flipX;
  applyViewTransform($("#primary-canvas"), view);
}

function invertView() {
  const view = activeViewState();
  if (!view) return;
  view.invert = !view.invert;
  scheduleWindowRender(true);
}

function resetActiveView() {
  if (!state.activeImageId) return;
  state.views.delete(state.activeImageId);
  $("#measurement-value").textContent = "--";
  updateViewerControls();
  scheduleWindowRender(true);
}

function updateViewerControls() {
  const image = imageById(state.activeImageId);
  const view = image ? viewState(image.id) : null;
  const imageLoaded = Boolean(image);
  $$("[data-tool]").forEach((button) => { button.disabled = !imageLoaded; });
  $$("[data-layout]").forEach((button) => {
    button.disabled = button.dataset.layout === "two-up"
      ? (state.session?.study.images.length || 0) < 2
      : !imageLoaded;
  });
  ["fit-view", "zoom-out", "zoom-in", "rotate-view", "flip-view", "invert-view", "reset-view", "metadata-button"]
    .forEach((id) => { $(`#${id}`).disabled = !imageLoaded; });
  $("#window-preset").disabled = !image;
  $("#window-preset").innerHTML = image
    ? [...image.window_presets.map((item) => `<option value="${item.id}">${escapeHtml(item.label)}</option>`), '<option value="custom">Custom</option>'].join("")
    : "<option>Original</option>";
  if (view) $("#window-preset").value = view.preset;
  $("#zoom-level").textContent = view ? `${Math.round(view.zoom * 100)}%` : "100%";
  $("#window-width").textContent = view ? formatNumber(view.width) : "--";
  $("#window-center").textContent = view ? formatNumber(view.center) : "--";
  $("#image-dimensions").textContent = image ? `${image.width} × ${image.height}` : "--";
  $("#measurement-value").textContent = view?.ruler ? measurementText(view.ruler, image) : "--";
  if (!imageLoaded) $("#active-tool").textContent = "--";
}

async function runAnalysis() {
  if (!state.session || !$("#blind-read").value.trim()) return;
  hideError();
  $("#blind-read").disabled = true;
  $("#commit-read").classList.add("hidden");
  $("#new-session").classList.remove("hidden");
  activatePanel("evidence");
  $("#analysis-progress").classList.remove("hidden");
  $("#evidence-content").classList.add("hidden");
  resetProgress();
  try {
    await streamSse(`/api/sessions/${state.session.id}/analyze`, {
      observation: $("#blind-read").value.trim(),
    }, handleAnalysisEvent);
  } catch (error) {
    showError(error.message);
  }
}

function handleAnalysisEvent(event) {
  if (event.type === "queue") $("#progress-message").textContent = event.message || `Queue position ${event.position}`;
  if (event.type === "stage") {
    const row = $(`.progress-row[data-stage="${event.stage}"]`);
    if (row) row.className = `progress-row ${event.status}`;
    if (event.message) $("#progress-message").textContent = event.message;
  }
  if (event.type === "error") {
    showError(event.message);
    $("#progress-message").textContent = event.message;
  }
  if (event.type === "complete") {
    state.session = event.session;
    state.result = event.session.result;
    initializeRegions();
    renderEvidence();
    renderChat();
    renderAllViewports();
    $("#analysis-progress").classList.add("hidden");
    $("#evidence-content").classList.remove("hidden");
    $("#chat-input").disabled = false;
    $("#chat-send").disabled = false;
    $("#export-markdown").disabled = false;
    $("#export-json").disabled = false;
    $("#professor-state").textContent = "Grounded in this session";
    if (state.selectedCase) markCaseComplete(state.selectedCase.id);
  }
}

function resetProgress() {
  $$(".progress-row").forEach((row) => row.className = "progress-row");
  $("#progress-message").textContent = "Waiting for analysis.";
}

function initializeRegions() {
  state.regions.clear();
  (state.result?.evidence.regions || []).forEach((region) => state.regions.set(region.id, {
    ...region, visible: true, removed: false, opacity: 0.55, selected: false,
  }));
}

function renderEvidence() {
  const { evidence, reference, scorecard, tutor } = state.result;
  $("#subpanel-findings").innerHTML = [
    ...evidence.findings.slice(0, 8).map((item) => `
      <div class="evidence-row">
        <div><strong>${escapeHtml(item.label)}</strong><span class="source-label">${escapeHtml(item.source)}</span></div>
        <span class="numeric-label"><strong class="score-value">${Math.round(item.score * 100)}%</strong><button class="info-tip" data-tip="X-Raydar model probability for this label. It is not a calibrated clinical probability." aria-label="Explain finding probability">?</button></span>
        <p>${escapeHtml(item.explanation)}</p>
      </div>
    `),
    ...evidence.observations.map((item) => `
      <div class="evidence-row"><div><strong>${escapeHtml(item.label)}</strong><span class="source-label">${escapeHtml(item.source)}</span></div><p>${escapeHtml(item.description)}</p></div>
    `),
    ...evidence.model_runs.map(modelRunHtml),
  ].join("");
  renderRegions();
  const labels = reference?.labels?.length ? reference.labels.join(", ") : "No labeled abnormality";
  $("#subpanel-comparison").innerHTML = `
    <div class="evidence-row"><strong>Matched evidence</strong><span>${escapeHtml(scorecard.matched_findings.join(", ") || "None")}</span></div>
    <div class="evidence-row"><strong>Needs re-check</strong><span>${escapeHtml(scorecard.missed_findings.join(", ") || "None")}</span></div>
    <div class="score-grid">
      ${scoreCell("Coverage", scorecard.coverage_score, "How many high-ranking evidence labels were mentioned in the blind read.")}
      ${scoreCell("Technique", scorecard.technique_score, "Whether projection, image quality, and key negatives were documented.")}
      ${scoreCell("Uncertainty", scorecard.uncertainty_score, "Whether the blind read used appropriate uncertainty language.")}
    </div>
    <div class="evidence-row"><strong>Overall blind-read score</strong><span class="numeric-label"><strong class="score-value">${scorecard.total_score}</strong><button class="info-tip" data-tip="Educational heuristic combining finding coverage, read technique, and uncertainty wording. Not a clinical competency score." aria-label="Explain total score">?</button></span><p>${escapeHtml(scorecard.practice_focus)}</p></div>
    ${reference ? `<div class="reference-block"><span>PUBLIC CASE REFERENCE</span><strong>${escapeHtml(labels)}</strong><p>${escapeHtml(reference.teaching_point)} Source: ${escapeHtml(reference.source)}.</p></div>` : ""}
    <div class="professor-review">
      ${reviewSection("What you said", [tutor.student_read_assessment])}
      ${reviewSection("What the models suggest", tutor.model_evidence)}
      ${reviewSection("Professor assessment", [tutor.professor_assessment])}
      ${reviewSection("How to read it", tutor.reading_approach)}
      ${reviewSection("Uncertainty", tutor.uncertainty)}
    </div>
    ${modelRunHtml(tutor.model_run)}
  `;
  refreshIcons();
}

function reviewSection(label, items) {
  return `<section class="review-section"><span>${escapeHtml(label)}</span>${(items || []).map((item) => `<p>${escapeHtml(item)}</p>`).join("")}</section>`;
}

function scoreCell(label, value, tip) {
  return `<div class="score-cell"><span>${label}</span><span class="numeric-label"><strong>${value}</strong><button class="info-tip" data-tip="${escapeHtml(tip)}" aria-label="Explain ${escapeHtml(label)} score">?</button></span></div>`;
}

function modelRunHtml(run) {
  if (!run) return "";
  const ttft = run.time_to_first_token_ms == null ? "" : ` · TTFT ${run.time_to_first_token_ms} ms`;
  const source = run.model_source || run.model_id;
  const modelRevision = run.model_revision ? ` @ ${run.model_revision.slice(0, 12)}` : "";
  const runtimeRevision = run.runtime_revision ? ` ${run.runtime_revision}` : "";
  return `<div class="evidence-row"><div><strong>${escapeHtml(run.role)}</strong><span class="source-label">${escapeHtml(source + modelRevision)} · ${escapeHtml(run.runtime + runtimeRevision)}</span></div><span class="numeric-label"><strong>${run.latency_ms} ms</strong><button class="info-tip" data-tip="End-to-end latency recorded for this model run." aria-label="Explain model latency">?</button></span><p>Status: ${escapeHtml(run.status)}${escapeHtml(ttft)}${run.detail ? ` · ${escapeHtml(run.detail)}` : ""}</p></div>`;
}

function renderRegions() {
  if (!state.result) return;
  const regions = [...state.regions.values()];
  $("#subpanel-regions").innerHTML = regions.length ? `
    ${regions.map((region) => `
      <div class="region-row ${region.removed ? "removed" : ""}" data-region="${region.id}">
        <div class="region-main">
          <div><strong>${escapeHtml(region.label)}</strong><span class="source-label">${escapeHtml(region.source)} · ${region.image_id === state.activeImageId ? "active image" : "other image"}</span></div>
          <div class="region-actions">
            <button class="icon-button" data-region-action="focus" title="Focus region"><i data-lucide="focus"></i></button>
            <button class="icon-button" data-region-action="visibility" title="${region.visible ? "Hide" : "Show"} region"><i data-lucide="${region.visible ? "eye" : "eye-off"}"></i></button>
            <button class="icon-button" data-region-action="remove" title="Remove region"><i data-lucide="x"></i></button>
          </div>
        </div>
        <label class="region-opacity">Opacity <input type="range" min="10" max="100" value="${Math.round(region.opacity * 100)}" data-region-opacity></label>
      </div>
    `).join("")}
    <button class="secondary-action restore-action" id="restore-regions"><i data-lucide="history"></i>Restore removed regions</button>
  ` : '<p class="utility-note">No valid localization regions were accepted for this run.</p>';
  $$("[data-region]").forEach((row) => {
    const region = state.regions.get(row.dataset.region);
    row.querySelectorAll("[data-region-action]").forEach((button) => button.addEventListener("click", () => regionAction(region, button.dataset.regionAction)));
    row.querySelector("[data-region-opacity]")?.addEventListener("input", (event) => {
      region.opacity = Number(event.target.value) / 100;
      renderAllViewports();
    });
  });
  $("#restore-regions")?.addEventListener("click", () => {
    state.regions.forEach((region) => { region.removed = false; region.visible = true; });
    renderRegions(); renderAllViewports();
  });
  refreshIcons();
}

async function regionAction(region, action) {
  if (action === "visibility") region.visible = !region.visible;
  if (action === "remove") region.removed = true;
  if (action === "focus") {
    state.activeImageId = region.image_id || state.activeImageId;
    const view = activeViewState();
    view.zoom = 1.65;
    view.panX = 0; view.panY = 0;
    state.regions.forEach((item) => item.selected = item.id === region.id);
    renderStudyList();
    await loadVisibleImages();
    updateViewerControls();
  }
  renderRegions();
  renderAllViewports();
}

function renderChat() {
  const messages = state.session?.messages || [];
  $("#chat-messages").innerHTML = messages.length
    ? messages.map((message) => messageHtml(message)).join("")
    : '<div class="chat-empty">Ask the professor about this study.</div>';
  $("#chat-messages").scrollTop = $("#chat-messages").scrollHeight;
}

function messageHtml(message, streaming = false) {
  const run = message.model_run;
  const metrics = run ? `
    <span class="numeric-label">${run.latency_ms} ms<button class="info-tip" data-tip="End-to-end response latency." aria-label="Explain response latency">?</button></span>
    ${run.time_to_first_token_ms == null ? "" : `<span class="numeric-label">TTFT ${run.time_to_first_token_ms} ms<button class="info-tip" data-tip="Time from request start until the first generated token." aria-label="Explain time to first token">?</button></span>`}
    ${run.tokens_per_second == null ? "" : `<span class="numeric-label">${run.tokens_per_second.toFixed(1)} tok/s<button class="info-tip" data-tip="Average generated tokens per second for this response." aria-label="Explain token throughput">?</button></span>`}
  ` : "";
  const chips = (message.evidence_sources || []).map((item) => `<span class="evidence-chip">${escapeHtml(item)}</span>`).join("");
  return `<div class="message ${message.role}" ${streaming ? 'id="streaming-message"' : ""}>
    <div class="message-role"><span>${message.role === "assistant" ? "Professor" : "You"}</span><span>${streaming ? "Generating" : ""}</span></div>
    <div class="message-body">${formatMessageContent(message.content)}</div>
    <div class="message-meta">${metrics}${chips}</div>
  </div>`;
}

function formatMessageContent(value) {
  const inline = (text) => escapeHtml(text).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  return String(value || "").split(/\r?\n/).map((line) => {
    const heading = line.match(/^#{1,3}\s+(.+)$/);
    if (heading) return `<strong class="message-section-title">${inline(heading[1])}</strong>`;
    const bullet = line.match(/^[-*]\s+(.+)$/);
    if (bullet) return `<span class="message-list-item">${inline(bullet[1])}</span>`;
    const numbered = line.match(/^\d+\.\s+(.+)$/);
    if (numbered) return `<span class="message-list-item numbered">${inline(numbered[1])}</span>`;
    return line ? `<span class="message-line">${inline(line)}</span>` : '<span class="message-spacer"></span>';
  }).join("");
}

async function sendChat() {
  const input = $("#chat-input");
  const question = input.value.trim();
  if (!question || !state.session?.result || $("#chat-send").disabled) return;
  input.value = "";
  $("#chat-send").disabled = true;
  $("#professor-state").textContent = "Generating";
  state.session.messages.push({ role: "user", content: question, evidence_sources: [] });
  renderChat();
  $("#chat-messages").insertAdjacentHTML("beforeend", messageHtml({ role: "assistant", content: "", evidence_sources: [] }, true));
  try {
    await streamSse(`/api/sessions/${state.session.id}/chat`, { message: question }, (event) => {
      if (event.type === "delta") {
        const body = $("#streaming-message .message-body");
        body.textContent += event.content;
        $("#chat-messages").scrollTop = $("#chat-messages").scrollHeight;
      }
      if (event.type === "error") throw new Error(event.message);
      if (event.type === "complete") {
        $("#streaming-message")?.remove();
        state.session.messages.push(event.message);
        renderChat();
      }
    });
  } catch (error) {
    $("#streaming-message")?.remove();
    showError(error.message);
  } finally {
    $("#chat-send").disabled = false;
    $("#professor-state").textContent = "Grounded in this session";
  }
}

async function streamSse(url, payload, onEvent) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.detail || `Request failed (${response.status})`);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const dataLine = block.split("\n").find((line) => line.startsWith("data:"));
      if (!dataLine) continue;
      const event = JSON.parse(dataLine.slice(5).trim());
      onEvent(event);
    }
    if (done) break;
  }
}

function activatePanel(name) {
  $$(".inspector-tab").forEach((button) => button.classList.toggle("active", button.dataset.panel === name));
  $$(".inspector-panel").forEach((panel) => panel.classList.toggle("active", panel.id === `panel-${name}`));
}

function activateSubpanel(name) {
  $$(".subtab").forEach((button) => button.classList.toggle("active", button.dataset.subpanel === name));
  $$(".subpanel").forEach((panel) => panel.classList.toggle("active", panel.id === `subpanel-${name}`));
}

function updateCommitState() {
  $("#commit-read").disabled = !state.session || !$("#blind-read").value.trim();
}

function openMetadata() {
  const image = imageById(state.activeImageId);
  if (!image) return;
  $("#metadata-content").innerHTML = Object.entries(image.metadata)
    .filter(([, value]) => value !== "" && value !== null && value !== 0)
    .map(([key, value]) => `<div class="metadata-row"><span>${escapeHtml(titleCase(key))}</span><strong>${escapeHtml(Array.isArray(value) ? value.join(" × ") : value)}</strong></div>`)
    .join("") || '<p class="utility-note">No metadata available.</p>';
  $("#metadata-dialog").showModal();
}

function openOnboarding() {
  state.onboardingStep = 0;
  renderOnboarding();
  $("#onboarding-dialog").showModal();
}

function closeOnboarding() {
  localStorage.setItem("radiology-onboarding-complete", "true");
  $("#onboarding-dialog").close();
}

function advanceOnboarding() {
  const steps = $$(".walkthrough-step");
  if (state.onboardingStep >= steps.length - 1) return closeOnboarding();
  state.onboardingStep += 1;
  renderOnboarding();
}

function renderOnboarding() {
  const steps = $$(".walkthrough-step");
  steps.forEach((step, index) => step.classList.toggle("active", index === state.onboardingStep));
  $$(".step-dots i").forEach((dot, index) => dot.classList.toggle("active", index === state.onboardingStep));
  $("#walkthrough-next").textContent = state.onboardingStep === steps.length - 1 ? "Start" : "Next";
}

function applySavedTheme() {
  const saved = localStorage.getItem("radiology-theme") || "dark";
  applyTheme(saved);
}

function cycleTheme() {
  const order = ["dark", "light", "system"];
  const current = localStorage.getItem("radiology-theme") || "dark";
  const next = order[(order.indexOf(current) + 1) % order.length];
  localStorage.setItem("radiology-theme", next);
  applyTheme(next);
}

function applyTheme(theme) {
  const resolved = theme === "system"
    ? matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark"
    : theme;
  document.documentElement.dataset.theme = resolved;
  $("#theme-button")?.setAttribute("title", `Theme: ${theme}`);
}

function handleShortcut(event) {
  if (["TEXTAREA", "INPUT", "SELECT"].includes(event.target.tagName)) return;
  const key = event.key.toLowerCase();
  if (key === "f") fitView();
  if (key === "i") invertView();
  if (key === "r") setTool("ruler");
  if (key === "w") setTool("window");
  if (event.key === "Escape") setTool("pan");
  if (key === "+" || key === "=") changeZoom(0.15);
  if (key === "-") changeZoom(-0.15);
}

function exportMarkdown() {
  if (!state.result) return;
  const chat = (state.session.messages || []).map((message) => `**${message.role === "assistant" ? "Professor" : "Trainee"}**\n\n${message.content}`).join("\n\n");
  const regions = [...state.regions.values()].map((region) => `- ${region.label}: ${region.removed ? "removed" : region.visible ? "visible" : "hidden"} at ${Math.round(region.opacity * 100)}% opacity`).join("\n");
  download("radiology-training-session.md", `${state.result.session_note}\n\n## Professor conversation\n\n${chat}\n\n## Region state\n\n${regions}\n`, "text/markdown");
}

function exportJson() {
  if (!state.result) return;
  download("radiology-training-session.json", JSON.stringify({
    session: state.session,
    region_state: [...state.regions.values()],
    feedback: currentFeedback(),
    runtime: state.runtime,
  }, null, 2), "application/json");
}

function setRating(rating) {
  state.rating = rating;
  $$("#rating button").forEach((button) => button.classList.toggle("active", Number(button.dataset.rating) === rating));
}

async function submitFeedback() {
  if (!state.rating) return showError("Select a usefulness rating first.");
  try {
    await api("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(currentFeedback()),
    });
    $("#submit-feedback").textContent = "Saved to session";
  } catch (error) {
    showError(error.message);
  }
}

function currentFeedback() {
  return {
    session_id: state.session?.id || null,
    case_id: state.session?.study.case_id || null,
    rating: state.rating || 1,
    comment: $("#feedback-comment").value.trim(),
  };
}

function markCaseComplete(caseId) {
  const completed = completedCases();
  if (!completed.includes(caseId)) completed.push(caseId);
  localStorage.setItem("radiology-completed-cases", JSON.stringify(completed));
  renderCases();
  updateProgress();
}

function completedCases() {
  try { return JSON.parse(localStorage.getItem("radiology-completed-cases") || "[]"); }
  catch { return []; }
}

function updateProgress() {
  $("#progress-count").textContent = `${completedCases().length} / 3`;
}

function setBusy(busy, label = "") {
  $("#runtime-state span").textContent = busy
    ? label
    : state.runtime?.runtime_status === "ready"
      ? "Models ready"
      : state.runtime?.runtime_status === "demo"
        ? "Practice mode"
        : state.runtime?.runtime_status === "on-demand"
          ? "GPU on demand"
          : "Checking";
}

function showError(message) {
  const banner = $("#error-banner");
  banner.textContent = message;
  banner.classList.remove("hidden");
}

function hideError() { $("#error-banner").classList.add("hidden"); }

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
  return data;
}

function download(filename, content, type) {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([content], { type }));
  link.download = filename;
  link.click();
  URL.revokeObjectURL(link.href);
}

function refreshIcons() {
  if (window.lucide) window.lucide.createIcons();
}

function formatNumber(value) {
  return Math.abs(value) >= 100 ? Math.round(value).toString() : Number(value).toFixed(1);
}

function titleCase(value) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;",
  })[char]);
}
