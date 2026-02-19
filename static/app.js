const promptInput = document.getElementById("prompt");
const generateBtn = document.getElementById("generateBtn");
const progressCard = document.getElementById("progressCard");
const storyCard = document.getElementById("storyCard");
const progressBar = document.getElementById("progressBar");
const statusText = document.getElementById("statusText");
const statusList = document.getElementById("statusList");
const storyTitle = document.getElementById("storyTitle");
const storyHint = document.getElementById("storyHint");
const frameTitle = document.getElementById("frameTitle");
const storyText = document.getElementById("storyText");
const slideImage = document.getElementById("slideImage");
const slideCaption = document.getElementById("slideCaption");
const slideDots = document.getElementById("slideDots");
const prevFrameBtn = document.getElementById("prevFrameBtn");
const nextFrameBtn = document.getElementById("nextFrameBtn");
const playBtn = document.getElementById("playBtn");
const storyAudio = document.getElementById("storyAudio");

let pollHandle = null;
let currentAudioUrl = null;
let storyFrames = [];
let currentFrameIndex = 0;
let lastProgressMarker = "";
let lastProgressPulse = 0;
let stageStartedAt = 0;

const PROGRESS_PULSE_MS = 5000;

function setBusy(isBusy) {
  generateBtn.disabled = isBusy;
  generateBtn.textContent = isBusy ? "Generating..." : "Generate Story";
}

function formatStageLine(status, stageElapsedSeconds = 0) {
  const stage = status.stage || "Preparing";
  const message = status.message || "Waiting";
  const percent = Number.isFinite(status.progress) ? `${status.progress}%` : "";
  const elapsed = stageElapsedSeconds > 0 ? ` (${stageElapsedSeconds}s)` : "";
  return `${stage}: ${message}${percent ? ` (${percent})` : ""}${elapsed}`;
}

function appendStatusLine(line) {
  const li = document.createElement("li");
  li.textContent = `${new Date().toLocaleTimeString()} — ${line}`;
  statusList.appendChild(li);
  statusList.scrollTop = statusList.scrollHeight;
}

function updateProgress(status) {
  const percent = Math.max(0, Math.min(100, Number(status.progress) || 0));
  progressBar.style.width = `${percent}%`;

  const now = Date.now();
  const marker = `${status.stage || ""}|${status.message || ""}|${status.progress || 0}`;
  const stageElapsed = stageStartedAt ? Math.floor((now - stageStartedAt) / 1000) : 0;

  if (marker !== lastProgressMarker) {
    stageStartedAt = now;
    lastProgressMarker = marker;
    lastProgressPulse = now;
    const line = formatStageLine(status, 0);
    statusText.textContent = line;
    appendStatusLine(line);
    return;
  }

  statusText.textContent = formatStageLine(status, stageElapsed);

  if (status.status === "running" && status.progress < 100 && now - lastProgressPulse >= PROGRESS_PULSE_MS) {
    appendStatusLine(formatStageLine(status, stageElapsed));
    lastProgressPulse = now;
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderMarkdown(text) {
  const raw = String(text || "").trim();
  if (!raw) {
    return "<p>No scene text available.</p>";
  }

  let safe = escapeHtml(raw);

  safe = safe.replace(/^######\s+(.*)$/gm, "<h6>$1</h6>");
  safe = safe.replace(/^#####\s+(.*)$/gm, "<h5>$1</h5>");
  safe = safe.replace(/^####\s+(.*)$/gm, "<h4>$1</h4>");
  safe = safe.replace(/^###\s+(.*)$/gm, "<h3>$1</h3>");
  safe = safe.replace(/^##\s+(.*)$/gm, "<h2>$1</h2>");
  safe = safe.replace(/^#\s+(.*)$/gm, "<h1>$1</h1>");

  safe = safe.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  safe = safe.replace(/(?<!\*)\*([^*\n]+?)\*(?!\*)/g, "<em>$1</em>");
  safe = safe.replace(/`([^`]+)`/g, "<code>$1</code>");

  safe = safe.replace(/(?:^\s*[-*+]\s+.+(?:\n|$))+?/gm, (block) => {
    const lines = block
      .trim()
      .split("\n")
      .map((line) => `<li>${line.replace(/^\s*[-*+]\s+/, "").trim()}</li>`)
      .join("");
    return `<ul>${lines}</ul>`;
  });

  return safe
    .split(/\n{2,}/)
    .map((para) => `<p>${para.replaceAll("\n", "<br>")}</p>`)
    .join("");
}

function parseJsonObjectText(raw) {
  if (typeof raw !== "string") {
    return null;
  }

  const trimmed = raw.trim();
  if (!trimmed || (trimmed[0] !== "{" && trimmed[0] !== "[")) {
    return null;
  }

  try {
    return JSON.parse(trimmed);
  } catch {
    return null;
  }
}

function normalizeStoryText(raw, scenes, fallbackStory = "") {
  if (typeof raw === "string" && raw.includes("{")) {
    const parsed = parseJsonObjectText(raw);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      if (typeof parsed.story === "string") {
        return parsed.story;
      }
      if (typeof parsed.text === "string") {
        return parsed.text;
      }
      if (typeof parsed.summary === "string") {
        return parsed.summary;
      }
    }
  }

  if (typeof raw === "string") {
    return raw;
  }

  if (Array.isArray(scenes)) {
    return scenes
      .map((scene) => (scene?.text || ""))
      .filter(Boolean)
      .join("\n\n");
  }

  return fallbackStory || "";
}

function buildFrames(payload) {
  const scenes = Array.isArray(payload.scenes) ? payload.scenes : [];
  const images = Array.isArray(payload.images) ? payload.images : [];
  const story = normalizeStoryText(payload.story, scenes, "");
  const fallbackLines = story
    .split(/\n{2,}/)
    .map((segment) => segment.trim())
    .filter(Boolean);

  const targetCount = Math.max(scenes.length, images.length, fallbackLines.length, 1);
  const maxFrames = Math.min(10, targetCount);

  const frames = [];
  for (let i = 0; i < maxFrames; i += 1) {
    const scene = scenes[i] || {};
    const image = images[i] || null;
    const sceneText =
      typeof scene.text === "string" && scene.text.trim().length > 0
        ? scene.text
        : fallbackLines[i] || "";

    frames.push({
      title: scene.text ? `Scene ${i + 1}` : `Scene ${i + 1}`,
      text: sceneText || fallbackLines[0] || "",
      caption: scene.imagePrompt || image?.caption || `Scene ${i + 1}`,
      image: image,
    });
  }

  return frames;
}

function setActiveFrame(nextIndex) {
  if (!storyFrames.length) {
    return;
  }

  const clamped = Math.max(0, Math.min(storyFrames.length - 1, nextIndex));
  currentFrameIndex = clamped;

  const frame = storyFrames[currentFrameIndex];
  const hasUsableImage = frame?.image && frame.image.b64 && !frame.image.error;

  frameTitle.textContent = frame.title || `Scene ${currentFrameIndex + 1}`;
  storyText.innerHTML = renderMarkdown(frame.text);

  if (hasUsableImage) {
    slideImage.src = `data:${frame.image.mimeType || "image/png"};base64,${frame.image.b64}`;
    slideImage.classList.remove("missing");
  } else {
    slideImage.removeAttribute("src");
    slideImage.classList.add("missing");
  }

  slideImage.alt = hasUsableImage ? frame.caption || "Story scene illustration" : "No image available for this scene";
  slideCaption.textContent = frame.caption || `Scene ${currentFrameIndex + 1}`;

  slideDots.querySelectorAll("button").forEach((button, index) => {
    button.classList.toggle("active", index === currentFrameIndex);
  });
}

function renderDots() {
  slideDots.innerHTML = "";

  storyFrames.forEach((_frame, index) => {
    const dot = document.createElement("button");
    dot.type = "button";
    dot.className = "dot";
    dot.setAttribute("aria-label", `Go to scene ${index + 1}`);
    dot.addEventListener("click", () => setActiveFrame(index));
    slideDots.appendChild(dot);
  });

  slideDots.hidden = storyFrames.length <= 1;
}

function syncFrameToAudio() {
  if (!storyFrames.length || !storyAudio.duration || Number.isNaN(storyAudio.duration)) {
    return;
  }

  const ratio = Math.min(1, Math.max(0, storyAudio.currentTime / storyAudio.duration));
  const targetIndex = Math.min(storyFrames.length - 1, Math.floor(ratio * storyFrames.length));
  if (targetIndex !== currentFrameIndex) {
    setActiveFrame(targetIndex);
  }
}

function showResult(payload = {}) {
  const title = payload.title || "Untitled Story";
  const scenes = Array.isArray(payload.scenes) ? payload.scenes : [];
  const story = payload.story || "";

  storyTitle.textContent = title;
  storyHint.textContent =
    scenes.length > 0
      ? `Generated with ${scenes.length} scene${scenes.length === 1 ? "" : "s"}.`
      : "Your generated scene.";

  storyFrames = buildFrames({ scenes, images: payload.images || [], story });
  currentFrameIndex = 0;
  renderFrame();
  playBtn.disabled = false;
  storyCard.hidden = false;
}

function renderFrame() {
  renderDots();
  setActiveFrame(currentFrameIndex);
}

function clearResult() {
  storyTitle.textContent = "";
  storyHint.textContent = "";
  storyText.textContent = "";
  frameTitle.textContent = "";
  slideImage.removeAttribute("src");
  slideImage.classList.remove("missing");
  slideCaption.textContent = "";
  slideDots.innerHTML = "";
  storyFrames = [];
  currentFrameIndex = 0;

  if (currentAudioUrl) {
    URL.revokeObjectURL(currentAudioUrl);
    currentAudioUrl = null;
  }

  storyAudio.removeAttribute("src");
  storyAudio.hidden = false;
  storyAudio.disabled = true;
  playBtn.disabled = true;
  playBtn.textContent = "Play Story";
}

function renderError(text) {
  statusText.textContent = text;
  storyTitle.textContent = "Story generation failed";
  storyHint.textContent = text;
  storyText.innerHTML = `<p class="error">${escapeHtml(text)}</p>`;
  storyCard.hidden = false;
  playBtn.disabled = true;
  storyAudio.removeAttribute("src");
}

function renderAudio(audioPayload) {
  if (!audioPayload || !audioPayload.b64) {
    storyAudio.removeAttribute("src");
    storyAudio.disabled = true;
    playBtn.disabled = true;
    return;
  }

  if (currentAudioUrl) {
    URL.revokeObjectURL(currentAudioUrl);
    currentAudioUrl = null;
  }

  const blob = new Blob(
    [Uint8Array.from(atob(audioPayload.b64), (char) => char.charCodeAt(0))],
    { type: audioPayload.mimeType || "audio/wav" },
  );
  currentAudioUrl = URL.createObjectURL(blob);
  storyAudio.src = currentAudioUrl;
  storyAudio.hidden = false;
  storyAudio.removeAttribute("disabled");

  playBtn.disabled = false;
}

function resetProgressState() {
  lastProgressMarker = "";
  stageStartedAt = 0;
  lastProgressPulse = 0;
  statusList.textContent = "";
  progressBar.style.width = "0%";
  statusText.textContent = "Queued for generation...";
}

async function pollStatus(jobId) {
  try {
    const response = await fetch(`/api/story/${encodeURIComponent(jobId)}`);
    if (!response.ok) {
      throw new Error(`Status check failed: ${response.status}`);
    }

    const status = await response.json();
    updateProgress(status);

    if (status.status === "completed") {
      updateProgress({ ...status, message: "Done" });
      showResult(status.result || {});
      renderAudio(status.result?.audio);
      setBusy(false);
      stopPolling();
      return;
    }

    if (status.status === "failed") {
      const failureMessage = status.result?.error || status.message || "Generation failed.";
      setBusy(false);
      renderError(failureMessage);
      stopPolling();
      return;
    }

    pollHandle = setTimeout(() => pollStatus(jobId), 1250);
  } catch (error) {
    setBusy(false);
    stopPolling();
    renderError(error.message || "Unable to fetch generation status.");
  }
}

function stopPolling() {
  if (pollHandle) {
    clearTimeout(pollHandle);
    pollHandle = null;
  }
}

generateBtn.addEventListener("click", async () => {
  const prompt = promptInput.value.trim();
  if (prompt.length < 10) {
    statusText.textContent = "Please provide a richer prompt (at least 10 characters).";
    progressCard.hidden = false;
    storyCard.hidden = true;
    return;
  }

  stopPolling();
  clearResult();
  storyCard.hidden = true;
  progressCard.hidden = false;
  resetProgressState();
  setBusy(true);

  try {
    const response = await fetch("/api/story", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    });

    if (!response.ok) {
      const err = await response.text();
      throw new Error(err || `Generation request failed (${response.status})`);
    }

    const job = await response.json();
    const startStatus = {
      status: "running",
      stage: "Queued",
      message: "Queued for generation.",
      progress: 0,
    };
    updateProgress(startStatus);
    pollStatus(job.jobId);
  } catch (error) {
    setBusy(false);
    renderError(error.message || "Could not start story generation.");
  }
});

prevFrameBtn.addEventListener("click", () => {
  if (!storyFrames.length) return;
  const next = currentFrameIndex === 0 ? storyFrames.length - 1 : currentFrameIndex - 1;
  setActiveFrame(next);
});

nextFrameBtn.addEventListener("click", () => {
  if (!storyFrames.length) return;
  const next = (currentFrameIndex + 1) % storyFrames.length;
  setActiveFrame(next);
});

playBtn.addEventListener("click", async () => {
  try {
    if (storyAudio.paused) {
      await storyAudio.play();
      playBtn.textContent = "Pause Story";
    } else {
      storyAudio.pause();
      playBtn.textContent = "Play Story";
    }
  } catch {
    renderError("Unable to play audio. Check your browser permissions.");
  }
});

storyAudio.addEventListener("ended", () => {
  playBtn.textContent = "Play Story";
});

storyAudio.addEventListener("pause", () => {
  if (!storyAudio.ended) {
    playBtn.textContent = "Play Story";
  }
});

storyAudio.addEventListener("timeupdate", syncFrameToAudio);

playBtn.disabled = true;
storyAudio.disabled = true;
storyAudio.hidden = false;
