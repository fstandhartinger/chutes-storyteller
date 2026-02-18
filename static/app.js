const promptInput = document.getElementById("prompt");
const generateBtn = document.getElementById("generateBtn");
const progressCard = document.getElementById("progressCard");
const storyCard = document.getElementById("storyCard");
const progressBar = document.getElementById("progressBar");
const statusText = document.getElementById("statusText");
const statusList = document.getElementById("statusList");
const storyTitle = document.getElementById("storyTitle");
const storyText = document.getElementById("storyText");
const images = document.getElementById("images");
const playBtn = document.getElementById("playBtn");
const storyAudio = document.getElementById("storyAudio");

let pollHandle = null;
let currentAudioUrl = null;

function setBusy(isBusy) {
  generateBtn.disabled = isBusy;
  generateBtn.textContent = isBusy ? "Generating..." : "Generate Story";
}

function formatStageLine(status) {
  const stage = status.stage || "Preparing";
  const message = status.message || "Waiting";
  const percent = Number.isFinite(status.progress) ? `${status.progress}%` : "";
  return `${stage}: ${message}${percent ? ` (${percent})` : ""}`;
}

function updateProgress(status) {
  const percent = Math.max(0, Math.min(100, Number(status.progress) || 0));
  progressBar.style.width = `${percent}%`;
  statusText.textContent = formatStageLine(status);

  const li = document.createElement("li");
  li.textContent = `${new Date().toLocaleTimeString()} — ${formatStageLine(status)}`;
  statusList.appendChild(li);
  statusList.scrollTop = statusList.scrollHeight;
}

function renderSceneGallery(imagesPayload) {
  images.innerHTML = "";

  if (!Array.isArray(imagesPayload) || imagesPayload.length === 0) {
    images.innerHTML = "<p class=\"hint\">No scenes were generated yet.</p>";
    return;
  }

  imagesPayload.forEach((entry) => {
    const figure = document.createElement("figure");
    figure.className = "figure";

    const img = document.createElement("img");
    const mime = entry.mimeType || "image/png";
    img.src = `data:${mime};base64,${entry.b64}`;
    img.alt = entry.caption || "Story illustration";
    img.loading = "lazy";

    const caption = document.createElement("figcaption");
    caption.className = "caption";
    caption.textContent = entry.caption || "Story scene";

    figure.appendChild(img);
    figure.appendChild(caption);
    images.appendChild(figure);
  });
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
    [Uint8Array.from(atob(audioPayload.b64), (c) => c.charCodeAt(0))],
    { type: audioPayload.mimeType || "audio/wav" },
  );
  currentAudioUrl = URL.createObjectURL(blob);
  storyAudio.src = currentAudioUrl;
  storyAudio.hidden = false;
  storyAudio.removeAttribute("disabled");
  playBtn.disabled = false;
}

function showResult(payload) {
  storyTitle.textContent = payload.title || "Untitled Story";
  storyText.textContent = payload.story || "";
  renderSceneGallery(payload.images || []);
  renderAudio(payload.audio);
  storyCard.hidden = false;
}

function clearResult() {
  storyTitle.textContent = "";
  storyText.textContent = "";
  images.innerHTML = "";
  if (currentAudioUrl) {
    URL.revokeObjectURL(currentAudioUrl);
    currentAudioUrl = null;
  }
  storyAudio.removeAttribute("src");
  storyAudio.hidden = false;
  storyAudio.disabled = true;
  playBtn.disabled = true;
}

function stopPolling() {
  if (pollHandle) {
    clearTimeout(pollHandle);
    pollHandle = null;
  }
}

function renderError(text) {
  statusText.textContent = text;
  storyTitle.textContent = "Story generation failed";
  storyText.textContent = text;
  storyCard.hidden = false;
  images.innerHTML = "<p class=\"hint\">No media available.</p>";
  playBtn.disabled = true;
  storyAudio.removeAttribute("src");
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
  progressBar.style.width = "0%";
  statusList.textContent = "";
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
    updateProgress(job);
    pollStatus(job.jobId);
  } catch (error) {
    setBusy(false);
    renderError(error.message || "Could not start story generation.");
  }
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

playBtn.disabled = true;
storyAudio.disabled = true;
storyAudio.hidden = false;
