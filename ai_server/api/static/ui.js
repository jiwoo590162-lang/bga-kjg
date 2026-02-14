const boardEl = document.getElementById("board");
const turnEl = document.getElementById("turn");
const messageEl = document.getElementById("message");
const movesEl = document.getElementById("moves");
const resetBtn = document.getElementById("reset");
const refreshBtn = document.getElementById("refresh");
const aiStageEl = document.getElementById("ai-stage");
const aiMeterEl = document.querySelector("#ai-meter span");
const aiPolicyEl = document.getElementById("ai-policy");
const aiValueEl = document.getElementById("ai-value");
const aiMctsEl = document.getElementById("ai-mcts");
const aiCandidatesEl = document.getElementById("ai-candidates");
const aiEvaluatedEl = document.getElementById("ai-evaluated");
const aiBestEl = document.getElementById("ai-best");
const trainStatusEl = document.getElementById("train-status");
const trainStepEl = document.getElementById("train-step");
const trainLossEl = document.getElementById("train-loss");
const trainStartBtn = document.getElementById("train-start");
const autoTrainBtn = document.getElementById("auto-train");
const targetLossInput = document.getElementById("target-loss");
const datasetCountEl = document.getElementById("dataset-count");
const saveDataBtn = document.getElementById("save-data");
const loadDataBtn = document.getElementById("load-data");
const traceListEl = document.getElementById("trace-list");
const trainTraceEl = document.getElementById("train-trace");
const datasetLatestEl = document.getElementById("dataset-latest");
const datasetChart = document.getElementById("dataset-chart");
const runSmartTrainBtn = document.getElementById("run-smart-train");
let traceCount = 0;
const datasetHistory = [];
let autoLoadAttempted = false;

let selected = null;
let state = null;
let legalTargets = [];

const pieceLabels = {
  rK: "將",
  rG: "士",
  rR: "車",
  rH: "馬",
  rE: "象",
  rC: "砲",
  rS: "兵",
  bK: "將",
  bG: "士",
  bR: "車",
  bH: "馬",
  bE: "象",
  bC: "砲",
  bS: "卒"
};

function coordToPos(y, x) {
  const files = "abcdefghi";
  return `${files[x]}${y + 1}`;
}

function renderBoard() {
  boardEl.innerHTML = "";
  const board = state.board;
  for (let y = 9; y >= 0; y--) {
    for (let x = 0; x < 9; x++) {
      const cell = document.createElement("div");
      cell.className = "cell";
      const pos = coordToPos(y, x);
      cell.dataset.pos = pos;
      if (selected === pos) {
        cell.classList.add("selected");
      }
      if (legalTargets.includes(pos)) {
        cell.classList.add("legal");
      }

      const piece = board[y][x];
      if (piece) {
        const span = document.createElement("span");
        span.className = `piece ${piece[0] === "r" ? "red" : "blue"}`;
        span.textContent = pieceLabels[piece] || piece;
        cell.appendChild(span);
      }

      cell.addEventListener("click", () => handleClick(pos, piece));
      boardEl.appendChild(cell);
    }
  }
}

function setMessage(text, isError = false) {
  messageEl.textContent = text;
  messageEl.style.color = isError ? "#c0392b" : "#7b5c35";
}

async function loadState() {
  const res = await fetch("/state");
  state = await res.json();
  turnEl.textContent = `Turn: ${state.turn}`;
  renderBoard();
}

async function loadLegal(fromPos) {
  const res = await fetch(`/legal?from_pos=${fromPos}`);
  if (!res.ok) {
    return [];
  }
  const data = await res.json();
  return data.moves || [];
}

async function makeMove(fromPos, toPos) {
  startAiThinking();
  const res = await fetch("/move", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ from_pos: fromPos, to_pos: toPos })
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Move failed");
  }
  const data = await res.json();
  movesEl.prepend(
    createMoveLine(`Human: ${data.human_move} | AI: ${data.ai_move || "-"}`)
  );
  if (data.mcts) {
    aiCandidatesEl.textContent = data.mcts.candidates ?? "-";
    aiEvaluatedEl.textContent = data.mcts.evaluated ?? "-";
    aiBestEl.textContent = data.mcts.best_score ?? "-";
    aiMctsEl.textContent = "탐색 완료";
  }
  if (data.ai_eval !== undefined && data.ai_eval !== null) {
    aiValueEl.textContent = `평가 점수: ${data.ai_eval}`;
  }
  await loadState();
  finishAiThinking();
}

function startAiThinking() {
  aiStageEl.textContent = "Thinking...";
  aiMeterEl.style.width = "40%";
  aiPolicyEl.textContent = "정책망 후보 생성";
  aiValueEl.textContent = "가치 평가 중";
  aiMctsEl.textContent = "시뮬레이션 진행";
}

function finishAiThinking() {
  aiStageEl.textContent = "Done";
  aiMeterEl.style.width = "100%";
  aiPolicyEl.textContent = "후보 수 확정";
  aiValueEl.textContent = "가치 계산 완료";
  aiMctsEl.textContent = "최종 수 선택";
  setTimeout(() => {
    aiStageEl.textContent = "Idle";
    aiMeterEl.style.width = "0%";
  }, 800);
}

async function refreshAiStatus() {
  const res = await fetch("/ai_status_light");
  if (!res.ok) {
    return;
  }
  const data = await res.json();
  if (data.mcts) {
    aiCandidatesEl.textContent = data.mcts.candidates ?? "-";
    aiEvaluatedEl.textContent = data.mcts.evaluated ?? "-";
    aiBestEl.textContent = data.mcts.best_score ?? "-";
  }
  if (data.train) {
    trainStatusEl.textContent = data.train.status ?? "-";
    trainStepEl.textContent = `${data.train.step ?? 0}/${data.train.max_steps ?? 0}`;
    trainLossEl.textContent = data.train.loss ?? "-";
    if (autoTrainBtn) {
      autoTrainBtn.textContent = `Auto Train: ${data.train.auto ? "On" : "Off"}`;
    }
    if (datasetCountEl) {
      datasetCountEl.textContent = data.train.dataset_size ?? "-";
    }
    if (datasetLatestEl) {
      datasetLatestEl.textContent = data.train.dataset_size ?? "-";
    }
    if (datasetChart && data.train.dataset_size !== undefined && data.train.dataset_size !== null) {
      datasetHistory.push(Number(data.train.dataset_size));
      if (datasetHistory.length > 60) {
        datasetHistory.shift();
      }
      drawDatasetChart();
    }
    if (!autoLoadAttempted && data.train.dataset_size === 0) {
      autoLoadAttempted = true;
      fetch("/train/load?path=ai_server/neuralnet/data.pt", { method: "POST" }).catch(() => {});
    }
    if (targetLossInput && data.train.target_loss !== undefined && data.train.target_loss !== null) {
      targetLossInput.value = data.train.target_loss;
    }
  }
}

async function refreshTrace() {
  const res = await fetch("/ai_status?light=false");
  if (!res.ok) {
    return;
  }
  const data = await res.json();
  if (Array.isArray(data.trace)) {
    if (data.trace.length < traceCount) {
      traceCount = 0;
      traceListEl.innerHTML = "";
    }
    const newLines = data.trace.slice(traceCount);
    if (newLines.length) {
      traceListEl.insertAdjacentHTML(
        "beforeend",
        newLines.map((line) => `<div>${line}</div>`).join("")
      );
      traceCount = data.trace.length;
      traceListEl.scrollTop = traceListEl.scrollHeight;
    }
  }
  if (Array.isArray(data.train_trace)) {
    trainTraceEl.innerHTML = data.train_trace.map((line) => `<div>${line}</div>`).join("");
  }
}

trainStartBtn.addEventListener("click", async () => {
  await fetch("/train/start", { method: "POST" });
  await refreshAiStatus();
});

autoTrainBtn.addEventListener("click", async () => {
  const turnOn = autoTrainBtn.textContent.includes("Off");
  let url = turnOn ? "/train/auto/start" : "/train/auto/stop";
  if (turnOn && targetLossInput && targetLossInput.value) {
    url = `/train/auto/start?target_loss=${encodeURIComponent(targetLossInput.value)}`;
  }
  await fetch(url, { method: "POST" });
  await refreshAiStatus();
});

saveDataBtn.addEventListener("click", async () => {
  await fetch("/train/save", { method: "POST" });
  await refreshAiStatus();
});

loadDataBtn.addEventListener("click", async () => {
  await fetch("/train/load", { method: "POST" });
  await refreshAiStatus();
});

runSmartTrainBtn.addEventListener("click", async () => {
  setMessage("Smart Train started...");
  await fetch("/train/kif/load?path=ai_server/neuralnet/kif_example.txt", { method: "POST" });
  await fetch("/records/generate?count=1000&max_moves=60&sims=5&path=ai_server/neuralnet/records_fast.json&auto_load=true&auto_train_steps=800", { method: "POST" });
  await refreshAiStatus();
  setMessage("Smart Train finished (generation started).");
});

function createMoveLine(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div;
}

function handleClick(pos, piece) {
  if (!selected) {
    if (!piece) {
      setMessage("Select a piece first.");
      return;
    }
    selected = pos;
    loadLegal(pos).then((moves) => {
      legalTargets = moves;
      renderBoard();
      setMessage(`Selected ${pos}`);
    });
    return;
  }

  const fromPos = selected;
  const toPos = pos;
  selected = null;
  legalTargets = [];
  renderBoard();
  setMessage(`Trying ${fromPos} -> ${toPos}`);
  makeMove(fromPos, toPos).catch((err) => {
    setMessage(err.message, true);
  });
}

resetBtn.addEventListener("click", async () => {
  await fetch("/reset", { method: "POST" });
  movesEl.innerHTML = "";
  selected = null;
  legalTargets = [];
  await loadState();
  setMessage("Board reset.");
});

refreshBtn.addEventListener("click", async () => {
  selected = null;
  legalTargets = [];
  await loadState();
  setMessage("State refreshed.");
});

loadState().catch(() => setMessage("Failed to load state.", true));
setInterval(refreshAiStatus, 1000);
setInterval(refreshTrace, 3000);

function drawDatasetChart() {
  if (!datasetChart) return;
  const ctx = datasetChart.getContext("2d");
  const w = datasetChart.width;
  const h = datasetChart.height;
  ctx.clearRect(0, 0, w, h);
  if (datasetHistory.length < 2) return;

  const min = Math.min(...datasetHistory);
  const max = Math.max(...datasetHistory);
  const range = Math.max(1, max - min);

  ctx.strokeStyle = "rgba(155, 106, 47, 0.6)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  datasetHistory.forEach((v, i) => {
    const x = (i / (datasetHistory.length - 1)) * (w - 10) + 5;
    const y = h - ((v - min) / range) * (h - 10) - 5;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}
