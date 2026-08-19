/**
 * L.U.N.A.I.T.E. 27B — J.A.R.V.I.S. Tactical Assistant Logic
 * Full System Immersion, Live Knowledge Graph, Deep PC Access
 * Created by Swasthik Shetty
 */

// ─── Application State ───────────────────────────────────────────────────────
const state = {
  voiceActive: false,
  soundFxEnabled: true,
  defaultModel: "lunaite-ai",
  lastEntityId: null
};

// ─── Web Audio API Sound Synthesizer ─────────────────────────────────────────
const audioCtx = new (window.AudioContext || window.webkitAudioContext)();

function playHudSound(type = 'blip') {
  if (!state.soundFxEnabled || !audioCtx) return;
  try {
    if (audioCtx.state === 'suspended') {
      audioCtx.resume();
    }
    const now = audioCtx.currentTime;
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.connect(gain);
    gain.connect(audioCtx.destination);

    if (type === 'blip') {
      osc.type = 'sine';
      osc.frequency.setValueAtTime(880, now);
      osc.frequency.exponentialRampToValueAtTime(1760, now + 0.04);
      gain.gain.setValueAtTime(0.05, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.04);
      osc.start(now);
      osc.stop(now + 0.04);
    } else if (type === 'engage') {
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(440, now);
      osc.frequency.exponentialRampToValueAtTime(1100, now + 0.15);
      gain.gain.setValueAtTime(0.08, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.15);
      osc.start(now);
      osc.stop(now + 0.15);
    } else if (type === 'transmit') {
      osc.type = 'sine';
      osc.frequency.setValueAtTime(1200, now);
      osc.frequency.exponentialRampToValueAtTime(600, now + 0.08);
      gain.gain.setValueAtTime(0.07, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.08);
      osc.start(now);
      osc.stop(now + 0.08);
    } else if (type === 'alert') {
      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(520, now);
      osc.frequency.setValueAtTime(780, now + 0.08);
      gain.gain.setValueAtTime(0.06, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.16);
      osc.start(now);
      osc.stop(now + 0.16);
    }
  } catch (e) {}
}

// ─── DOM References ─────────────────────────────────────────────────────────
const dom = {
  hudClock: document.getElementById('hudClock'),
  hudStatus: document.getElementById('hudStatus'),
  jarvisAlert: document.getElementById('jarvisAlert'),
  
  // Arc Reactor & Visualizer
  arcReactorCore: document.getElementById('arcReactorCore'),
  hudCoreState: document.getElementById('hudCoreState'),
  hudCoreSub: document.getElementById('hudCoreSub'),
  hologramSpectrum: document.getElementById('hologramSpectrum'),
  
  // Telemetry Bars
  barCpu: document.getElementById('barCpu'),
  valCpu: document.getElementById('valCpu'),
  barRam: document.getElementById('barRam'),
  valRam: document.getElementById('valRam'),
  barDisk: document.getElementById('barDisk'),
  valDisk: document.getElementById('valDisk'),
  barBat: document.getElementById('barBat'),
  valBat: document.getElementById('valBat'),
  telUptime: document.getElementById('telUptime'),
  telProcs: document.getElementById('telProcs'),
  telFreq: document.getElementById('telFreq'),

  // Voice Controls
  btnToggleVoice: document.getElementById('btnToggleVoice'),
  voiceBtnText: document.getElementById('voiceBtnText'),
  btnMicInput: document.getElementById('btnMicInput'),
  voiceStatusBadge: document.getElementById('voiceStatusBadge'),
  voiceWaveContainer: document.getElementById('voiceWaveContainer'),
  voiceWaveLabel: document.getElementById('voiceWaveLabel'),

  // Chat Terminal
  chatMessages: document.getElementById('chatMessages'),
  chatInput: document.getElementById('chatInput'),
  btnSendChat: document.getElementById('btnSendChat'),
  btnClearChat: document.getElementById('btnClearChat'),

  // PowerShell REPL
  psReplDrawer: document.getElementById('psReplDrawer'),
  psReplInput: document.getElementById('psReplInput'),

  // Knowledge Graph
  kgSvg: document.getElementById('kgSvg'),
  kgNodeCount: document.getElementById('kgNodeCount'),
  kgEmptyMsg: document.getElementById('kgEmptyMsg'),
  kgTooltip: document.getElementById('kgTooltip')
};

// ─── Initialization ─────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  setupHudClock();
  setupVoiceMode();
  setupChat();
  setupArcReactorClick();
  startTelemetryLoop();
  initKnowledgeGraph();
  loadInitialKnowledgeGraph();
});

// ─── System Clock ───────────────────────────────────────────────────────────
function setupHudClock() {
  const update = () => {
    const now = new Date();
    const pad = (n) => String(n).padStart(2, '0');
    if (dom.hudClock) {
      dom.hudClock.textContent = `${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;
    }
  };
  update();
  setInterval(update, 1000);
}

// ─── Arc Reactor Interactive Click ──────────────────────────────────────────
function setupArcReactorClick() {
  if (!dom.arcReactorCore) return;
  dom.arcReactorCore.addEventListener('click', () => {
    playHudSound('engage');
    if (dom.btnToggleVoice) dom.btnToggleVoice.click();
  });
}

// ─── Two-Way Voice Engine ───────────────────────────────────────────────────
let recognition = null;
const synth = window.speechSynthesis;

function setupVoiceMode() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
      setCoreState('LISTENING', 'Receiving your voice stream...');
      if (dom.btnMicInput) dom.btnMicInput.classList.add('active');
      if (dom.voiceStatusBadge) dom.voiceStatusBadge.style.display = 'inline-block';
      if (dom.voiceWaveContainer) dom.voiceWaveContainer.style.display = 'flex';
      if (dom.hologramSpectrum) dom.hologramSpectrum.classList.add('active');
    };

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      if (transcript && transcript.trim()) {
        dom.chatInput.value = transcript;
        sendChatMessage();
      }
    };

    recognition.onerror = () => {
      resetVoiceUI();
    };

    recognition.onend = () => {
      resetVoiceUI();
    };
  }

  if (dom.btnMicInput) {
    dom.btnMicInput.addEventListener('click', () => {
      playHudSound('engage');
      if (!recognition) {
        alert("Microphone recognition is not supported in this browser. Please use Chrome or Edge!");
        return;
      }
      try {
        recognition.start();
      } catch (e) {
        recognition.stop();
      }
    });
  }

  if (dom.btnToggleVoice) {
    dom.btnToggleVoice.addEventListener('click', () => {
      state.voiceActive = !state.voiceActive;
      playHudSound('engage');
      
      if (state.voiceActive) {
        dom.btnToggleVoice.classList.add('active');
        if (dom.voiceBtnText) dom.voiceBtnText.textContent = "VOICE ACTIVE // LISTENING";
        speakAloud("J.A.R.V.I.S. neural systems online, Swasthik. I am listening.");
        if (recognition) {
          try { recognition.start(); } catch (e) {}
        }
      } else {
        dom.btnToggleVoice.classList.remove('active');
        if (dom.voiceBtnText) dom.voiceBtnText.textContent = "ENGAGE VOICE MODE";
        if (synth) synth.cancel();
        resetVoiceUI();
      }
    });
  }
}

function setCoreState(mode, subText) {
  if (dom.arcReactorCore) {
    dom.arcReactorCore.classList.remove('speaking', 'listening');
    if (mode === 'SPEAKING') dom.arcReactorCore.classList.add('speaking');
    if (mode === 'LISTENING') dom.arcReactorCore.classList.add('listening');
  }
  if (dom.hudCoreState) dom.hudCoreState.textContent = mode;
  if (dom.hudCoreSub && subText) dom.hudCoreSub.textContent = subText;
}

function resetVoiceUI() {
  setCoreState('SYSTEM READY', 'Click orb or mic to engage');
  if (dom.btnMicInput) dom.btnMicInput.classList.remove('active');
  if (dom.voiceStatusBadge) dom.voiceStatusBadge.style.display = 'none';
  if (dom.voiceWaveContainer) dom.voiceWaveContainer.style.display = 'none';
  if (dom.hologramSpectrum) dom.hologramSpectrum.classList.remove('active');
}

function speakAloud(text) {
  if (!synth) return;
  synth.cancel();

  // Strip code, markdown symbols, links
  const clean = text
    .replace(/```[\s\S]*?```/g, " I have executed the code block for you. ")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/[#*_~>\[\]]/g, "")
    .replace(/https?:\/\/[^\s]+/g, "web link")
    .trim();

  if (!clean) return;

  const utterance = new SpeechSynthesisUtterance(clean);
  utterance.rate = 1.05;
  utterance.pitch = 1.0;

  utterance.onstart = () => {
    setCoreState('SPEAKING', 'Transmitting audio response...');
    if (dom.hologramSpectrum) dom.hologramSpectrum.classList.add('active');
  };

  utterance.onend = () => {
    setCoreState('SYSTEM READY', 'Click orb or mic to engage');
    if (dom.hologramSpectrum) dom.hologramSpectrum.classList.remove('active');
    if (state.voiceActive && recognition) {
      setTimeout(() => {
        try { recognition.start(); } catch (e) {}
      }, 300);
    }
  };

  synth.speak(utterance);
}

// ─── Live System Telemetry Loop ─────────────────────────────────────────────
async function startTelemetryLoop() {
  const fetchTelemetry = async () => {
    try {
      const res = await fetch('/api/system/telemetry');
      if (!res.ok) return;
      const data = await res.json();

      // CPU
      if (dom.barCpu) dom.barCpu.style.width = `${Math.min(data.cpu_percent, 100)}%`;
      if (dom.valCpu) dom.valCpu.textContent = `${Math.round(data.cpu_percent)}%`;

      // RAM
      if (dom.barRam) dom.barRam.style.width = `${Math.min(data.ram_percent, 100)}%`;
      if (dom.valRam) dom.valRam.textContent = `${Math.round(data.ram_percent)}%`;

      // Disk
      if (dom.barDisk) dom.barDisk.style.width = `${Math.min(data.disk_percent, 100)}%`;
      if (dom.valDisk) dom.valDisk.textContent = `${Math.round(data.disk_percent)}%`;

      // Battery
      if (data.battery_percent !== null) {
        if (dom.barBat) dom.barBat.style.width = `${Math.min(data.battery_percent, 100)}%`;
        if (dom.valBat) {
          const chg = data.battery_charging ? " ⚡" : "";
          dom.valBat.textContent = `${Math.round(data.battery_percent)}%${chg}`;
        }
      } else {
        if (dom.valBat) dom.valBat.textContent = "AC";
        if (dom.barBat) dom.barBat.style.width = "100%";
      }

      // Details
      if (dom.telUptime && data.uptime_seconds) {
        const hrs = Math.floor(data.uptime_seconds / 3600);
        const mins = Math.floor((data.uptime_seconds % 3600) / 60);
        dom.telUptime.textContent = `UPTIME: ${hrs}h ${mins}m`;
      }
      if (dom.telProcs && data.process_count) {
        dom.telProcs.textContent = `PROCS: ${data.process_count}`;
      }
      if (dom.telFreq && data.cpu_freq_mhz) {
        dom.telFreq.textContent = `CPU: ${(data.cpu_freq_mhz / 1000).toFixed(2)} GHz`;
      }
      if (dom.hudStatus) {
        dom.hudStatus.textContent = "ONLINE";
        dom.hudStatus.style.color = "var(--cyan)";
      }
    } catch (e) {
      if (dom.hudStatus) {
        dom.hudStatus.textContent = "STANDBY";
        dom.hudStatus.style.color = "var(--rose)";
      }
    }
  };

  fetchTelemetry();
  setInterval(fetchTelemetry, 2500);
}

// ─── Tactical App Action Protocols ──────────────────────────────────────────
window.triggerAppAction = async function(actionStr) {
  playHudSound('transmit');
  const assistBubble = document.createElement('div');
  assistBubble.className = 'msg-box assistant';
  assistBubble.innerHTML = `
    <div class="msg-sender">PROTOCOL EXECUTING</div>
    <div class="msg-body">⚡ Dispatched: <code>${escapeHtml(actionStr)}</code>...</div>
  `;
  dom.chatMessages.appendChild(assistBubble);
  dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;

  try {
    const res = await fetch('/api/app/action', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action_str: actionStr })
    });
    const data = await res.json();
    if (data.status === 'success') {
      assistBubble.innerHTML = `
        <div class="msg-sender">PROTOCOL CONFIRMED</div>
        <div class="msg-body">${escapeHtml(data.result)}</div>
      `;
      if (state.voiceActive) speakAloud(data.result);
    } else {
      assistBubble.innerHTML = `
        <div class="msg-sender" style="color: var(--rose);">PROTOCOL ERROR</div>
        <div class="msg-body">${escapeHtml(data.message || "Unknown error")}</div>
      `;
    }
  } catch (e) {
    assistBubble.innerHTML = `
      <div class="msg-sender" style="color: var(--rose);">ERROR</div>
      <div class="msg-body">Connection error: ${escapeHtml(e.message)}</div>
    `;
  }
  dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
};

// ─── Deep PC Quick Actions ──────────────────────────────────────────────────
window.doScreenshot = async function() {
  playHudSound('transmit');
  setCoreState('PROCESSING', 'Capturing screen...');
  try {
    const res = await fetch('/api/system/screenshot', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) });
    const data = await res.json();
    if (data.thumb_b64) {
      const bubble = document.createElement('div');
      bubble.className = 'msg-box assistant';
      bubble.innerHTML = `
        <div class="msg-sender">📸 SCREEN CAPTURED</div>
        <div class="msg-body">Saved to: <code>${escapeHtml(data.path)}</code></div>
        <img class="msg-screenshot" src="data:image/png;base64,${data.thumb_b64}" alt="Screenshot Thumbnail" />
      `;
      dom.chatMessages.appendChild(bubble);
      dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
      if (state.voiceActive) speakAloud("Screenshot captured successfully.");
    } else {
      alert("Screenshot failed: " + (data.detail || data.error || "Unknown error"));
    }
  } catch (e) {
    alert("Screenshot error: " + e.message);
  }
  resetVoiceUI();
};

window.doClipboard = async function() {
  playHudSound('blip');
  try {
    const res = await fetch('/api/system/clipboard', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: "read" })
    });
    const data = await res.json();
    const bubble = document.createElement('div');
    bubble.className = 'msg-box assistant';
    bubble.innerHTML = `
      <div class="msg-sender">📋 CLIPBOARD CONTENTS</div>
      <div class="msg-body">${escapeHtml(data.result)}</div>
    `;
    dom.chatMessages.appendChild(bubble);
    dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
  } catch (e) {
    alert("Clipboard error: " + e.message);
  }
};

window.doListApps = async function() {
  triggerAppAction("system:list_apps");
};

window.openPSRepl = function() {
  playHudSound('engage');
  if (dom.psReplDrawer) {
    dom.psReplDrawer.style.display = 'flex';
    if (dom.psReplInput) dom.psReplInput.focus();
  }
};

window.closePSRepl = function() {
  if (dom.psReplDrawer) dom.psReplDrawer.style.display = 'none';
};

window.executePSRepl = async function() {
  const cmd = dom.psReplInput ? dom.psReplInput.value.trim() : "";
  if (!cmd) return;
  playHudSound('transmit');

  const bubble = document.createElement('div');
  bubble.className = 'msg-box user';
  bubble.innerHTML = `
    <div class="msg-sender">POWERSHELL EXECUTION</div>
    <div class="msg-body"><code>${escapeHtml(cmd)}</code></div>
  `;
  dom.chatMessages.appendChild(bubble);
  dom.psReplInput.value = "";
  dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;

  try {
    const res = await fetch('/api/system/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ command: cmd })
    });
    const data = await res.json();
    const assistBubble = document.createElement('div');
    assistBubble.className = 'msg-box assistant';
    assistBubble.innerHTML = `
      <div class="msg-sender">⚡ POWERSHELL OUTPUT</div>
      <div class="msg-body"><pre style="font-family:var(--font-mono);font-size:0.75rem;white-space:pre-wrap;">${escapeHtml(data.output)}</pre></div>
    `;
    dom.chatMessages.appendChild(assistBubble);
    dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
  } catch (e) {
    alert("PowerShell Error: " + e.message);
  }
};

window.openFileBrowser = function() {
  triggerAppAction("explorer:downloads");
};

// ─── Chat Messaging with Streaming & Knowledge Graph Trigger ────────────────
function setupChat() {
  if (dom.btnSendChat) dom.btnSendChat.addEventListener('click', sendChatMessage);
  if (dom.chatInput) {
    dom.chatInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') sendChatMessage();
    });
  }

  if (dom.psReplInput) {
    dom.psReplInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') executePSRepl();
    });
  }

  if (dom.btnClearChat) {
    dom.btnClearChat.addEventListener('click', () => {
      playHudSound('blip');
      dom.chatMessages.innerHTML = `
        <div class="msg-box assistant">
          <div class="msg-sender">L.U.N.A.I.T.E. 27B</div>
          <div class="msg-body">Console refreshed. What is your command, Swasthik?</div>
        </div>
      `;
      if (synth) synth.cancel();
    });
  }
}

async function sendChatMessage() {
  const text = dom.chatInput.value.trim();
  if (!text) return;

  playHudSound('transmit');

  // User bubble
  const userBubble = document.createElement('div');
  userBubble.className = 'msg-box user';
  userBubble.innerHTML = `
    <div class="msg-sender">SWASTHIK SHETTY</div>
    <div class="msg-body">${escapeHtml(text)}</div>
  `;
  dom.chatMessages.appendChild(userBubble);
  dom.chatInput.value = '';
  dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;

  // Assistant bubble with blinking cursor
  const assistBubble = document.createElement('div');
  assistBubble.className = 'msg-box assistant';
  assistBubble.innerHTML = `
    <div class="msg-sender">L.U.N.A.I.T.E. 27B</div>
    <div class="msg-body stream-body"><span class="stream-cursor">▋</span></div>
  `;
  dom.chatMessages.appendChild(assistBubble);
  dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;

  const bodyEl = assistBubble.querySelector('.stream-body');
  let accumulated = '';

  setCoreState('PROCESSING', 'Streaming response...');

  try {
    const res = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: state.defaultModel, prompt: text })
    });

    if (!res.ok || !res.body) {
      const errData = await res.json().catch(() => ({}));
      bodyEl.textContent = errData.detail || 'Unable to reach local Ollama inference engine.';
      assistBubble.querySelector('.msg-sender').style.color = 'var(--rose)';
      resetVoiceUI();
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // SSE lines: "data: {...}\n\n"
      const lines = buffer.split('\n');
      buffer = lines.pop(); // keep incomplete tail

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith('data:')) continue;
        const jsonStr = trimmed.slice(5).trim();
        if (!jsonStr) continue;

        try {
          const chunk = JSON.parse(jsonStr);
          if (chunk.error) {
            bodyEl.innerHTML = `<span style="color:var(--rose)">${escapeHtml(chunk.error)}</span>`;
            break;
          }
          if (chunk.token) {
            accumulated += chunk.token;
            bodyEl.innerHTML = escapeHtml(accumulated) + '<span class="stream-cursor">▋</span>';
            dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
          }
          if (chunk.done) {
            bodyEl.textContent = accumulated;
            if (state.voiceActive) speakAloud(accumulated);
            // Trigger automatic entity extraction into knowledge graph
            extractAndAddEntities(text, accumulated);
          }
        } catch (e) {}
      }
    }

    if (bodyEl.querySelector('.stream-cursor')) {
      bodyEl.textContent = accumulated;
      if (state.voiceActive) speakAloud(accumulated);
      extractAndAddEntities(text, accumulated);
    }

  } catch (e) {
    assistBubble.innerHTML = `
      <div class="msg-sender" style="color: var(--rose);">ERROR</div>
      <div class="msg-body">Connection error: ${escapeHtml(e.message)}</div>
    `;
  }

  resetVoiceUI();
  dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
}

// ─── D3.js Force-Directed Live Knowledge Graph Engine ───────────────────────
let kgSimulation = null;
let kgSvg = null;
let kgG = null;
let kgData = { nodes: [], links: [] };

function initKnowledgeGraph() {
  if (!dom.kgSvg || typeof d3 === 'undefined') return;

  const container = dom.kgSvg.parentElement;
  const width = container.clientWidth || 300;
  const height = container.clientHeight || 400;

  kgSvg = d3.select(dom.kgSvg)
    .attr("width", "100%")
    .attr("height", "100%")
    .attr("viewBox", [-width / 2, -height / 2, width, height]);

  kgSvg.selectAll("*").remove();

  // Zoom behavior
  const zoom = d3.zoom()
    .scaleExtent([0.2, 4])
    .on("zoom", (event) => {
      kgG.attr("transform", event.transform);
    });

  kgSvg.call(zoom);

  kgG = kgSvg.append("g");

  // Force simulation
  kgSimulation = d3.forceSimulation()
    .force("link", d3.forceLink().id(d => d.id).distance(55))
    .force("charge", d3.forceManyBody().strength(-120))
    .force("center", d3.forceCenter(0, 0))
    .force("collision", d3.forceCollide().radius(22));
}

async function loadInitialKnowledgeGraph() {
  try {
    const res = await fetch('/api/knowledge/graph');
    if (!res.ok) return;
    const data = await res.json();
    kgData = data;
    renderKnowledgeGraph();
  } catch (e) {}
}

function getNodeColor(type) {
  switch (type) {
    case 'person':  return 'var(--gold)';
    case 'concept': return 'var(--cyan)';
    case 'tool':    return 'var(--green)';
    case 'place':   return 'var(--purple)';
    case 'event':   return 'var(--orange)';
    default:        return 'var(--cyan)';
  }
}

function renderKnowledgeGraph() {
  if (!kgG || !kgSimulation || typeof d3 === 'undefined') return;

  const nodes = kgData.nodes || [];
  const links = kgData.links || [];

  if (dom.kgNodeCount) {
    dom.kgNodeCount.textContent = `${nodes.length} NODES`;
  }

  if (dom.kgEmptyMsg) {
    dom.kgEmptyMsg.style.display = nodes.length === 0 ? 'flex' : 'none';
  }

  // Links
  const link = kgG.selectAll(".kg-link")
    .data(links, d => `${d.source.id || d.source}-${d.target.id || d.target}`)
    .join("line")
    .attr("class", "kg-link");

  // Nodes
  const node = kgG.selectAll(".kg-node")
    .data(nodes, d => d.id)
    .join(
      enter => {
        const g = enter.append("g")
          .attr("class", "kg-node")
          .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended));

        g.append("circle")
          .attr("r", 9)
          .attr("fill", d => getNodeColor(d.type))
          .attr("stroke", "#010810")
          .attr("stroke-width", 1.5)
          .style("filter", "drop-shadow(0 0 6px rgba(0,220,255,0.4))");

        g.append("text")
          .attr("x", 12)
          .attr("y", 3)
          .text(d => d.label || d.id);

        g.on("click", (event, d) => {
          event.stopPropagation();
          playHudSound('blip');
          if (dom.chatInput) {
            dom.chatInput.value = `Tell me more about ${d.label}`;
            sendChatMessage();
          }
        });

        g.on("mouseover", (event, d) => {
          if (dom.kgTooltip) {
            dom.kgTooltip.style.display = 'block';
            dom.kgTooltip.innerHTML = `<strong>${escapeHtml(d.label)}</strong><br><span style="color:var(--text-m);font-size:0.6rem;">TYPE: ${(d.type || 'concept').toUpperCase()}</span><br><span style="font-size:0.6rem;color:var(--cyan);">Click to query Lunaite ›</span>`;
          }
        });

        g.on("mouseout", () => {
          if (dom.kgTooltip) dom.kgTooltip.style.display = 'none';
        });

        return g;
      },
      update => update,
      exit => exit.remove()
    );

  kgSimulation.nodes(nodes);
  kgSimulation.force("link").links(links);
  kgSimulation.alpha(0.8).restart();

  kgSimulation.on("tick", () => {
    link
      .attr("x1", d => d.source.x)
      .attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x)
      .attr("y2", d => d.target.y);

    node
      .attr("transform", d => `translate(${d.x},${d.y})`);
  });
}

function dragstarted(event, d) {
  if (!event.active) kgSimulation.alphaTarget(0.3).restart();
  d.fx = d.x;
  d.fy = d.y;
}

function dragged(event, d) {
  d.fx = event.x;
  d.fy = event.y;
}

function dragended(event, d) {
  if (!event.active) kgSimulation.alphaTarget(0);
  d.fx = null;
  d.fy = null;
}

window.fitGraphView = function() {
  if (!kgSvg || typeof d3 === 'undefined') return;
  kgSvg.transition().duration(750).call(
    d3.zoom().transform,
    d3.zoomIdentity
  );
};

window.clearKnowledgeGraph = async function() {
  playHudSound('blip');
  try {
    await fetch('/api/knowledge/clear', { method: 'DELETE' });
    kgData = { nodes: [], links: [] };
    renderKnowledgeGraph();
  } catch (e) {}
};

// ─── Automated Entity Extraction & Knowledge Node Generator ─────────────────
async function extractAndAddEntities(userPrompt, aiResponse) {
  const combined = `${userPrompt} ${aiResponse}`;

  // Predefined keyword clusters
  const clusters = [
    { pattern: /\b(Swasthik Shetty|Swasthik)\b/i, id: "swasthik_shetty", label: "Swasthik Shetty", type: "person" },
    { pattern: /\b(Lunaite AI|Lunaite 27B|Lunaite 10B|Lunaite)\b/i, id: "lunaite_ai", label: "Lunaite AI", type: "concept" },
    { pattern: /\b(Stark Industries|J\.A\.R\.V\.I\.S\.|JARVIS)\b/i, id: "jarvis_arch", label: "J.A.R.V.I.S. Arch", type: "concept" },
    { pattern: /\b(Spotify|Discord|Outlook|PowerShell|Explorer|Chrome|VS Code)\b/i, id: "tool_", dynamic: true, type: "tool" },
    { pattern: /\b(Neural Network|Transformer|MoE|Sparse MoE|Attention|Qwen|LoRA|Quantization)\b/i, id: "concept_", dynamic: true, type: "concept" },
    { pattern: /\b(Black Hole|Chirp Mass|Gravitational Wave|Astrophysics|General Relativity)\b/i, id: "phys_", dynamic: true, type: "concept" }
  ];

  let detectedEntities = [];

  for (const rule of clusters) {
    if (rule.dynamic) {
      const matches = combined.match(new RegExp(rule.pattern, 'gi'));
      if (matches) {
        for (const m of matches) {
          const clean = m.trim();
          const id = `${rule.id}${clean.toLowerCase().replace(/\s+/g, '_')}`;
          if (!detectedEntities.some(e => e.id === id)) {
            detectedEntities.push({ id, label: clean, type: rule.type });
          }
        }
      }
    } else {
      if (rule.pattern.test(combined)) {
        if (!detectedEntities.some(e => e.id === rule.id)) {
          detectedEntities.push({ id: rule.id, label: rule.label, type: rule.type });
        }
      }
    }
  }

  // Capitalized proper noun extractor (e.g. "Quantum Gravity", "London", "Nvidia RTX")
  const properNouns = aiResponse.match(/\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b/g);
  if (properNouns) {
    for (const noun of properNouns.slice(0, 4)) {
      const clean = noun.trim();
      const lower = clean.toLowerCase();
      if (["the", "this", "that", "how", "what", "when", "where", "system", "swasthik", "lunaite"].includes(lower)) continue;
      const id = `entity_${lower.replace(/\s+/g, '_')}`;
      if (!detectedEntities.some(e => e.id === id)) {
        detectedEntities.push({ id, label: clean, type: "concept" });
      }
    }
  }

  // Add detected entities to knowledge graph via API
  for (const ent of detectedEntities) {
    try {
      const sourceId = state.lastEntityId || "lunaite_ai";
      const payload = {
        id: ent.id,
        label: ent.label,
        type: ent.type,
        source_id: sourceId !== ent.id ? sourceId : undefined,
        link_label: "related_to"
      };

      const res = await fetch('/api/knowledge/add', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (data.graph) {
        kgData = data.graph;
        renderKnowledgeGraph();
      }
      state.lastEntityId = ent.id;
    } catch (e) {}
  }
}

// ─── Utility ───────────────────────────────────────────────────────────────
function escapeHtml(text) {
  if (typeof text !== 'string') text = String(text || '');
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
