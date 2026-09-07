/**
 * app.js
 * ======
 * Client-side interactive logic for Multilingual AI Voice Assistant & RAG.
 */

let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let currentAudioElement = null;

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", () => {
    loadIndexedDocuments();
    setupDropZone();
    setupInputListeners();
});

/* ============================================================
   CHAT MESSAGING
   ============================================================ */

async function submitMessage(queryText = null) {
    const inputElem = document.getElementById("userInput");
    const message = queryText || inputElem.value.trim();

    if (!message) return;

    // Clear input
    inputElem.value = "";
    inputElem.style.height = "auto";

    // Hide welcome box if visible
    const welcomeBox = document.querySelector(".chat-welcome");
    if (welcomeBox) welcomeBox.style.display = "none";

    // Append User Message to UI
    appendChatMessage("user", message);

    // Selected Language
    const selectedLang = document.getElementById("languageSelect").value;

    // Show Assistant Loading Indicator
    const loadingId = appendLoadingMessage();

    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: message,
                language: selectedLang
            })
        });

        const data = await response.json();
        removeLoadingMessage(loadingId);

        if (response.ok) {
            appendChatMessage("assistant", data.reply, data.audio_url, data.language_name);
            if (data.audio_url) {
                playTTSAudio(data.audio_url);
            }
        } else {
            appendChatMessage("assistant", "⚠️ Error processing request: " + (data.detail || "Server error"));
        }
    } catch (err) {
        removeLoadingMessage(loadingId);
        appendChatMessage("assistant", "⚠️ Connection error. Please check backend server.");
        console.error("Chat error:", err);
    }
}

function sendQuickQuery(text) {
    submitMessage(text);
}

function appendChatMessage(role, text, audioUrl = null, langName = null) {
    const container = document.getElementById("chatContainer");
    const msgDiv = document.createElement("div");
    msgDiv.className = `chat-msg ${role}`;

    const avatarIcon = role === "user" ? "fa-user" : "fa-robot";

    let html = `
        <div class="msg-avatar"><i class="fa-solid ${avatarIcon}"></i></div>
        <div class="msg-content">
            <p>${escapeHtml(text)}</p>
    `;

    if (langName) {
        html += `<span class="badge status-badge" style="margin-top: 6px; display: inline-block;">${langName}</span>`;
    }

    if (audioUrl) {
        html += `
            <br>
            <button class="msg-audio-btn" onclick="playTTSAudio('${audioUrl}')">
                <i class="fa-solid fa-volume-high"></i> Listen Speech
            </button>
        `;
    }

    html += `</div>`;
    msgDiv.innerHTML = html;

    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;
}

function appendLoadingMessage() {
    const id = "loading_" + Date.now();
    const container = document.getElementById("chatContainer");
    const msgDiv = document.createElement("div");
    msgDiv.className = "chat-msg assistant";
    msgDiv.id = id;
    msgDiv.innerHTML = `
        <div class="msg-avatar"><i class="fa-solid fa-robot"></i></div>
        <div class="msg-content">
            <div class="spinner"></div> Thinking & Retrieving context...
        </div>
    `;
    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;
    return id;
}

function removeLoadingMessage(id) {
    const elem = document.getElementById(id);
    if (elem) elem.remove();
}

function playTTSAudio(url) {
    if (currentAudioElement) {
        currentAudioElement.pause();
    }
    currentAudioElement = new Audio(url);
    currentAudioElement.play().catch(e => console.error("Audio playback error:", e));
}

/* ============================================================
   MICROPHONE VOICE RECORDING
   ============================================================ */

async function toggleRecording() {
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
            uploadRecordedAudio(audioBlob);
        };

        mediaRecorder.start();
        isRecording = true;
        document.getElementById("micBtn").classList.add("active");
        document.getElementById("recordingOverlay").classList.remove("hidden");
    } catch (err) {
        alert("Microphone permission denied or microphone not found.");
        console.error("Mic error:", err);
    }
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        document.getElementById("micBtn").classList.remove("active");
        document.getElementById("recordingOverlay").classList.add("hidden");

        // Stop all audio tracks
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
    }
}

async function uploadRecordedAudio(blob) {
    const formData = new FormData();
    formData.append("file", blob, "user_speech.wav");

    const loadingId = appendLoadingMessage();

    try {
        const response = await fetch("/api/stt", {
            method: "POST",
            body: formData
        });

        const data = await response.json();
        removeLoadingMessage(loadingId);

        if (response.ok && data.text) {
            submitMessage(data.text);
        } else {
            alert("Speech transcription failed. Please speak again.");
        }
    } catch (err) {
        removeLoadingMessage(loadingId);
        console.error("STT Error:", err);
    }
}

/* ============================================================
   RAG DOCUMENT UPLOAD & MANAGEMENT
   ============================================================ */

function setupDropZone() {
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("drag-over");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("drag-over");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("drag-over");
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (fileInput.files.length > 0) {
            handleFileUpload(fileInput.files[0]);
        }
    });
}

async function handleFileUpload(file) {
    const progressElem = document.getElementById("uploadProgress");
    progressElem.classList.remove("hidden");

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("/api/upload", {
            method: "POST",
            body: formData
        });

        const data = await response.json();
        progressElem.classList.add("hidden");

        if (response.ok) {
            loadIndexedDocuments();
            alert(`✅ Ingested '${data.filename}' into RAG knowledge base (${data.chunks_added} chunks).`);
        } else {
            alert("⚠️ Document upload failed: " + (data.detail || "Upload error"));
        }
    } catch (err) {
        progressElem.classList.add("hidden");
        alert("⚠️ Connection error uploading document.");
        console.error("Upload error:", err);
    }
}

async function loadIndexedDocuments() {
    try {
        const response = await fetch("/api/documents");
        const data = await response.json();

        const docList = document.getElementById("docList");
        const countBadge = document.getElementById("docCountBadge");

        countBadge.innerText = `${data.documents.length} files (${data.total_chunks} chunks)`;

        if (!data.documents || data.documents.length === 0) {
            docList.innerHTML = `
                <div class="empty-docs">
                    <i class="fa-solid fa-file-circle-question"></i>
                    <p>No documents uploaded yet.</p>
                </div>
            `;
            return;
        }

        docList.innerHTML = "";
        data.documents.forEach(doc => {
            const item = document.createElement("div");
            item.className = "doc-item";
            item.innerHTML = `
                <div class="doc-info">
                    <i class="fa-solid fa-file-pdf"></i>
                    <span class="doc-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</span>
                </div>
                <button class="delete-doc-btn" onclick="deleteDocument('${escapeHtml(doc.filename)}')" title="Delete Document">
                    <i class="fa-solid fa-trash"></i>
                </button>
            `;
            docList.appendChild(item);
        });
    } catch (err) {
        console.error("Failed to load documents:", err);
    }
}

async function deleteDocument(filename) {
    if (!confirm(`Delete '${filename}' from RAG Knowledge Base?`)) return;

    try {
        const response = await fetch(`/api/documents/${encodeURIComponent(filename)}`, {
            method: "DELETE"
        });

        if (response.ok) {
            loadIndexedDocuments();
        }
    } catch (err) {
        console.error("Delete failed:", err);
    }
}

/* ============================================================
   UTILS & MODALS
   ============================================================ */

function setupInputListeners() {
    const inputElem = document.getElementById("userInput");

    // [Fix 9] Auto-resize textarea height as user types
    inputElem.addEventListener("input", () => {
        inputElem.style.height = "auto";
        inputElem.style.height = Math.min(inputElem.scrollHeight, 160) + "px";
    });

    inputElem.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submitMessage();
        }
    });
}

function openArchModal() {
    document.getElementById("archModal").classList.remove("hidden");
}

function closeArchModal() {
    document.getElementById("archModal").classList.add("hidden");
}

function escapeHtml(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
