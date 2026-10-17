import glob
import json
import os
import threading
import webbrowser
from datetime import datetime
from typing import Dict, List, Optional, Tuple;

import requests
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template_string, request, stream_with_context

load_dotenv()

app = Flask(__name__)

# =====================================================================
# 1. CẤU HÌNH HỆ THỐNG & QUẢN LÝ TRẠNG THÁI (BACKEND OOP)
# =====================================================================
class LiemConfig:
    MAX_CONTEXT_TURNS = 12  # Giữ 12 message gần nhất như bản 2.1
    QUESTION_LIMIT = 5      # Giới hạn bản Free
    SECRET_KEY = "key"      # Mã kích hoạt Premium
    PROMPT_FILE = "prompt_v30.txt"

    GEMINI_API_KEY = (
        os.getenv("GEMINI_API_KEY_V20")
        or os.getenv("GEMINI_API_KEY_V15")
        or os.getenv("GEMINI_API_KEY")
    )
    GROQ_API_KEY = os.getenv("API_KEY_V10") or os.getenv("GROQ_API_KEY")

    MODELS = {
        "gemini-4-argon": {"name": "Gemini 4 Argon (Pro - v2.0)", "provider": "gemini"},
        "gemini-3.6-flash": {"name": "Gemini 3.6 Flash (Fast - v1.5)", "provider": "gemini"},
        "gemini-3.5-flash": {"name": "Gemini 3.5 Flash (Stable - v2.1)", "provider": "gemini"},
        "qwen/qwen3.8-27b": {"name": "Qwen 3.8 27B (Groq - v1.0)", "provider": "groq"},
    }


class ChatState:
    def __init__(self):
        self.user_name: str = "Lieman"
        self.model_id: str = "gemini-3.5-flash"
        self.temperature: float = 0.7
        self.is_premium: bool = False
        self.num_of_questions: int = 0
        self.is_persistent: bool = True
        self.history: List[Dict[str, str]] = []
        self.http_session = requests.Session()

    def get_system_prompt(self) -> str:
        for file_candidate in [LiemConfig.PROMPT_FILE, "prompt_v20.txt", "prompt_v15.txt"]:
            if os.path.isfile(file_candidate):
                try:
                    with open(file_candidate, "r", encoding="utf-8") as f:
                        return f.read().replace("{name}", self.user_name)
                except Exception:
                    pass
        return f"You are LiemAI Lumique 3, an intelligent, helpful AI collaborator chatting with {self.user_name}."

    def get_trimmed_history(self) -> List[Dict[str, str]]:
        sliced = self.history[-LiemConfig.MAX_CONTEXT_TURNS:] if len(self.history) > LiemConfig.MAX_CONTEXT_TURNS else list(self.history)
        while sliced and sliced[0]["role"] != "user":
            sliced.pop(0)
        return sliced

    def to_gemini_payload(self) -> Dict:
        contents = [
            {
                "role": "user" if msg["role"] == "user" else "model",
                "parts": [{"text": msg["content"]}]
            }
            for msg in self.get_trimmed_history()
        ]
        return {
            "contents": contents,
            "systemInstruction": {"parts": [{"text": self.get_system_prompt()}]},
            "generationConfig": {"temperature": self.temperature}
        }

    def to_groq_payload(self) -> Dict:
        messages = [{"role": "system", "content": self.get_system_prompt()}]
        for msg in self.get_trimmed_history():
            messages.append({"role": msg["role"], "content": msg["content"]})
        return {
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True
        }

    def auto_save(self):
        if not self.is_persistent:
            return
        filename = f"session_{self.user_name}.json"
        self.save_to_file(filename)

    def save_to_file(self, filename: str):
        if not filename.endswith(".json"):
            filename += ".json"
        # Định dạng tương thích ngược hoàn toàn với LiemAI 2.0 & 2.1
        data = [
            {
                "role": "user" if m["role"] == "user" else "model",
                "parts": [{"text": m["content"]}],
                "time": m.get("time", "")
            }
            for m in self.history
        ]
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return filename

    def load_from_file(self, filename: str):
        if not filename.endswith(".json"):
            filename += ".json"
        with open(filename, "r", encoding="utf-8") as f:
            raw = json.load(f)
        loaded = []
        for item in raw:
            role = "user" if item.get("role") == "user" else "assistant"
            content = item["parts"][0]["text"] if "parts" in item and item["parts"] else item.get("content", "")
            loaded.append({"role": role, "content": content, "time": item.get("time", "")})
        self.history = loaded


state = ChatState()


# =====================================================================
# 2. GIAO DIỆN WEB (HTML5 + TAILWIND CSS + MARKED.JS + HIGHLIGHT.JS)
# =====================================================================
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="vi" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LiemAI Lumique 3</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/tokyo-night-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        body { font-family: 'Inter', sans-serif; background-color: #0e1117; color: #e2e8f0; }
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
        .prose pre { background-color: #090d16 !important; border: 1px solid #1e293b; border-radius: 0.75rem; padding: 1rem; margin: 0.75rem 0; overflow-x: auto; position: relative; }
        .prose code { font-size: 0.875rem; color: #38bdf8; }
        .prose pre code { color: #e2e8f0; }
        .prose p { margin-bottom: 0.75rem; line-height: 1.65; }
        .prose ul, .prose ol { margin-left: 1.5rem; margin-bottom: 0.75rem; list-style-type: disc; }
        .prose table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
        .prose th, .prose td { border: 1px solid #334155; padding: 0.5rem 0.75rem; text-align: left; }
        .prose th { background-color: #1e293b; }
    </style>
</head>
<body class="h-screen flex overflow-hidden select-none">

    <!-- SIDEBAR TRÁI -->
    <aside class="w-72 bg-[#161b22] border-r border-slate-800 flex flex-col justify-between p-4 shrink-0">
        <div class="space-y-4 overflow-y-auto pr-1">
            <!-- Logo & Version -->
            <div class="flex items-center justify-between px-2 pt-1">
                <div class="flex items-center gap-3">
                    <!-- Ảnh cờ online, nếu lỗi mạng (onerror) sẽ tự ẩn đi và hiện chữ VN thay thế -->
                    <div class="relative w-10 h-10 shrink-0">
                        <img src="https://flagcdn.com/w80/vn.png" alt="VN Flag" 
                             class="w-10 h-10 object-contain rounded-md shadow-md relative z-10"
                             onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
                        <!-- Khối chữ VN dự phòng (mặc định ẩn, chỉ hiện lên khi ảnh lỗi) -->
                        <div class="absolute inset-0 hidden w-10 h-10 rounded-xl bg-gradient-to-tr from-red-600 via-rose-500 to-amber-400 items-center justify-center font-extrabold text-white text-xs tracking-wider shadow-lg shadow-red-500/20 border border-red-400/30 z-0">
                            VN
                        </div>
                    </div>
                    <div>
                        <h1 class="font-bold text-base tracking-tight text-white">LiemAI <span class="text-xs px-1.5 py-0.5 rounded bg-rose-500/20 text-rose-300 font-semibold">Lumique 3</span></h1>
                        <p class="text-[11px] text-slate-400">Multi-Model Web Studio</p>
                    </div>
                </div>
            </div>

            <!-- Nút Cuộc trò chuyện mới -->
            <button onclick="resetChat()" class="w-full py-2.5 px-3.5 rounded-xl bg-slate-800/80 hover:bg-slate-700/80 border border-slate-700 text-sm font-medium text-slate-200 flex items-center gap-2 transition">
                <span>＋</span> Cuộc trò chuyện mới
            </button>

            <!-- Hồ sơ & Cấu hình -->
            <div class="bg-[#0d1117] p-3 rounded-xl border border-slate-800/80 space-y-3">
                <div>
                    <label class="text-[11px] uppercase tracking-wider text-slate-400 font-semibold">Tên của bạn ({name})</label>
                    <input id="userNameInput" type="text" value="Lieman" onchange="updateSettings()"
                        class="mt-1 w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-sm text-white focus:outline-none focus:border-indigo-500">
                </div>
                <div>
                    <div class="flex justify-between text-[11px] text-slate-400 font-semibold">
                        <span>ĐỘ SÁNG TẠO (TEMP)</span>
                        <span id="tempVal" class="text-indigo-400">0.7</span>
                    </div>
                    <input id="tempSlider" type="range" min="0" max="1.5" step="0.1" value="0.7"
                        oninput="document.getElementById('tempVal').innerText = this.value" onchange="updateSettings()"
                        class="w-full accent-indigo-500 mt-1 cursor-pointer">
                </div>
                <div class="flex items-center justify-between pt-1">
                    <span class="text-xs text-slate-400">Tự động lưu JSON</span>
                    <input id="autoSaveToggle" type="checkbox" checked onchange="updateSettings()" class="accent-indigo-500 w-4 h-4 cursor-pointer">
                </div>
            </div>

            <!-- Quản lý Phiên (Sessions) -->
            <div class="space-y-2">
                <div class="flex items-center justify-between px-1">
                    <span class="text-[11px] uppercase tracking-wider text-slate-400 font-semibold">Lịch sử đã lưu (.json)</span>
                    <button onclick="saveSessionPrompt()" class="text-xs text-indigo-400 hover:text-indigo-300 font-medium">+ Lưu mới</button>
                </div>
                <div id="sessionList" class="space-y-1 max-h-48 overflow-y-auto text-xs"></div>
            </div>
        </div>

        <!-- Khung Quota & Nâng cấp Premium -->
        <div class="pt-3 border-t border-slate-800 space-y-2.5">
            <div id="quotaBox" class="bg-gradient-to-br from-slate-900 to-slate-800 p-3 rounded-xl border border-slate-700/80">
                <div class="flex items-center justify-between mb-1.5">
                    <span id="planBadge" class="text-xs font-semibold px-2 py-0.5 rounded bg-amber-500/20 text-amber-300">⚡ FREE TIER</span>
                    <span id="quotaText" class="text-xs text-slate-400">0 / 5 câu</span>
                </div>
                <div class="w-full bg-slate-700 h-1.5 rounded-full overflow-hidden mb-2.5">
                    <div id="quotaBar" class="bg-amber-400 h-full transition-all duration-300" style="width: 0%"></div>
                </div>
                <div id="keyInputGroup" class="flex gap-1.5">
                    <input id="premiumKeyInput" type="password" placeholder="Nhập key Premium..."
                        class="w-full bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1 text-xs text-white focus:outline-none focus:border-purple-500">
                    <button onclick="unlockPremium()" class="px-2.5 py-1 bg-purple-600 hover:bg-purple-500 text-white rounded-lg text-xs font-medium shrink-0 transition">Mở</button>
                </div>
            </div>
            <button onclick="exportMarkdown()" class="w-full py-2 px-3 rounded-xl bg-slate-800/60 hover:bg-slate-800 text-xs text-slate-300 flex items-center justify-center gap-1.5 border border-slate-800 transition">
                📥 Xuất hội thoại ra Markdown (.md)
            </button>
        </div>
    </aside>

    <!-- KHU VỰC CHAT CHÍNH -->
    <main class="flex-1 flex flex-col bg-[#0e1117] relative select-text">
        <!-- Thanh Header chọn Model -->
        <header class="h-14 border-b border-slate-800/80 px-6 flex items-center justify-between bg-[#0e1117]/90 backdrop-blur z-10">
            <div class="flex items-center gap-3">
                <span class="text-xs text-slate-400 font-medium">Mô hình:</span>
                <select id="modelSelect" onchange="updateSettings()"
                    class="bg-[#161b22] border border-slate-700 text-slate-100 text-sm rounded-xl px-3 py-1.5 focus:outline-none focus:border-indigo-500 font-medium cursor-pointer">
                    <option value="gemini-4-argon">✨ Gemini 4 Argon (Siêu trí tuệ - Bản 2.0)</option>
                    <option value="gemini-3.6-flash">⚡ Gemini 3.6 Flash (Tốc độ cao - Bản 1.5)</option>
                    <option value="gemini-3.5-flash" selected>🚀 Gemini 3.5 Flash (Ổn định - Bản 2.1)</option>
                    <option value="qwen/qwen3.8-27b">🔥 Qwen 3.8 27B qua Groq (Bản 1.0)</option>
                </select>
            </div>
            <div class="flex items-center gap-2">
                <button onclick="undoLastTurn()" title="Thu hồi câu hỏi gần nhất" class="text-xs px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition">↩ Hoàn tác (/undo)</button>
            </div>
        </header>

        <!-- Luồng tin nhắn -->
        <div id="chatContainer" class="flex-1 overflow-y-auto px-4 md:px-16 lg:px-32 py-6 space-y-6">
            <!-- Màn hình chào mừng mặc định -->
            <div id="welcomeScreen" class="h-full flex flex-col items-center justify-center text-center max-w-2xl mx-auto space-y-5 py-12">
                <div class="w-14 h-14 rounded-2xl bg-gradient-to-tr from-blue-600 via-indigo-500 to-purple-500 flex items-center justify-center text-2xl font-bold text-white shadow-xl shadow-indigo-500/20">✨</div>
                <h2 class="text-2xl md:text-3xl font-bold bg-gradient-to-r from-blue-400 via-indigo-300 to-purple-400 bg-clip-text text-transparent">
                    Xin chào <span id="welcomeName">Lieman</span>, hôm nay tôi có thể giúp gì cho bạn?
                </h2>
                <p class="text-sm text-slate-400">LiemAI Lumique 3 tích hợp truyền tải trực tiếp (SSE Streaming), đính kèm tài liệu/code và chuyển đổi nóng giữa Gemini & Groq.</p>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-3 w-full text-left pt-2">
                    <button onclick="quickPrompt('Hãy tóm tắt các điểm mạnh của kiến trúc OOP trong dự án của tôi.')" class="p-3.5 rounded-xl bg-[#161b22] hover:bg-slate-800/90 border border-slate-800 transition">
                        <div class="text-xs font-semibold text-indigo-400 mb-1">💻 Phân tích Code & OOP</div>
                        <div class="text-xs text-slate-300">Kiểm tra cấu trúc hướng đối tượng và đề xuất cải tiến</div>
                    </button>
                    <button onclick="quickPrompt('Hãy giúp tôi phác thảo diễn biến tiếp theo cho tiểu thuyết.')" class="p-3.5 rounded-xl bg-[#161b22] hover:bg-slate-800/90 border border-slate-800 transition">
                        <div class="text-xs font-semibold text-purple-400 mb-1">✒️ Sáng tác & Cốt truyện</div>
                        <div class="text-xs text-slate-300">Phát triển kịch bản và chiều sâu tâm lý nhân vật</div>
                    </button>
                </div>
            </div>
        </div>

        <!-- Thanh công cụ nhập liệu dưới đáy -->
        <div class="px-4 md:px-16 lg:px-32 pb-5 pt-2 bg-gradient-to-t from-[#0e1117] via-[#0e1117] to-transparent">
            <div class="max-w-4xl mx-auto">
                <!-- Chip hiển thị file đính kèm -->
                <div id="attachedFileChip" class="hidden mb-2 inline-flex items-center gap-2 bg-indigo-950/80 border border-indigo-500/40 text-indigo-200 px-3 py-1 rounded-xl text-xs">
                    <span>📎</span>
                    <span id="attachedFileName" class="font-medium"></span>
                    <button onclick="clearAttachment()" class="hover:text-red-400 ml-1 font-bold">✕</button>
                </div>

                <!-- Khung nhập tin nhắn -->
                <div class="bg-[#161b22] border border-slate-700/90 focus-within:border-indigo-500 rounded-2xl p-2.5 flex items-end gap-2 shadow-xl">
                    <!-- Nút đính kèm file thay cho lệnh /file cũ -->
                    <input type="file" id="fileInput" class="hidden" onchange="handleFileSelect(event)">
                    <button onclick="document.getElementById('fileInput').click()" title="Đính kèm file code/văn bản"
                        class="p-2 rounded-xl hover:bg-slate-800 text-slate-400 hover:text-indigo-400 transition shrink-0">
                        📎
                    </button>

                    <textarea id="messageInput" rows="1" placeholder="Nhập tin nhắn cho LiemAI (Enter để gửi, Shift+Enter xuống dòng)..."
                        oninput="autoResize(this)" onkeydown="handleKeydown(event)"
                        class="w-full bg-transparent text-sm text-slate-100 placeholder-slate-500 focus:outline-none resize-none max-h-44 py-1.5"></textarea>

                    <button id="stopBtn" onclick="stopGeneration()" class="hidden px-3 py-2 rounded-xl bg-red-600/90 hover:bg-red-500 text-white text-xs font-semibold shrink-0 transition">
                        ⏹ Dừng
                    </button>
                    <button id="sendBtn" onclick="sendMessage()" class="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shrink-0 transition">
                        Gửi ➤
                    </button>
                </div>
            </div>
        </div>
    </main>

    <script>
        let attachedFile = { name: "", content: "" };
        let abortController = null;

        marked.setOptions({
            highlight: function(code, lang) {
                if (lang && hljs.getLanguage(lang)) {
                    return hljs.highlight(code, { language: lang }).value;
                }
                return hljs.highlightAuto(code).value;
            },
            breaks: true
        });

        function autoResize(el) {
            el.style.height = 'auto';
            el.style.height = Math.min(el.scrollHeight, 176) + 'px';
        }

        function handleKeydown(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        }

        function quickPrompt(text) {
            const input = document.getElementById('messageInput');
            input.value = text;
            sendMessage();
        }

        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = function(e) {
                attachedFile = { name: file.name, content: e.target.result };
                document.getElementById('attachedFileName').innerText = `${file.name} (${(file.size/1024).toFixed(1)} KB)`;
                document.getElementById('attachedFileChip').classList.remove('hidden');
            };
            reader.readAsText(file, "UTF-8");
        }

        function clearAttachment() {
            attachedFile = { name: "", content: "" };
            document.getElementById('fileInput').value = "";
            document.getElementById('attachedFileChip').classList.add('hidden');
        }

        function createMessageBubble(role, htmlContent) {
            document.getElementById('welcomeScreen')?.classList.add('hidden');
            const container = document.getElementById('chatContainer');
            const wrapper = document.createElement('div');

            if (role === 'user') {
                wrapper.className = "flex justify-end";
                wrapper.innerHTML = `
                    <div class="max-w-2xl bg-indigo-600/20 border border-indigo-500/30 text-slate-100 rounded-2xl rounded-tr-sm px-4 py-3 text-sm whitespace-pre-wrap">${escapeHtml(htmlContent)}</div>
                `;
            } else {
                wrapper.className = "flex gap-3 items-start max-w-4xl";
                wrapper.innerHTML = `
                    <div class="w-8 h-8 rounded-xl bg-gradient-to-tr from-black-600 to-red-500 flex items-center justify-center text-xs font-bold text-white shrink-0 mt-0.5">AI</div>
                    <div class="flex-1 bg-[#161b22] border border-slate-800 rounded-2xl rounded-tl-sm px-5 py-3.5 text-sm prose prose-invert max-w-none message-content">${htmlContent}</div>
                `;
            }
            container.appendChild(wrapper);
            container.scrollTop = container.scrollHeight;
            return wrapper.querySelector('.message-content');
        }

        function escapeHtml(str) {
            return str.replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
        }

        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const text = input.value.trim();
            if (!text && !attachedFile.content) return;

            let displayMsg = attachedFile.name ? `📎 [Đính kèm: ${attachedFile.name}]\n${text}` : text;
            createMessageBubble('user', displayMsg);

            const payload = {
                message: text || "Hãy phân tích nội dung file đính kèm này.",
                file_name: attachedFile.name,
                file_content: attachedFile.content
            };

            input.value = "";
            input.style.height = 'auto';
            clearAttachment();

            const aiBubble = createMessageBubble('assistant', '<span class=" text-slate-400 animate-pulse">Đang suy nghĩ...</span>');
            document.getElementById('sendBtn').classList.add('hidden');
            document.getElementById('stopBtn').classList.remove('hidden');

            abortController = new AbortController();
            let fullMarkdown = "";

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                    signal: abortController.signal
                });

                if (response.status === 403) {
                    const errData = await response.json();
                    aiBubble.innerHTML = `<span class="text-amber-400 font-semibold">🔒 ${errData.error}</span>`;
                    document.getElementById('premiumKeyInput').focus();
                    return;
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder("utf-8");
                let buffer = "";

                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split("\n\n");
                    buffer = lines.pop();

                    for (const line of lines) {
                        if (line.startsWith("data: ")) {
                            const data = JSON.parse(line.slice(6));
                            if (data.error) {
                                aiBubble.innerHTML = `<span class="text-red-400 font-semibold">✖ ${escapeHtml(data.error)}</span>`;
                            } else if (data.text) {
                                fullMarkdown += data.text;
                                aiBubble.innerHTML = marked.parse(fullMarkdown);
                                document.getElementById('chatContainer').scrollTop = document.getElementById('chatContainer').scrollHeight;
                            }
                        }
                    }
                }
            } catch (err) {
                if (err.name === 'AbortError') {
                    fullMarkdown += "\n\n*(Đã dừng sinh văn bản)*";
                    aiBubble.innerHTML = marked.parse(fullMarkdown);
                } else {
                    aiBubble.innerHTML = `<span class="text-red-400">✖ Lỗi kết nối: ${err.message}</span>`;
                }
            } finally {
                document.getElementById('sendBtn').classList.remove('hidden');
                document.getElementById('stopBtn').classList.add('hidden');
                refreshStatus();
            }
        }

        function stopGeneration() {
            if (abortController) abortController.abort();
        }

        async function updateSettings() {
            const user_name = document.getElementById('userNameInput').value.trim() || "User";
            const model_id = document.getElementById('modelSelect').value;
            const temperature = parseFloat(document.getElementById('tempSlider').value);
            const is_persistent = document.getElementById('autoSaveToggle').checked;
            document.getElementById('welcomeName').innerText = user_name;

            await fetch('/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_name, model_id, temperature, is_persistent })
            });
            refreshStatus();
        }

        async function unlockPremium() {
            const key = document.getElementById('premiumKeyInput').value.trim();
            const res = await fetch('/api/unlock', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ key })
            });
            const data = await res.json();
            if (data.success) {
                document.getElementById('premiumKeyInput').value = "";
                refreshStatus();
            } else {
                alert("Key không chính xác! (Gợi ý: nhập 'key')");
            }
        }

        async function resetChat() {
            await fetch('/api/reset', { method: 'POST' });
            location.reload();
        }

        async function undoLastTurn() {
            await fetch('/api/undo', { method: 'POST' });
            await reloadChatView();
        }

        async function saveSessionPrompt() {
            const name = prompt("Nhập tên file muốn lưu (.json):", `session_${document.getElementById('userNameInput').value}.json`);
            if (!name) return;
            await fetch('/api/sessions/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename: name })
            });
            refreshStatus();
        }

        async function loadSession(filename) {
            await fetch('/api/sessions/load', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename })
            });
            await reloadChatView();
        }

        async function reloadChatView() {
            const res = await fetch('/api/status');
            const data = await res.json();
            const container = document.getElementById('chatContainer');
            container.innerHTML = "";
            for (const m of data.history) {
                const bubble = createMessageBubble(m.role, m.role === 'user' ? m.content : marked.parse(m.content));
            }
            refreshStatus();
        }

        function exportMarkdown() {
            window.location.href = '/api/export_md';
        }

        async function refreshStatus() {
            const res = await fetch('/api/status');
            const data = await res.json();

            if (data.is_premium) {
                document.getElementById('planBadge').className = "text-xs font-semibold px-2 py-0.5 rounded bg-purple-500/20 text-purple-300";
                document.getElementById('planBadge').innerText = "👑 PREMIUM";
                document.getElementById('quotaText').innerText = "Không giới hạn";
                document.getElementById('quotaBar').style.width = "100%";
                document.getElementById('quotaBar').className = "bg-gradient-to-r from-indigo-500 to-purple-500 h-full";
                document.getElementById('keyInputGroup').classList.add('hidden');
            } else {
                const pct = Math.min((data.num_of_questions / data.question_limit) * 100, 100);
                document.getElementById('quotaText').innerText = `${data.num_of_questions} / ${data.question_limit} câu`;
                document.getElementById('quotaBar').style.width = `${pct}%`;
            }

            const sList = document.getElementById('sessionList');
            sList.innerHTML = data.sessions.map(f => `
                <button onclick="loadSession('${f}')" class="w-full text-left px-2.5 py-1.5 rounded-lg hover:bg-slate-800 text-slate-300 truncate flex items-center gap-2 transition">
                    <span>💬</span> <span class="truncate">${f}</span>
                </button>
            `).join('');
        }

        window.onload = refreshStatus;
    </script>
</body>
</html>
"""


# =====================================================================
# 3. CÁC ĐIỂM CUỐI API (FLASK ROUTES & SSE STREAMING)
# =====================================================================
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/status")
def api_status():
    json_files = sorted(glob.glob("*.json"))
    return jsonify({
        "user_name": state.user_name,
        "model_id": state.model_id,
        "temperature": state.temperature,
        "is_premium": state.is_premium,
        "num_of_questions": state.num_of_questions,
        "question_limit": LiemConfig.QUESTION_LIMIT,
        "sessions": json_files,
        "history": state.history
    })


@app.route("/api/settings", methods=["POST"])
def api_settings():
    data = request.json or {}
    state.user_name = data.get("user_name", state.user_name)
    state.model_id = data.get("model_id", state.model_id)
    state.temperature = float(data.get("temperature", state.temperature))
    state.is_persistent = bool(data.get("is_persistent", state.is_persistent))
    return jsonify({"success": True})


@app.route("/api/unlock", methods=["POST"])
def api_unlock():
    key = (request.json or {}).get("key", "").strip()
    if key == LiemConfig.SECRET_KEY:
        state.is_premium = True
        return jsonify({"success": True})
    return jsonify({"success": False})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    state.history.clear()
    return jsonify({"success": True})


@app.route("/api/undo", methods=["POST"])
def api_undo():
    if len(state.history) >= 2 and state.history[-1]["role"] == "assistant":
        state.history.pop()
        state.history.pop()
    elif state.history:
        state.history.pop()
    return jsonify({"success": True})


@app.route("/api/sessions/save", methods=["POST"])
def api_save_session():
    filename = (request.json or {}).get("filename", f"session_{state.user_name}.json")
    saved = state.save_to_file(filename)
    return jsonify({"success": True, "filename": saved})


@app.route("/api/sessions/load", methods=["POST"])
def api_load_session():
    filename = (request.json or {}).get("filename", "")
    state.load_from_file(filename)
    return jsonify({"success": True})


@app.route("/api/export_md")
def api_export_md():
    lines = [
        f"# Lịch sử trò chuyện LiemAI - {state.user_name}",
        f"*Xuất lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n---\n"
    ]
    for m in state.history:
        who = f"🧑 **{state.user_name}**" if m["role"] == "user" else "🤖 **LiemAI**"
        lines.append(f"### {who} `[{m.get('time', '')}]`\n\n{m['content']}\n\n---\n")
    md_content = "\n".join(lines)
    return Response(
        md_content,
        mimetype="text/markdown",
        headers={"Content-Disposition": f"attachment;filename=LiemAI_{state.user_name}.md"}
    )


@app.route("/api/chat", methods=["POST"])
def api_chat():
    if not state.is_premium and state.num_of_questions >= LiemConfig.QUESTION_LIMIT:
        return jsonify({
            "error": f"Bạn đã hết {LiemConfig.QUESTION_LIMIT} lượt hỏi miễn phí! Vui lòng nhập mã khóa Lumique Premium ở góc trái màn hình."
        }), 403

    data = request.json or {}
    user_msg = data.get("message", "")
    file_name = data.get("file_name", "")
    file_content = data.get("file_content", "")

    if file_name and file_content:
        full_prompt = f"[File đính kèm: {file_name}]\n```\n{file_content}\n```\n\n{user_msg}"
    else:
        full_prompt = user_msg

    state.history.append({
        "role": "user",
        "content": full_prompt,
        "time": datetime.now().strftime("%H:%M:%S")
    })

    provider = LiemConfig.MODELS.get(state.model_id, {}).get("provider", "gemini")

    def generate_sse():
        full_answer = ""
        try:
            if provider == "gemini":
                if not LiemConfig.GEMINI_API_KEY:
                    state.history.pop()
                    yield f"data: {json.dumps({'error': 'Thiếu GEMINI_API_KEY trong file .env!'})}\n\n"
                    return
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{state.model_id}:streamGenerateContent?alt=sse"
                headers = {"Content-Type": "application/json", "x-goog-api-key": LiemConfig.GEMINI_API_KEY}
                payload = state.to_gemini_payload()
            else:
                if not LiemConfig.GROQ_API_KEY:
                    state.history.pop()
                    yield f"data: {json.dumps({'error': 'Thiếu API_KEY_V10 (Groq) trong file .env!'})}\n\n"
                    return
                url = "https://api.groq.com/openai/v1/chat/completions"
                headers = {"Authorization": f"Bearer {LiemConfig.GROQ_API_KEY}", "Content-Type": "application/json"}
                payload = state.to_groq_payload()

            resp = state.http_session.post(url, headers=headers, json=payload, stream=True, timeout=(10, 60))

            if resp.status_code != 200:
                state.history.pop()
                yield f"data: {json.dumps({'error': f'Lỗi HTTP {resp.status_code}: {resp.text}'})}\n\n"
                return

            for line in resp.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8", errors="ignore").strip()
                if not decoded.startswith("data: "):
                    continue
                json_str = decoded[6:].strip()
                if json_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(json_str)
                    if provider == "gemini":
                        piece = chunk["candidates"][0]["content"]["parts"][0]["text"]
                    else:
                        piece = chunk["choices"][0]["delta"].get("content", "")
                    if piece:
                        full_answer += piece
                        yield f"data: {json.dumps({'text': piece})}\n\n"
                except (KeyError, IndexError, json.JSONDecodeError):
                    pass

            if full_answer:
                state.history.append({
                    "role": "assistant",
                    "content": full_answer,
                    "time": datetime.now().strftime("%H:%M:%S")
                })
                if not state.is_premium:
                    state.num_of_questions += 1
                state.auto_save()
            else:
                state.history.pop()

        except Exception as e:
            state.history.pop()
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(stream_with_context(generate_sse()), mimetype="text/event-stream")


if __name__ == "__main__":
    port = 5000
    url = f"http://127.0.0.1:{port}"
    print(f"🚀 Đang khởi động LiemAI Lumique 3 tại: {url}")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)