import glob
import json
import os
import threading
import webbrowser
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template_string, request, stream_with_context

load_dotenv()

app = Flask(__name__)

# =====================================================================
# 1. CẤU HÌNH HỆ THỐNG & QUẢN LÝ TRẠNG THÁI (BACKEND OOP)
# =====================================================================
class LiemConfig:
    MAX_CONTEXT_TURNS = 12  # Giữ 12 message gần nhất như bản 2.1[cite: 13, 23]
    QUESTION_LIMIT = 5      # Giới hạn bản Free[cite: 13, 23]
    SECRET_KEY = "key"      # Mã kích hoạt Premium[cite: 13, 23]
    PROMPT_FILE = "prompt_v30.txt"
    INTELLIGENCE_FILE = "liemai_intelligence.json"

    GEMINI_API_KEY = (
        os.getenv("GEMINI_API_KEY_V20")
        or os.getenv("GEMINI_API_KEY_V15")
        or os.getenv("GEMINI_API_KEY")
    )
    GROQ_API_KEY = os.getenv("API_KEY_V10") or os.getenv("GROQ_API_KEY")

    MODELS = {
        "gemini-3.7-flash": {"name": "✨ Gemini 3.7 Flash (Thế hệ mới)", "provider": "gemini"},
        "gemini-3.6-flash": {"name": "Gemini 3.6 Flash (Fast - v1.5)", "provider": "gemini"},
        "gemini-3.5-flash": {"name": "Gemini 3.5 Flash (Stable - v2.1)", "provider": "gemini"},
        "qwen/qwen3.8-27b": {"name": "Qwen 3.8 27B (Groq - v1.0)", "provider": "groq"},
    }


class ChatState:
    def __init__(self):
        self.user_name: str = "Lieman"
        self.model_id: str = "gemini-3.5-flash"
        self.temperature: float = 0.7
        self.language: str = "vi"  # "vi" hoặc "en"
        self.is_premium: bool = False
        self.num_of_questions: int = 0
        self.is_persistent: bool = True
        self.history: List[Dict[str, str]] = []
        self.http_session = requests.Session()

        # Trí thông minh cá nhân & Kỹ năng (Skills)
        self.personal_memory: str = (
            "Người dùng là sinh viên tại ĐH Bách Khoa Hà Nội, đam mê lập trình và toán học."
        )
        self.skills: List[Dict] = [
            {
                "id": "skill_oop",
                "name": "💻 Chuyên gia lập trình",
                "instruction": "Khi phân tích code, luôn kiểm tra kỹ 4 tính chất OOP, nguyên lý SOLID, bắt lỗi tiềm ẩn và đề xuất code sạch, tối ưu.",
                "enabled": False
            },
            {
                "id": "skill_novel",
                "name": "✒️ Cố vấn Tiểu thuyết & Cốt truyện",
                "instruction": "Khi thảo luận về văn học hoặc kịch bản, hãy phân tích sâu tâm lý nhân vật, tính nhất quán của cốt truyện và dùng văn phong giàu cảm xúc.",
                "enabled": False
            },
            {
                "id": "skill_concise",
                "name": "⚡ Trả lời siêu cô đọng",
                "instruction": "Trả lời đi thẳng vào trọng tâm, súc tích, ưu tiên dùng gạch đầu dòng và bảng biểu, bỏ qua các câu dẫn dắt rườm rà.",
                "enabled": False
            }
        ]
        self.load_intelligence()

    def load_intelligence(self):
        if os.path.isfile(LiemConfig.INTELLIGENCE_FILE):
            try:
                with open(LiemConfig.INTELLIGENCE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.personal_memory = data.get("personal_memory", self.personal_memory)
                if "skills" in data and isinstance(data["skills"], list):
                    self.skills = data["skills"]
            except Exception:
                pass

    def save_intelligence(self):
        try:
            with open(LiemConfig.INTELLIGENCE_FILE, "w", encoding="utf-8") as f:
                json.dump(
                    {"personal_memory": self.personal_memory, "skills": self.skills},
                    f,
                    ensure_ascii=False,
                    indent=2
                )
        except Exception:
            pass

    def get_system_prompt(self) -> str:
        base_prompt = f"You are LiemAI Lumique 3, an intelligent, helpful AI collaborator chatting with {self.user_name}."
        for file_candidate in [LiemConfig.PROMPT_FILE, "prompt_v20.txt", "prompt_v15.txt"]:
            if os.path.isfile(file_candidate):
                try:
                    with open(file_candidate, "r", encoding="utf-8") as f:
                        base_prompt = f.read().replace("{name}", self.user_name)
                        break
                except Exception:
                    pass

        extra_blocks = []
        if self.personal_memory.strip():
            extra_blocks.append(f"[PERSONAL INTELLIGENCE & USER MEMORY]\n{self.personal_memory.strip()}")

        active_skills = [s for s in self.skills if s.get("enabled")]
        if active_skills:
            skill_lines = "\n".join([f"- {s['name']}: {s['instruction']}" for s in active_skills])
            extra_blocks.append(f"[ACTIVE CUSTOM SKILLS]\n{skill_lines}")

        # Chỉ dẫn ngôn ngữ rõ ràng theo lựa chọn giao diện
        if self.language == "en":
            extra_blocks.append(
                "[LANGUAGE DIRECTIVE]\nThe user has selected English as their UI language. Please respond fluently in English unless explicitly asked otherwise."
            )
        else:
            extra_blocks.append(
                "[LANGUAGE DIRECTIVE]\nNgười dùng đã chọn Tiếng Việt làm ngôn ngữ giao diện. Hãy ưu tiên phản hồi bằng Tiếng Việt tự nhiên, chuẩn mực."
            )

        if extra_blocks:
            return base_prompt + "\n\n" + "\n\n".join(extra_blocks)
        return base_prompt

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
# 2. GIAO DIỆN WEB (TÔNG ĐỎ DỊU + 3 THEMES + SONG NGỮ VI/EN + SKILLS)
# =====================================================================
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="vi" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LiemAI Lumique 3</title>

    <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 256 256'%3E%3Cdefs%3E%3CradialGradient id='bgGrad' cx='50%25' cy='45%25' r='60%25'%3E%3Cstop offset='0%25' stop-color='%23b91c1c'/%3E%3Cstop offset='65%25' stop-color='%2388081f'/%3E%3Cstop offset='100%25' stop-color='%2345030e'/%3E%3C/radialGradient%3E%3ClinearGradient id='goldLight' x1='0' y1='0' x2='1' y2='1'%3E%3Cstop offset='0%25' stop-color='%23fff1b8'/%3E%3Cstop offset='100%25' stop-color='%23d4a034'/%3E%3C/linearGradient%3E%3ClinearGradient id='goldDark' x1='0' y1='0' x2='1' y2='1'%3E%3Cstop offset='0%25' stop-color='%23ba8623'/%3E%3Cstop offset='100%25' stop-color='%23684507'/%3E%3C/linearGradient%3E%3ClinearGradient id='obsLight' x1='0' y1='0' x2='1' y2='1'%3E%3Cstop offset='0%25' stop-color='%23383a42'/%3E%3Cstop offset='100%25' stop-color='%231f2127'/%3E%3C/linearGradient%3E%3ClinearGradient id='obsDark' x1='0' y1='0' x2='1' y2='1'%3E%3Cstop offset='0%25' stop-color='%23181a1f'/%3E%3Cstop offset='100%25' stop-color='%230b0c0e'/%3E%3C/linearGradient%3E%3Cfilter id='starShadow' x='-20%25' y='-20%25' width='140%25' height='140%25'%3E%3CfeDropShadow dx='0' dy='6' stdDeviation='7' flood-color='%23000' flood-opacity='0.6'/%3E%3C/filter%3E%3C/defs%3E%3Crect width='256' height='256' rx='58' fill='url(%23bgGrad)' stroke='%23f43f5e' stroke-opacity='0.25' stroke-width='2'/%3E%3Cg filter='url(%23starShadow)' transform='translate(128,128)' stroke='%23fef08a' stroke-width='1.2' stroke-linejoin='round'%3E%3C!-- Outer 10-point Obsidian Star --%3E%3Cg id='outerStar'%3E%3Cpath d='M0,-112 L0,0 L20.5,-39.5 Z' fill='url(%23obsLight)'/%3E%3Cpath d='M0,-112 L0,0 L-20.5,-39.5 Z' fill='url(%23obsDark)'/%3E%3Cpath d='M65.8,-90.6 L0,0 L43.9,-5.7 Z' fill='url(%23obsLight)'/%3E%3Cpath d='M65.8,-90.6 L0,0 L20.5,-39.5 Z' fill='url(%23obsDark)'/%3E%3Cpath d='M106.5,-34.6 L0,0 L43.9,-5.7 Z' fill='url(%23obsLight)'/%3E%3Cpath d='M106.5,-34.6 L0,0 L35.6,26.7 Z' fill='url(%23obsDark)'/%3E%3Cpath d='M106.5,34.6 L0,0 L35.6,26.7 Z' fill='url(%23obsLight)'/%3E%3Cpath d='M106.5,34.6 L0,0 L0,44.3 Z' fill='url(%23obsDark)'/%3E%3Cpath d='M65.8,90.6 L0,0 L0,44.3 Z' fill='url(%23obsLight)'/%3E%3Cpath d='M65.8,90.6 L0,0 L-35.6,26.7 Z' fill='url(%23obsDark)'/%3E%3Cpath d='M0,112 L0,0 L-35.6,26.7 Z' fill='url(%23obsLight)'/%3E%3Cpath d='M0,112 L0,0 L-43.9,-5.7 Z' fill='url(%23obsDark)'/%3E%3Cpath d='M-65.8,90.6 L0,0 L-43.9,-5.7 Z' fill='url(%23obsLight)'/%3E%3Cpath d='M-65.8,90.6 L0,0 L-20.5,-39.5 Z' fill='url(%23obsDark)'/%3E%3Cpath d='M-106.5,34.6 L0,0 L-20.5,-39.5 Z' fill='url(%23obsLight)'/%3E%3Cpath d='M-106.5,34.6 L0,0 L-43.9,-5.7 Z' fill='url(%23obsDark)'/%3E%3Cpath d='M-106.5,-34.6 L0,0 L-43.9,-5.7 Z' fill='url(%23obsLight)'/%3E%3Cpath d='M-106.5,-34.6 L0,0 L-20.5,-39.5 Z' fill='url(%23obsDark)'/%3E%3Cpath d='M-65.8,-90.6 L0,0 L-20.5,-39.5 Z' fill='url(%23obsLight)'/%3E%3Cpath d='M-65.8,-90.6 L0,0 L0,-44.3 Z' fill='url(%23obsDark)'/%3E%3C/g%3E%3C!-- Inner 10-point Gold Star --%3E%3Cg id='innerStar' stroke='%23fff' stroke-opacity='0.4' stroke-width='0.8'%3E%3Cpath d='M0,-54 L0,0 L10.5,-18.2 Z' fill='url(%23goldLight)'/%3E%3Cpath d='M0,-54 L0,0 L-10.5,-18.2 Z' fill='url(%23goldDark)'/%3E%3Cpath d='M31.7,-43.7 L0,0 L20.9,-2.7 Z' fill='url(%23goldLight)'/%3E%3Cpath d='M31.7,-43.7 L0,0 L10.5,-18.2 Z' fill='url(%23goldDark)'/%3E%3Cpath d='M51.4,-16.7 L0,0 L20.9,-2.7 Z' fill='url(%23goldLight)'/%3E%3Cpath d='M51.4,-16.7 L0,0 L17.0,12.7 Z' fill='url(%23goldDark)'/%3E%3Cpath d='M51.4,16.7 L0,0 L17.0,12.7 Z' fill='url(%23goldLight)'/%3E%3Cpath d='M51.4,16.7 L0,0 L0,21.2 Z' fill='url(%23goldDark)'/%3E%3Cpath d='M31.7,43.7 L0,0 L0,21.2 Z' fill='url(%23goldLight)'/%3E%3Cpath d='M31.7,43.7 L0,0 L-17.0,12.7 Z' fill='url(%23goldDark)'/%3E%3Cpath d='M0,54 L0,0 L-17.0,12.7 Z' fill='url(%23goldLight)'/%3E%3Cpath d='M0,54 L0,0 L-20.9,-2.7 Z' fill='url(%23goldDark)'/%3E%3Cpath d='M-31.7,43.7 L0,0 L-20.9,-2.7 Z' fill='url(%23goldLight)'/%3E%3Cpath d='M-31.7,43.7 L0,0 L-10.5,-18.2 Z' fill='url(%23goldDark)'/%3E%3Cpath d='M-51.4,16.7 L0,0 L-10.5,-18.2 Z' fill='url(%23goldLight)'/%3E%3Cpath d='M-51.4,16.7 L0,0 L-20.9,-2.7 Z' fill='url(%23goldDark)'/%3E%3Cpath d='M-51.4,-16.7 L0,0 L-20.9,-2.7 Z' fill='url(%23goldLight)'/%3E%3Cpath d='M-51.4,-16.7 L0,0 L-10.5,-18.2 Z' fill='url(%23goldDark)'/%3E%3Cpath d='M-31.7,-43.7 L0,0 L-10.5,-18.2 Z' fill='url(%23goldLight)'/%3E%3Cpath d='M-31.7,-43.7 L0,0 L0,-21.2 Z' fill='url(%23goldDark)'/%3E%3C/g%3E%3Ccircle cx='0' cy='0' r='3.5' fill='%23fff6d1' stroke='%23b45309' stroke-width='0.8'/%3E%3C/g%3E%3C/svg%3E">    
    
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/tokyo-night-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        :root, html[data-theme="dark"] {
            --bg-main: #0f0f12;
            --bg-sidebar: #16151a;
            --bg-card: #1d1b22;
            --bg-input: #18161c;
            --border-color: #2d262e;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --bubble-ai: #18161c;
            --bubble-user: rgba(225, 29, 72, 0.16);
            --bubble-user-border: rgba(244, 63, 94, 0.35);
            --code-bg: #09090b;
        }

        html[data-theme="light"] {
            --bg-main: #fffafb;
            --bg-sidebar: #f8f1f3;
            --bg-card: #ffffff;
            --bg-input: #ffffff;
            --border-color: #ead5dc;
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --bubble-ai: #ffffff;
            --bubble-user: #ffe4e6;
            --bubble-user-border: #fda4af;
            --code-bg: #1e1e24;
        }

        html[data-theme="contrast"] {
            --bg-main: #000000;
            --bg-sidebar: #000000;
            --bg-card: #0a0a0a;
            --bg-input: #050505;
            --border-color: #fb7185;
            --text-primary: #ffffff;
            --text-secondary: #fda4af;
            --bubble-ai: #0a0a0a;
            --bubble-user: #3b0712;
            --bubble-user-border: #fb7185;
            --code-bg: #000000;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-main);
            color: var(--text-primary);
            transition: background-color 0.2s ease, color 0.2s ease;
        }
        .theme-sidebar { background-color: var(--bg-sidebar); border-color: var(--border-color); }
        .theme-main { background-color: var(--bg-main); }
        .theme-card { background-color: var(--bg-card); border-color: var(--border-color); }
        .theme-input { background-color: var(--bg-input); border-color: var(--border-color); color: var(--text-primary); }
        .theme-text-sub { color: var(--text-secondary); }
        .theme-border { border-color: var(--border-color); }
        .theme-bubble-ai { background-color: var(--bubble-ai); border-color: var(--border-color); color: var(--text-primary); }
        .theme-bubble-user { background-color: var(--bubble-user); border-color: var(--bubble-user-border); color: var(--text-primary); }

        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #4c353e; border-radius: 3px; }
        .prose pre { background-color: var(--code-bg) !important; border: 1px solid var(--border-color); border-radius: 0.75rem; padding: 1rem; margin: 0.75rem 0; overflow-x: auto; color: #f8fafc; }
        .prose code { font-size: 0.875rem; color: #fb7185; font-weight: 500; }
        .prose pre code { color: #e2e8f0; }
        .prose p { margin-bottom: 0.75rem; line-height: 1.65; }
        .prose ul, .prose ol { margin-left: 1.5rem; margin-bottom: 0.75rem; list-style-type: disc; }
        .prose table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
        .prose th, .prose td { border: 1px solid var(--border-color); padding: 0.5rem 0.75rem; text-align: left; }
        .prose th { background-color: rgba(244, 63, 94, 0.12); }
    </style>
</head>
<body class="h-screen flex overflow-hidden select-none">

    <!-- SIDEBAR TRÁI -->
    <aside class="w-72 theme-sidebar border-r flex flex-col justify-between p-4 shrink-0">
        <div class="space-y-3.5 overflow-y-auto pr-1">
            <!-- Logo & Version -->
            <div class="flex items-center justify-between px-2 pt-1">
                <div class="flex items-center gap-3">
                    <div class="relative w-10 h-10 shrink-0">
                        <img src="https://flagcdn.com/w80/vn.png" alt="VN Flag" 
                             class="w-10 h-10 object-contain rounded-md shadow-md relative z-10"
                             onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
                        <div class="absolute inset-0 hidden w-10 h-10 rounded-xl bg-gradient-to-tr from-red-600 via-rose-500 to-amber-400 items-center justify-center font-extrabold text-white text-xs tracking-wider shadow-lg shadow-rose-500/20 border border-rose-400/30 z-0">
                            VN
                        </div>
                    </div>
                    <div>
                        <h1 class="font-bold text-base tracking-tight">LiemAI <span class="text-xs px-1.5 py-0.5 rounded bg-rose-500/20 text-rose-400 font-semibold">Lumique 3</span></h1>
                        <p class="text-[11px] theme-text-sub">Multi-Model Web Studio</p>
                    </div>
                </div>
            </div>

            <!-- Nút Cuộc trò chuyện mới -->
            <button onclick="resetChat()" class="w-full py-2 px-3.5 rounded-xl theme-card border hover:border-rose-500/60 text-sm font-medium flex items-center gap-2 transition shadow-sm">
                <span class="text-rose-400 font-bold">＋</span> <span data-i18n="newChat">Cuộc trò chuyện mới</span>
            </button>

            <!-- Hồ sơ & Cấu hình -->
            <div class="theme-card p-3 rounded-xl border space-y-2.5">
                <div>
                    <label class="text-[11px] uppercase tracking-wider theme-text-sub font-semibold" data-i18n="userNameLabel">Tên của bạn ({name})</label>
                    <input id="userNameInput" type="text" value="Lieman" onchange="updateSettings()"
                        class="mt-1 w-full theme-input border rounded-lg px-2.5 py-1.5 text-sm focus:outline-none focus:border-rose-500">
                </div>
                <div>
                    <div class="flex justify-between text-[11px] theme-text-sub font-semibold">
                        <span data-i18n="tempLabel">ĐỘ SÁNG TẠO (TEMP)</span>
                        <span id="tempVal" class="text-rose-400 font-bold">0.7</span>
                    </div>
                    <input id="tempSlider" type="range" min="0" max="1.5" step="0.1" value="0.7"
                        oninput="document.getElementById('tempVal').innerText = this.value" onchange="updateSettings()"
                        class="w-full accent-rose-500 mt-1 cursor-pointer">
                </div>
                <div class="flex items-center justify-between pt-0.5">
                    <span class="text-xs theme-text-sub" data-i18n="autoSaveLabel">Tự động lưu JSON</span>
                    <input id="autoSaveToggle" type="checkbox" checked onchange="updateSettings()" class="accent-rose-500 w-4 h-4 cursor-pointer">
                </div>
            </div>

            <!-- KỸ NĂNG & TRÍ THÔNG MINH CÁ NHÂN -->
            <div class="theme-card p-3 rounded-xl border space-y-2">
                <div class="flex items-center justify-between">
                    <span class="text-[11px] uppercase tracking-wider theme-text-sub font-semibold flex items-center gap-1" data-i18n="skillsHeader">
                        🧠 Kỹ năng & Trí nhớ
                    </span>
                    <button onclick="openSkillsModal()" class="text-xs text-rose-400 hover:text-rose-300 font-semibold transition" data-i18n="manageBtn">
                        + Quản lý
                    </button>
                </div>
                <div id="sidebarSkillsList" class="space-y-1 max-h-28 overflow-y-auto text-xs"></div>
            </div>

            <!-- Quản lý Phiên (Sessions) -->
            <div class="space-y-1.5">
                <div class="flex items-center justify-between px-1">
                    <span class="text-[11px] uppercase tracking-wider theme-text-sub font-semibold" data-i18n="savedSessions">Lịch sử đã lưu (.json)</span>
                    <button onclick="saveSessionPrompt()" class="text-xs text-rose-400 hover:text-rose-300 font-medium" data-i18n="newSaveBtn">+ Lưu mới</button>
                </div>
                <div id="sessionList" class="space-y-1 max-h-32 overflow-y-auto text-xs"></div>
            </div>
        </div>

        <!-- Khung Quota & Nâng cấp Premium -->
        <div class="pt-3 border-t theme-border space-y-2">
            <div id="quotaBox" class="theme-card p-3 rounded-xl border">
                <div class="flex items-center justify-between mb-1.5">
                    <span id="planBadge" class="text-xs font-semibold px-2 py-0.5 rounded bg-amber-500/20 text-amber-400">⚡ FREE TIER</span>
                    <span id="quotaText" class="text-xs theme-text-sub">0 / 5 câu</span>
                </div>
                <div class="w-full bg-neutral-700/40 h-1.5 rounded-full overflow-hidden mb-2.5">
                    <div id="quotaBar" class="bg-gradient-to-r from-rose-500 to-amber-400 h-full transition-all duration-300" style="width: 0%"></div>
                </div>
                <div id="keyInputGroup" class="flex gap-1.5">
                    <input id="premiumKeyInput" type="password" placeholder="Nhập key Premium..."
                        class="w-full theme-input border rounded-lg px-2.5 py-1 text-xs focus:outline-none focus:border-rose-500">
                    <button onclick="unlockPremium()" class="px-2.5 py-1 bg-rose-600 hover:bg-rose-500 text-white rounded-lg text-xs font-medium shrink-0 transition shadow-sm" data-i18n="unlockBtn">Mở</button>
                </div>
            </div>
            <button onclick="exportMarkdown()" class="w-full py-2 px-3 rounded-xl theme-card hover:border-rose-500/50 text-xs flex items-center justify-center gap-1.5 border transition" data-i18n="exportMdBtn">
                📥 Xuất hội thoại ra Markdown (.md)
            </button>
        </div>
    </aside>

    <!-- KHU VỰC CHAT CHÍNH -->
    <main class="flex-1 flex flex-col theme-main relative select-text">
        <!-- Thanh Header: Chọn Model + Chọn Giao diện + Chọn Ngôn ngữ + Nút Thoát -->
        <header class="h-14 border-b theme-border px-6 flex items-center justify-between backdrop-blur z-10">
            <div class="flex items-center gap-2 sm:gap-3 overflow-x-auto">
                <span class="text-xs theme-text-sub font-medium hidden md:inline" data-i18n="modelLabel">Mô hình:</span>
                <select id="modelSelect" onchange="updateSettings()"
                    class="theme-card border text-xs sm:text-sm rounded-xl px-2.5 sm:px-3 py-1.5 focus:outline-none focus:border-rose-500 font-medium cursor-pointer">
                    <option value="gemini-3.7-flash">✨ Gemini 3.7 Flash (Thế hệ mới)</option>
                    <option value="gemini-3.6-flash">⚡ Gemini 3.6 Flash (Fast - v1.5)</option>
                    <option value="gemini-3.5-flash" selected>🚀 Gemini 3.5 Flash (Stable - v2.1)</option>
                    <option value="qwen/qwen3.8-27b">🔥 Qwen 3.8 27B qua Groq (Bản 1.0)</option>
                </select>

                <!-- BỘ CHỌN GIAO DIỆN -->
                <div class="flex items-center gap-1 pl-2 sm:pl-3 border-l theme-border">
                    <span class="text-xs theme-text-sub hidden lg:inline" data-i18n="themeLabel">Giao diện:</span>
                    <select id="themeSelect" onchange="changeTheme(this.value)"
                        class="theme-card border text-xs rounded-lg px-2 py-1.5 focus:outline-none focus:border-rose-500 font-medium cursor-pointer">
                        <option value="dark" data-i18n="themeDark">🌙 Tối (Dark)</option>
                        <option value="light" data-i18n="themeLight">☀️ Sáng (Light)</option>
                        <option value="contrast" data-i18n="themeContrast">⚡ Tương phản</option>
                    </select>
                </div>

                <!-- BỘ CHỌN NGÔN NGỮ (VI / EN) -->
                <div class="flex items-center gap-1 pl-2 sm:pl-3 border-l theme-border">
                    <span class="text-xs theme-text-sub hidden lg:inline" data-i18n="langLabel">Ngôn ngữ:</span>
                    <select id="langSelect" onchange="changeLanguage(this.value)"
                        class="theme-card border text-xs rounded-lg px-2 py-1.5 focus:outline-none focus:border-rose-500 font-medium cursor-pointer">
                        <option value="vi">🇻🇳 Tiếng Việt</option>
                        <option value="en">🇬🇧 English</option>
                    </select>
                </div>
            </div>

            <!-- Cụm nút góc trên bên phải -->
            <div class="flex items-center gap-1.5 sm:gap-2">
                <button onclick="openSkillsModal()" title="Kỹ năng & Trí thông minh cá nhân"
                    class="text-xs px-2.5 py-1.5 rounded-lg theme-card border hover:border-rose-500/60 transition flex items-center gap-1">
                    🧠 <span class="hidden md:inline" data-i18n="skillsNav">Trí nhớ & Skills</span>
                </button>
                <button onclick="undoLastTurn()" title="Thu hồi câu hỏi gần nhất"
                    class="text-xs px-2.5 py-1.5 rounded-lg theme-card border hover:border-rose-500/60 transition" data-i18n="undoBtn">
                    ↩ Hoàn tác
                </button>
                <button onclick="shutdownServer()" title="Tắt Server & Thoát"
                    class="text-xs px-2.5 py-1.5 rounded-lg bg-rose-500/15 hover:bg-rose-600 text-rose-400 hover:text-white border border-rose-500/30 font-semibold transition flex items-center gap-1" data-i18n="exitBtn">
                    ⏻ Thoát
                </button>
            </div>
        </header>

        <!-- Luồng tin nhắn -->
        <div id="chatContainer" class="flex-1 overflow-y-auto px-4 md:px-16 lg:px-32 py-6 space-y-6">
            <!-- Màn hình chào mừng mặc định -->
            <div id="welcomeScreen" class="h-full flex flex-col items-center justify-center text-center max-w-2xl mx-auto space-y-5 py-12">
                
                <!-- Logo LiemAI Premium 10 cánh vát khối đa diện -->
<div class="w-20 h-20 drop-shadow-2xl flex items-center justify-center">
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" class="w-full h-full">
        <defs>
            <radialGradient id="welcomeBgGrad" cx="50%" cy="45%" r="60%">
                <stop offset="0%" stop-color="#b91c1c"/>
                <stop offset="65%" stop-color="#88081f"/>
                <stop offset="100%" stop-color="#45030e"/>
            </radialGradient>
            <linearGradient id="wGoldL" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stop-color="#fff1b8"/><stop offset="100%" stop-color="#d4a034"/>
            </linearGradient>
            <linearGradient id="wGoldD" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stop-color="#ba8623"/><stop offset="100%" stop-color="#684507"/>
            </linearGradient>
            <linearGradient id="wObsL" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stop-color="#383a42"/><stop offset="100%" stop-color="#1f2127"/>
            </linearGradient>
            <linearGradient id="wObsD" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stop-color="#181a1f"/><stop offset="100%" stop-color="#0b0c0e"/>
            </linearGradient>
            <filter id="wShadow" x="-20%" y="-20%" width="140%" height="140%">
                <feDropShadow dx="0" dy="6" stdDeviation="6" flood-color="#000" flood-opacity="0.6"/>
            </filter>
        </defs>
        <rect width="256" height="256" rx="58" fill="url(#welcomeBgGrad)" stroke="#f43f5e" stroke-opacity="0.3" stroke-width="2"/>
        <g filter="url(#wShadow)" transform="translate(128,128)" stroke="#fef08a" stroke-width="1.2" stroke-linejoin="round">
            <!-- Ngôi sao 10 cánh Titan đen bên ngoài -->
            <path d="M0,-112 L0,0 L20.5,-39.5 Z" fill="url(#wObsL)"/>
            <path d="M0,-112 L0,0 L-20.5,-39.5 Z" fill="url(#wObsD)"/>
            <path d="M65.8,-90.6 L0,0 L43.9,-5.7 Z" fill="url(#wObsL)"/>
            <path d="M65.8,-90.6 L0,0 L20.5,-39.5 Z" fill="url(#wObsD)"/>
            <path d="M106.5,-34.6 L0,0 L43.9,-5.7 Z" fill="url(#wObsL)"/>
            <path d="M106.5,-34.6 L0,0 L35.6,26.7 Z" fill="url(#wObsD)"/>
            <path d="M106.5,34.6 L0,0 L35.6,26.7 Z" fill="url(#wObsL)"/>
            <path d="M106.5,34.6 L0,0 L0,44.3 Z" fill="url(#wObsD)"/>
            <path d="M65.8,90.6 L0,0 L0,44.3 Z" fill="url(#wObsL)"/>
            <path d="M65.8,90.6 L0,0 L-35.6,26.7 Z" fill="url(#wObsD)"/>
            <path d="M0,112 L0,0 L-35.6,26.7 Z" fill="url(#wObsL)"/>
            <path d="M0,112 L0,0 L-43.9,-5.7 Z" fill="url(#wObsD)"/>
            <path d="M-65.8,90.6 L0,0 L-43.9,-5.7 Z" fill="url(#wObsL)"/>
            <path d="M-65.8,90.6 L0,0 L-20.5,-39.5 Z" fill="url(#wObsD)"/>
            <path d="M-106.5,34.6 L0,0 L-20.5,-39.5 Z" fill="url(#wObsL)"/>
            <path d="M-106.5,34.6 L0,0 L-43.9,-5.7 Z" fill="url(#wObsD)"/>
            <path d="M-106.5,-34.6 L0,0 L-43.9,-5.7 Z" fill="url(#wObsL)"/>
            <path d="M-106.5,-34.6 L0,0 L-20.5,-39.5 Z" fill="url(#wObsD)"/>
            <path d="M-65.8,-90.6 L0,0 L-20.5,-39.5 Z" fill="url(#wObsL)"/>
            <path d="M-65.8,-90.6 L0,0 L0,-44.3 Z" fill="url(#wObsD)"/>

            <!-- Ngôi sao 10 cánh Vàng kim loại bên trong -->
            <g stroke="#fff" stroke-opacity="0.4" stroke-width="0.8">
                <path d="M0,-54 L0,0 L10.5,-18.2 Z" fill="url(#wGoldL)"/>
                <path d="M0,-54 L0,0 L-10.5,-18.2 Z" fill="url(#wGoldD)"/>
                <path d="M31.7,-43.7 L0,0 L20.9,-2.7 Z" fill="url(#wGoldL)"/>
                <path d="M31.7,-43.7 L0,0 L10.5,-18.2 Z" fill="url(#wGoldD)"/>
                <path d="M51.4,-16.7 L0,0 L20.9,-2.7 Z" fill="url(#wGoldL)"/>
                <path d="M51.4,-16.7 L0,0 L17.0,12.7 Z" fill="url(#wGoldD)"/>
                <path d="M51.4,16.7 L0,0 L17.0,12.7 Z" fill="url(#wGoldL)"/>
                <path d="M51.4,16.7 L0,0 L0,21.2 Z" fill="url(#wGoldD)"/>
                <path d="M31.7,43.7 L0,0 L0,21.2 Z" fill="url(#wGoldL)"/>
                <path d="M31.7,43.7 L0,0 L-17.0,12.7 Z" fill="url(#wGoldD)"/>
                <path d="M0,54 L0,0 L-17.0,12.7 Z" fill="url(#wGoldL)"/>
                <path d="M0,54 L0,0 L-20.9,-2.7 Z" fill="url(#wGoldD)"/>
                <path d="M-31.7,43.7 L0,0 L-20.9,-2.7 Z" fill="url(#wGoldL)"/>
                <path d="M-31.7,43.7 L0,0 L-10.5,-18.2 Z" fill="url(#wGoldD)"/>
                <path d="M-51.4,16.7 L0,0 L-10.5,-18.2 Z" fill="url(#wGoldL)"/>
                <path d="M-51.4,16.7 L0,0 L-20.9,-2.7 Z" fill="url(#wGoldD)"/>
                <path d="M-51.4,-16.7 L0,0 L-20.9,-2.7 Z" fill="url(#wGoldL)"/>
                <path d="M-51.4,-16.7 L0,0 L-10.5,-18.2 Z" fill="url(#wGoldD)"/>
                <path d="M-31.7,-43.7 L0,0 L-10.5,-18.2 Z" fill="url(#wGoldL)"/>
                <path d="M-31.7,-43.7 L0,0 L0,-21.2 Z" fill="url(#wGoldD)"/>
            </g>
            <circle cx="0" cy="0" r="3.5" fill="#fff6d1" stroke="#b45309" stroke-width="0.8"/>
        </g>
    </svg>
</div>

                <h2 class="text-2xl md:text-3xl font-bold bg-gradient-to-r from-rose-400 via-red-400 to-amber-300 bg-clip-text text-transparent">
                    <span data-i18n="welcomeGreeting">Xin chào</span> <span id="welcomeName">Lieman</span>, <span data-i18n="welcomeQuestion">hôm nay tôi có thể giúp gì cho bạn?</span>
                </h2>
                <p class="text-sm theme-text-sub" data-i18n="welcomeDesc">
                    LiemAI Lumique 3 tích hợp truyền tải trực tiếp (SSE Streaming), Trí thông minh cá nhân và chuyển đổi nóng giữa Gemini & Groq.
                </p>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-3 w-full text-left pt-2">
                    <button onclick="quickPromptByTag('oop')" class="p-3.5 rounded-xl theme-card hover:border-rose-500/60 border transition">
                        <div class="text-xs font-semibold text-rose-400 mb-1" data-i18n="card1Title">💻 Phân tích Code & OOP</div>
                        <div class="text-xs theme-text-sub" data-i18n="card1Desc">Kiểm tra cấu trúc hướng đối tượng và đề xuất cải tiến</div>
                    </button>
                    <button onclick="quickPromptByTag('novel')" class="p-3.5 rounded-xl theme-card hover:border-rose-500/60 border transition">
                        <div class="text-xs font-semibold text-amber-400 mb-1" data-i18n="card2Title">✒️ Sáng tác & Cốt truyện</div>
                        <div class="text-xs theme-text-sub" data-i18n="card2Desc">Phát triển kịch bản và chiều sâu tâm lý nhân vật</div>
                    </button>
                </div>
            </div>
        </div>

        <!-- Thanh công cụ nhập liệu dưới đáy -->
        <div class="px-4 md:px-16 lg:px-32 pb-5 pt-2">
            <div class="max-w-4xl mx-auto">
                <!-- Chip hiển thị file đính kèm -->
                <div id="attachedFileChip" class="hidden mb-2 inline-flex items-center gap-2 bg-rose-950/80 border border-rose-500/40 text-rose-200 px-3 py-1 rounded-xl text-xs">
                    <span>📎</span>
                    <span id="attachedFileName" class="font-medium"></span>
                    <button onclick="clearAttachment()" class="hover:text-red-400 ml-1 font-bold">✕</button>
                </div>

                <!-- Khung nhập tin nhắn -->
                <div class="theme-card border focus-within:border-rose-500 rounded-2xl p-2.5 flex items-end gap-2 shadow-xl">
                    <input type="file" id="fileInput" class="hidden" onchange="handleFileSelect(event)">
                    <button onclick="document.getElementById('fileInput').click()" title="Đính kèm file"
                        class="p-2 rounded-xl hover:bg-rose-500/10 theme-text-sub hover:text-rose-400 transition shrink-0">
                        📎
                    </button>

                    <textarea id="messageInput" rows="1" placeholder="Nhập tin nhắn cho LiemAI (Enter để gửi, Shift+Enter xuống dòng)..."
                        oninput="autoResize(this)" onkeydown="handleKeydown(event)"
                        class="w-full bg-transparent text-sm placeholder-slate-500 focus:outline-none resize-none max-h-44 py-1.5"></textarea>

                    <button id="stopBtn" onclick="stopGeneration()" class="hidden px-3 py-2 rounded-xl bg-red-600 hover:bg-red-500 text-white text-xs font-semibold shrink-0 transition" data-i18n="stopBtn">
                        ⏹ Dừng
                    </button>
                    <button id="sendBtn" onclick="sendMessage()" class="px-4 py-2 rounded-xl bg-gradient-to-r from-rose-600 to-red-500 hover:from-rose-500 hover:to-red-400 text-white text-xs font-semibold shrink-0 transition shadow-md shadow-rose-600/20" data-i18n="sendBtn">
                        Gửi ➤
                    </button>
                </div>
            </div>
        </div>
    </main>

    <!-- MODAL QUẢN LÝ KỸ NĂNG & TRÍ THÔNG MINH CÁ NHÂN -->
    <div id="skillsModal" class="hidden fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
        <div class="theme-card border rounded-2xl max-w-xl w-full p-5 space-y-4 shadow-2xl max-h-[90vh] flex flex-col">
            <div class="flex items-center justify-between border-b theme-border pb-3">
                <div class="flex items-center gap-2">
                    <span class="text-xl">🧠</span>
                    <div>
                        <h3 class="font-bold text-base" data-i18n="modalTitle">Kỹ năng & Trí thông minh cá nhân</h3>
                        <p class="text-xs theme-text-sub" data-i18n="modalSub">Tùy chỉnh bộ nhớ dài hạn và các kỹ năng chuyên môn cho LiemAI</p>
                    </div>
                </div>
                <button onclick="closeSkillsModal()" class="text-sm px-2.5 py-1 rounded-lg hover:bg-rose-500/20 hover:text-rose-400 transition">✕</button>
            </div>

            <div class="space-y-4 overflow-y-auto pr-1 flex-1">
                <!-- 1. Trí nhớ cá nhân -->
                <div>
                    <label class="text-xs font-semibold text-rose-400 uppercase tracking-wider block mb-1" data-i18n="sec1Title">
                        1. Trí thông minh cá nhân (Hồ sơ & Ngữ cảnh mặc định)
                    </label>
                    <textarea id="personalMemoryInput" rows="3" placeholder="Ví dụ: Tôi là sinh viên IT2 Bách Khoa, thích giải thích code bằng Java/Python ngắn gọn..."
                        class="w-full theme-input border rounded-xl p-3 text-xs focus:outline-none focus:border-rose-500"></textarea>
                    <div class="flex justify-end mt-1.5">
                        <button onclick="savePersonalMemory()" class="px-3 py-1.5 rounded-lg bg-rose-600 hover:bg-rose-500 text-white text-xs font-medium transition" data-i18n="saveMemoryBtn">
                            💾 Lưu Trí nhớ cá nhân
                        </button>
                    </div>
                </div>

                <!-- 2. Danh sách Kỹ năng (Skills) -->
                <div class="space-y-2">
                    <label class="text-xs font-semibold text-rose-400 uppercase tracking-wider block" data-i18n="sec2Title">
                        2. Danh sách Kỹ năng (Bật/Tắt linh hoạt)
                    </label>
                    <div id="modalSkillsList" class="space-y-2"></div>
                </div>

                <!-- 3. Thêm Kỹ năng mới -->
                <div class="p-3 rounded-xl border theme-border space-y-2">
                    <div class="text-xs font-semibold" data-i18n="sec3Title">＋ Thêm Kỹ năng (Skill) mới</div>
                    <input id="newSkillName" type="text" placeholder="Tên kỹ năng (VD: 🔥 Chuyên gia giải thuật C++)"
                        class="w-full theme-input border rounded-lg px-2.5 py-1.5 text-xs focus:outline-none focus:border-rose-500">
                    <textarea id="newSkillInstruction" rows="2" placeholder="Chỉ dẫn cho AI khi kỹ năng này bật (VD: Luôn phân tích độ phức tạp Time/Space Big-O...)"
                        class="w-full theme-input border rounded-lg px-2.5 py-1.5 text-xs focus:outline-none focus:border-rose-500"></textarea>
                    <button onclick="addNewSkill()" class="w-full py-1.5 rounded-lg bg-rose-500/20 hover:bg-rose-600 text-rose-300 hover:text-white border border-rose-500/30 text-xs font-semibold transition" data-i18n="addSkillBtn">
                        + Thêm vào bộ Kỹ năng
                    </button>
                </div>
            </div>
        </div>
    </div>

    <script>
        let attachedFile = { name: "", content: "" };
        let abortController = null;
        let currentLang = 'vi';

        // TỪ ĐIỂN SONG NGỮ (VIỆT - ANH) CHO TOÀN BỘ GIAO DIỆN
        const TRANSLATIONS = {
            vi: {
                newChat: "Cuộc trò chuyện mới",
                userNameLabel: "Tên của bạn ({name})",
                tempLabel: "ĐỘ SÁNG TẠO (TEMP)",
                autoSaveLabel: "Tự động lưu JSON",
                skillsHeader: "🧠 Kỹ năng & Trí nhớ",
                manageBtn: "+ Quản lý",
                savedSessions: "Lịch sử đã lưu (.json)",
                newSaveBtn: "+ Lưu mới",
                exportMdBtn: "📥 Xuất hội thoại ra Markdown (.md)",
                modelLabel: "Mô hình:",
                themeLabel: "Giao diện:",
                langLabel: "Ngôn ngữ:",
                themeDark: "🌙 Tối (Dark)",
                themeLight: "☀️ Sáng (Light)",
                themeContrast: "⚡ Tương phản",
                skillsNav: "Trí nhớ & Skills",
                undoBtn: "↩ Hoàn tác",
                exitBtn: "⏻ Thoát",
                welcomeGreeting: "Xin chào",
                welcomeQuestion: "hôm nay tôi có thể giúp gì cho bạn?",
                welcomeDesc: "LiemAI Lumique 3 tích hợp truyền tải trực tiếp (SSE Streaming), Trí thông minh cá nhân và chuyển đổi nóng giữa Gemini & Groq.",
                card1Title: "💻 Phân tích Code & OOP",
                card1Desc: "Kiểm tra cấu trúc hướng đối tượng và đề xuất cải tiến",
                card2Title: "✒️ Sáng tác & Cốt truyện",
                card2Desc: "Phát triển kịch bản và chiều sâu tâm lý nhân vật",
                inputPlaceholder: "Nhập tin nhắn cho LiemAI (Enter để gửi, Shift+Enter xuống dòng)...",
                stopBtn: "⏹ Dừng",
                sendBtn: "Gửi ➤",
                thinking: "Đang suy nghĩ...",
                modalTitle: "Kỹ năng & Trí thông minh cá nhân",
                modalSub: "Tùy chỉnh bộ nhớ dài hạn và các kỹ năng chuyên môn cho LiemAI",
                sec1Title: "1. Trí thông minh cá nhân (Hồ sơ & Ngữ cảnh mặc định)",
                sec1Placeholder: "Ví dụ: Tôi là sinh viên IT2 Bách Khoa, thích giải thích code bằng Java/Python ngắn gọn...",
                saveMemoryBtn: "💾 Lưu Trí nhớ cá nhân",
                sec2Title: "2. Danh sách Kỹ năng (Bật/Tắt linh hoạt)",
                sec3Title: "＋ Thêm Kỹ năng (Skill) mới",
                skillNamePlaceholder: "Tên kỹ năng (VD: 🔥 Chuyên gia giải thuật C++)",
                skillDescPlaceholder: "Chỉ dẫn cho AI khi kỹ năng này bật (VD: Luôn phân tích độ phức tạp Time/Space Big-O...)",
                addSkillBtn: "+ Thêm vào bộ Kỹ năng",
                unlockBtn: "Mở",
                keyPlaceholder: "Nhập key Premium...",
                skillActive: "Đang bật",
                skillInactive: "Đang tắt",
                btnDisable: "Tắt",
                btnEnable: "Bật",
                shutdownConfirm: "Bạn muốn tắt Server LiemAI Lumique 3 và thoát ứng dụng?",
                shutdownTitle: "Đã tắt Server LiemAI Lumique 3!",
                shutdownDesc: "Tiến trình Python đã kết thúc an toàn. Bạn có thể đóng tab trình duyệt này.",
                memorySavedAlert: "✔ Đã cập nhật Trí thông minh cá nhân!",
                keyWrongAlert: "Key không chính xác! (Gợi ý: nhập 'key')",
                missingSkillAlert: "Vui lòng nhập đầy đủ Tên kỹ năng và Chỉ dẫn!",
                promptNameSave: "Nhập tên file muốn lưu (.json):",
                promptOOP: "Hãy tóm tắt các điểm mạnh của kiến trúc OOP trong dự án của tôi.",
                promptNovel: "Hãy giúp tôi phác thảo diễn biến tiếp theo cho tiểu thuyết Biên Niên Sử Inependia."
            },
            en: {
                newChat: "New Conversation",
                userNameLabel: "Your Name ({name})",
                tempLabel: "CREATIVITY (TEMP)",
                autoSaveLabel: "Auto-save JSON",
                skillsHeader: "🧠 Skills & Memory",
                manageBtn: "+ Manage",
                savedSessions: "Saved Sessions (.json)",
                newSaveBtn: "+ New Save",
                exportMdBtn: "📥 Export to Markdown (.md)",
                modelLabel: "Model:",
                themeLabel: "Theme:",
                langLabel: "Language:",
                themeDark: "🌙 Dark",
                themeLight: "☀️ Light",
                themeContrast: "⚡ High Contrast",
                skillsNav: "Memory & Skills",
                undoBtn: "↩ Undo",
                exitBtn: "⏻ Exit",
                welcomeGreeting: "Hello",
                welcomeQuestion: "how can I assist you today?",
                welcomeDesc: "LiemAI Lumique 3 integrates real-time SSE streaming, Personal Intelligence, and hot-swapping between Gemini & Groq.",
                card1Title: "💻 OOP & Code Analysis",
                card1Desc: "Inspect OOP architecture and propose improvements",
                card2Title: "✒️ Storytelling & Novel",
                card2Desc: "Develop plot twists and character psychology",
                inputPlaceholder: "Type a message for LiemAI (Enter to send, Shift+Enter for newline)...",
                stopBtn: "⏹ Stop",
                sendBtn: "Send ➤",
                thinking: "Thinking...",
                modalTitle: "Custom Skills & Personal Intelligence",
                modalSub: "Customize long-term memory and specialized capabilities for LiemAI",
                sec1Title: "1. Personal Intelligence (Default Profile & Context)",
                sec1Placeholder: "E.g., I'm a CS sophomore at HUST, prefer concise Java/Python code explanations...",
                saveMemoryBtn: "💾 Save Personal Memory",
                sec2Title: "2. Custom Skills (Toggle ON/OFF)",
                sec3Title: "＋ Add New Skill",
                skillNamePlaceholder: "Skill name (e.g., 🔥 C++ Algorithm Expert)",
                skillDescPlaceholder: "Instructions for AI when active (e.g., Always analyze Big-O Time/Space complexity...)",
                addSkillBtn: "+ Add to Custom Skills",
                unlockBtn: "Unlock",
                keyPlaceholder: "Enter Premium key...",
                skillActive: "Active",
                skillInactive: "Inactive",
                btnDisable: "Disable",
                btnEnable: "Enable",
                shutdownConfirm: "Do you want to shutdown LiemAI Lumique 3 server and exit?",
                shutdownTitle: "LiemAI Lumique 3 Server Terminated!",
                shutdownDesc: "Python process has stopped safely. You can close this browser tab.",
                memorySavedAlert: "✔ Personal Intelligence updated successfully!",
                keyWrongAlert: "Invalid key! (Hint: enter 'key')",
                missingSkillAlert: "Please enter both skill name and instruction!",
                promptNameSave: "Enter filename to save (.json):",
                promptOOP: "Summarize the key strengths of OOP architecture in my project.",
                promptNovel: "Help me outline the next plot developments for the novel Inependia Chronicles."
            }
        };

        // HÀM ÁP DỤNG NGÔN NGỮ CHO GIAO DIỆN
        function changeLanguage(lang) {
            currentLang = lang;
            localStorage.setItem('liemai_lang', lang);
            const sel = document.getElementById('langSelect');
            if (sel && sel.value !== lang) sel.value = lang;

            const t = TRANSLATIONS[lang] || TRANSLATIONS.vi;
            document.querySelectorAll('[data-i18n]').forEach(el => {
                const key = el.getAttribute('data-i18n');
                if (t[key]) el.innerText = t[key];
            });

            // Cập nhật placeholders
            document.getElementById('messageInput').placeholder = t.inputPlaceholder;
            document.getElementById('personalMemoryInput').placeholder = t.sec1Placeholder;
            document.getElementById('newSkillName').placeholder = t.skillNamePlaceholder;
            document.getElementById('newSkillInstruction').placeholder = t.skillDescPlaceholder;
            document.getElementById('premiumKeyInput').placeholder = t.keyPlaceholder;

            // Đồng bộ sang Backend để AI biết ngữ cảnh ngôn ngữ
            updateSettings();
            refreshStatus();
        }

        (function initSavedLanguage() {
            const savedLang = localStorage.getItem('liemai_lang') || 'vi';
            currentLang = savedLang;
        })();

        marked.setOptions({
            highlight: function(code, lang) {
                if (lang && hljs.getLanguage(lang)) {
                    return hljs.highlight(code, { language: lang }).value;
                }
                return hljs.highlightAuto(code).value;
            },
            breaks: true
        });

        // QUẢN LÝ CHẾ ĐỘ GIAO DIỆN (DARK / LIGHT / CONTRAST)
        function changeTheme(themeName) {
            document.documentElement.setAttribute('data-theme', themeName);
            localStorage.setItem('liemai_theme', themeName);
            const sel = document.getElementById('themeSelect');
            if (sel && sel.value !== themeName) sel.value = themeName;
        }

        (function initSavedTheme() {
            const saved = localStorage.getItem('liemai_theme') || 'dark';
            changeTheme(saved);
        })();

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

        function quickPromptByTag(tag) {
            const t = TRANSLATIONS[currentLang] || TRANSLATIONS.vi;
            const input = document.getElementById('messageInput');
            input.value = (tag === 'oop') ? t.promptOOP : t.promptNovel;
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
                    <div class="max-w-2xl theme-bubble-user border rounded-2xl rounded-tr-sm px-4 py-3 text-sm whitespace-pre-wrap">${escapeHtml(htmlContent)}</div>
                `;
            } else {
                wrapper.className = "flex gap-3 items-start max-w-4xl";
                wrapper.innerHTML = `
                    <div class="w-8 h-8 rounded-xl bg-gradient-to-tr from-red-600 via-rose-500 to-amber-400 flex items-center justify-center text-xs font-bold text-white shrink-0 mt-0.5 shadow-md shadow-rose-500/20">AI</div>
                    <div class="flex-1 theme-bubble-ai border rounded-2xl rounded-tl-sm px-5 py-3.5 text-sm prose max-w-none message-content">${htmlContent}</div>
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

            const t = TRANSLATIONS[currentLang] || TRANSLATIONS.vi;
            let displayMsg = attachedFile.name ? `📎 [File: ${attachedFile.name}]\n${text}` : text;
            createMessageBubble('user', displayMsg);

            const payload = {
                message: text || (currentLang === 'en' ? "Please analyze this attached file." : "Hãy phân tích nội dung file đính kèm này."),
                file_name: attachedFile.name,
                file_content: attachedFile.content
            };

            input.value = "";
            input.style.height = 'auto';
            clearAttachment();

            const aiBubble = createMessageBubble('assistant', `<span class="theme-text-sub animate-pulse">${t.thinking}</span>`);
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
                                aiBubble.innerHTML = `<span class="text-rose-400 font-semibold">✖ ${escapeHtml(data.error)}</span>`;
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
                    fullMarkdown += (currentLang === 'en') ? "\n\n*(Generation stopped)*" : "\n\n*(Đã dừng sinh văn bản)*";
                    aiBubble.innerHTML = marked.parse(fullMarkdown);
                } else {
                    aiBubble.innerHTML = `<span class="text-rose-400">✖ Lỗi kết nối: ${err.message}</span>`;
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
                body: JSON.stringify({ user_name, model_id, temperature, is_persistent, language: currentLang })
            });
            refreshStatus();
        }

        async function unlockPremium() {
            const key = document.getElementById('premiumKeyInput').value.trim();
            const t = TRANSLATIONS[currentLang] || TRANSLATIONS.vi;
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
                alert(t.keyWrongAlert);
            }
        }

        // QUẢN LÝ SKILLS & PERSONAL INTELLIGENCE
        function openSkillsModal() {
            document.getElementById('skillsModal').classList.remove('hidden');
            refreshStatus();
        }

        function closeSkillsModal() {
            document.getElementById('skillsModal').classList.add('hidden');
        }

        async function savePersonalMemory() {
            const memory = document.getElementById('personalMemoryInput').value;
            const t = TRANSLATIONS[currentLang] || TRANSLATIONS.vi;
            await fetch('/api/intelligence/memory', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ personal_memory: memory })
            });
            refreshStatus();
            alert(t.memorySavedAlert);
        }

        async function toggleSkill(skillId) {
            await fetch('/api/intelligence/toggle_skill', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ id: skillId })
            });
            refreshStatus();
        }

        async function deleteSkill(skillId) {
            await fetch('/api/intelligence/delete_skill', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ id: skillId })
            });
            refreshStatus();
        }

        async function addNewSkill() {
            const name = document.getElementById('newSkillName').value.trim();
            const instruction = document.getElementById('newSkillInstruction').value.trim();
            const t = TRANSLATIONS[currentLang] || TRANSLATIONS.vi;
            if (!name || !instruction) {
                alert(t.missingSkillAlert);
                return;
            }
            await fetch('/api/intelligence/add_skill', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, instruction })
            });
            document.getElementById('newSkillName').value = "";
            document.getElementById('newSkillInstruction').value = "";
            refreshStatus();
        }

        // NÚT TẮT SERVER TRỰC TIẾP
        async function shutdownServer() {
            const t = TRANSLATIONS[currentLang] || TRANSLATIONS.vi;
            if (!confirm(t.shutdownConfirm)) return;
            try {
                await fetch('/api/shutdown', { method: 'POST' });
            } catch (e) {}
            document.body.innerHTML = `
                <div class="h-screen w-screen flex flex-col items-center justify-center theme-main text-center p-6 space-y-3">
                    <div class="w-14 h-14 rounded-2xl bg-gradient-to-tr from-red-600 via-rose-500 to-amber-400 flex items-center justify-center text-2xl text-white shadow-xl">👋</div>
                    <h2 class="text-xl font-bold">${t.shutdownTitle}</h2>
                    <p class="text-sm theme-text-sub">${t.shutdownDesc}</p>
                </div>
            `;
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
            const t = TRANSLATIONS[currentLang] || TRANSLATIONS.vi;
            const name = prompt(t.promptNameSave, `session_${document.getElementById('userNameInput').value}.json`);
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
                createMessageBubble(m.role, m.role === 'user' ? m.content : marked.parse(m.content));
            }
            refreshStatus();
        }

        function exportMarkdown() {
            window.location.href = '/api/export_md';
        }

        async function refreshStatus() {
            const res = await fetch('/api/status');
            const data = await res.json();
            const t = TRANSLATIONS[currentLang] || TRANSLATIONS.vi;

            if (data.is_premium) {
                document.getElementById('planBadge').className = "text-xs font-semibold px-2 py-0.5 rounded bg-rose-500/20 text-rose-400";
                document.getElementById('planBadge').innerText = "👑 PREMIUM";
                document.getElementById('quotaText').innerText = (currentLang === 'en') ? "Unlimited" : "Không giới hạn";
                document.getElementById('quotaBar').style.width = "100%";
                document.getElementById('quotaBar').className = "bg-gradient-to-r from-red-600 via-rose-500 to-amber-400 h-full";
                document.getElementById('keyInputGroup').classList.add('hidden');
            } else {
                const pct = Math.min((data.num_of_questions / data.question_limit) * 100, 100);
                const suffix = (currentLang === 'en') ? "prompts" : "câu";
                document.getElementById('quotaText').innerText = `${data.num_of_questions} / ${data.question_limit} ${suffix}`;
                document.getElementById('quotaBar').style.width = `${pct}%`;
            }

            // Render danh sách file JSON
            const sList = document.getElementById('sessionList');
            sList.innerHTML = data.sessions.map(f => `
                <button onclick="loadSession('${f}')" class="w-full text-left px-2.5 py-1.5 rounded-lg hover:bg-rose-500/10 truncate flex items-center gap-2 transition">
                    <span class="text-rose-400">💬</span> <span class="truncate">${f}</span>
                </button>
            `).join('');

            // Cập nhật giá trị vào modal trí nhớ
            const memInput = document.getElementById('personalMemoryInput');
            if (memInput && document.activeElement !== memInput) {
                memInput.value = data.personal_memory || "";
            }

            // Render Skills ở Sidebar
            const sideSkills = document.getElementById('sidebarSkillsList');
            sideSkills.innerHTML = data.skills.map(s => `
                <div onclick="toggleSkill('${s.id}')" class="flex items-center justify-between px-2 py-1 rounded-lg cursor-pointer hover:bg-rose-500/10 transition">
                    <span class="truncate ${s.enabled ? 'font-medium text-rose-400' : 'theme-text-sub line-through'}">${escapeHtml(s.name)}</span>
                    <span class="text-[10px] px-1.5 py-0.5 rounded ${s.enabled ? 'bg-rose-500/20 text-rose-300' : 'bg-neutral-700/30 theme-text-sub'}">${s.enabled ? 'ON' : 'OFF'}</span>
                </div>
            `).join('');

            // Render Skills trong Modal
            const modalSkills = document.getElementById('modalSkillsList');
            if (modalSkills) {
                modalSkills.innerHTML = data.skills.map(s => `
                    <div class="p-2.5 rounded-xl border theme-border flex items-start justify-between gap-2">
                        <div class="flex-1">
                            <div class="text-xs font-semibold flex items-center gap-2">
                                <span>${escapeHtml(s.name)}</span>
                                <span class="text-[10px] px-1.5 py-0.5 rounded ${s.enabled ? 'bg-rose-500/20 text-rose-400' : 'bg-neutral-700/30 theme-text-sub'}">${s.enabled ? t.skillActive : t.skillInactive}</span>
                            </div>
                            <p class="text-[11px] theme-text-sub mt-1">${escapeHtml(s.instruction)}</p>
                        </div>
                        <div class="flex items-center gap-1 shrink-0">
                            <button onclick="toggleSkill('${s.id}')" class="px-2 py-1 rounded-lg text-xs border theme-border hover:border-rose-500 transition">
                                ${s.enabled ? t.btnDisable : t.btnEnable}
                            </button>
                            <button onclick="deleteSkill('${s.id}')" class="px-2 py-1 rounded-lg text-xs text-rose-400 hover:bg-rose-500/20 transition">🗑</button>
                        </div>
                    </div>
                `).join('');
            }
        }

        window.onload = function() {
            changeLanguage(currentLang);
        };
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
    json_files = [
        f for f in sorted(glob.glob("*.json"))
        if f != LiemConfig.INTELLIGENCE_FILE
    ]
    return jsonify({
        "user_name": state.user_name,
        "model_id": state.model_id,
        "temperature": state.temperature,
        "language": state.language,
        "is_premium": state.is_premium,
        "num_of_questions": state.num_of_questions,
        "question_limit": LiemConfig.QUESTION_LIMIT,
        "sessions": json_files,
        "history": state.history,
        "personal_memory": state.personal_memory,
        "skills": state.skills
    })


@app.route("/api/settings", methods=["POST"])
def api_settings():
    data = request.json or {}
    state.user_name = data.get("user_name", state.user_name)
    state.model_id = data.get("model_id", state.model_id)
    state.temperature = float(data.get("temperature", state.temperature))
    state.is_persistent = bool(data.get("is_persistent", state.is_persistent))
    state.language = data.get("language", state.language)
    return jsonify({"success": True})


@app.route("/api/intelligence/memory", methods=["POST"])
def api_save_memory():
    data = request.json or {}
    state.personal_memory = data.get("personal_memory", "")
    state.save_intelligence()
    return jsonify({"success": True})


@app.route("/api/intelligence/add_skill", methods=["POST"])
def api_add_skill():
    data = request.json or {}
    name = data.get("name", "").strip()
    instruction = data.get("instruction", "").strip()
    if name and instruction:
        new_id = f"skill_{int(datetime.now().timestamp() * 1000)}"
        state.skills.append({
            "id": new_id,
            "name": name,
            "instruction": instruction,
            "enabled": True
        })
        state.save_intelligence()
    return jsonify({"success": True})


@app.route("/api/intelligence/toggle_skill", methods=["POST"])
def api_toggle_skill():
    skill_id = (request.json or {}).get("id", "")
    for s in state.skills:
        if s["id"] == skill_id:
            s["enabled"] = not s.get("enabled", False)
            break
    state.save_intelligence()
    return jsonify({"success": True})


@app.route("/api/intelligence/delete_skill", methods=["POST"])
def api_delete_skill():
    skill_id = (request.json or {}).get("id", "")
    state.skills = [s for s in state.skills if s["id"] != skill_id]
    state.save_intelligence()
    return jsonify({"success": True})


@app.route("/api/shutdown", methods=["POST"])
def api_shutdown():
    threading.Timer(0.4, lambda: os._exit(0)).start()
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
        f"# Lịch sử trò chuyện LiemAI Lumique 3 - {state.user_name}",
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
        limit_msg = (
            f"You have reached the {LiemConfig.QUESTION_LIMIT} free prompts limit! Please enter the Lumique Premium key in the bottom-left corner."
            if state.language == "en"
            else f"Bạn đã hết {LiemConfig.QUESTION_LIMIT} lượt hỏi miễn phí! Vui lòng nhập mã khóa Lumique Premium ở góc trái màn hình."
        )
        return jsonify({"error": limit_msg}), 403

    data = request.json or {}
    user_msg = data.get("message", "")
    file_name = data.get("file_name", "")
    file_content = data.get("file_content", "")

    if file_name and file_content:
        full_prompt = f"[File: {file_name}]\n```\n{file_content}\n```\n\n{user_msg}"
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