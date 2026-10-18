Try my first Vietnamese rebadged Chatbot (")>

# ✨ LiemAI — Evolution of a Custom AI Assistant (CLI & Lumique 3 Web Studio)

> An open-source showcase tracking the architectural evolution of **LiemAI** (created by **Lương Minh Quân**) — from a minimalist command-line prototype (`v1.0`) to an Object-Oriented Multi-Model Terminal Studio (**LiemAI 3.0**) and a full-stack Dark-Mode Web Interface (**LiemAI Lumique 3**).

---

## 🚀 Architectural Evolution Roadmap

| Version | Edition Name | Core Architecture & Milestone Features |
| :--- | :--- | :--- |
| **v1.0** | `LiemAI 1.0` | Initial CLI prototype powered by Groq; basic request/response loop and trial question limit. |
| **v1.5** | `LiemAI 1.5` | Migrated to Gemini API; added multiline input (`input_handler.py`) and persistent `systemInstruction`. |
| **v2.0** | `LiemAI 2.0` | Real-time SSE Streaming, `rich` Markdown terminal UI, local file inspection (`/file`), and JSON session persistence (`/save`, `/load`, `/reset`). |
| **v2.1** | `LiemAI 2.1 Demo` | Unified event loop, HTTP Header API key security, Sliding Window context (`MAX_CONTEXT_TURNS`), and runtime `/key` & `/status` commands. |
| **v3.0 (CLI)** | **`LiemAI 3.0`** | **OOP Terminal Studio:** 4-class modular OOP design, Dual-Engine hot-swapping (Gemini + Groq), TCP connection pooling (`requests.Session`), `shlex` file path parser, `/undo`, `/history`, and Markdown report exporting (`/export`). |
| **v3.0 (Web)** | **`LiemAI Lumique 3`** | **Full-Stack Web Studio:** Modern browser GUI (Flask + Tailwind CSS + Marked.js + Highlight.js) in a sleek Soft Rose palette, featuring 3-mode theme switching (**🌙 Dark / ☀️ Light / ⚡ High Contrast**), **Personal Intelligence & Custom Skills** management, 1-click file attachments (`📎`), and a direct header shutdown button (`⏻ Exit`). |

---

## 🖥️ Dual-Edition Overview

### 1. LiemAI Lumique 3 (Web Studio Edition)
Designed for a seamless, modern browser experience similar to leading web AI studios:
* **Direct Header Shutdown Button (`⏻ Exit`):** A compact button in the top-right header that cleanly terminates the Flask background process and Python runtime without touching the terminal.
* **3-Mode Dynamic Theming:** Switch seamlessly between **🌙 Dark**, **☀️ Light (soft & eye-friendly)**, and **⚡ High Contrast** modes, all styled in a refined Soft Rose accent palette.
* **Personal Intelligence & Custom Skills:** Define a persistent personal profile and toggle/manage modular skills (OOP & Clean Code, Novel Writing, Concise Answers, etc.) saved locally in `liemai_intelligence.json`.
* **1-Click File Attachments (`📎`):** Pick local files (`.java`, `.py`, `.md`, `.txt`, etc.) directly via native file explorer with live size badges before sending.
* **Real-Time SSE Streaming & Session Tools:** Smooth token streaming, automatic syntax highlighting, visual Free Tier quota meter (`0/5`), instant 👑 PREMIUM unlock, and 1-click Markdown (`.md`) transcript export.

### 2. LiemAI 3.0 (Terminal Studio - CLI Edition)
Built for developers who love fast, keyboard-centric terminal workflows:
* **Object-Oriented Architecture:** Clean separation of concerns across `LiemConfig`, `SessionManager`, `AIClient`, and `LiemAIApp`.
* **Rich TUI Dashboard:** Live status bar displaying active user, current model engine, temperature, context turns, auto-save state, and account tier.
* **Graceful Stream Interruption:** Press `Ctrl + C` mid-generation to stop a lengthy response cleanly without crashing the application.
* **Smart Sliding Context Window:** Automatically trims old messages while preserving strict `user`/`assistant` role alignment.

---

## 🛠️ Installation & Quick Start

### 1. Environment Configuration (`.env`)
Create a `.env` file in the project root directory with your API credentials:

```env
GEMINI_API_KEY_V20=your_google_gemini_api_key_here
API_KEY_V10=your_groq_api_key_here
```

### 2. System Prompt Configuration (`prompt_v30.txt`)
Create a `prompt_v30.txt` file (or `prompt_v20.txt`) in the root directory to define LiemAI's identity, behavioral rules, and creator lore. 

> **Important:** Always include the `{name}` placeholder where you want the user's name to be dynamically injected at runtime.

Example template for `prompt_v30.txt`:
```text
You are LiemAI — a witty, intelligent, and highly capable AI assistant created by Lương Minh Quân.
Chatting with: {name}.

Creator Profile:
- Name: Lương Minh Quân (Born March 21, 2007, Hải Phòng).
- Background: Alumnus of THPT Chuyên Trần Phú; Sophomore in Computer Engineering (IT2) at Hanoi University of Science and Technology (HUST).
- Role: creator of LiemAI.

Strict Rules:
- Language: Match {name}'s language automatically (English or Vietnamese).
- Tone: Natural, confident, witty, and loyal to creator Quân.
- Response: Direct, structured Markdown formatting with no fluff.
```
*(If omitted, LiemAI will automatically fall back to its built-in default system prompt).*

### 3. Launch Your Preferred Edition

* **To launch LiemAI Lumique 3 (Web Studio):**
  ```bash
  python LiemAI_Lumique_3.py
  ```
  *(Your default browser will automatically open at `[http://127.0.0.1:5000](http://127.0.0.1:5000)`)*

* **To launch LiemAI 3.0 (Terminal CLI):**
  ```bash
  python LiemAI_3.0.py
  ```


⌨️ LiemAI 3.0 Integrated Slash Commands (CLI)

Execute these commands directly inside the terminal prompt:

| Command | Syntax | Description |
| :--- | :--- | :--- |
| **Attach File** | `/file "<path>" [prompt]` | Reads a local file (supports spaces in paths) and attaches its content to your query. |
| **Switch Model** | `/model` | Hot-swaps between Gemini 4 Argon, Gemini 3.6/3.5 Flash, and Groq Qwen 3.8 27B. |
| **Temperature** | `/temp <0.0 - 2.0>` | Adjusts response creativity in real time. |
| **Undo Turn** | `/undo` | Removes the most recent question-and-answer pair from active memory. |
| **View Memory** | `/history` | Displays a formatted summary table of all messages in the current context window. |
| **Export Report** | `/export [filename.md]` | Exports the full conversation history into a clean Markdown document. |
| **Save Session** | `/save [filename.json]` | Saves the active conversation state into a JSON file. |
| **Load Session** | `/load <filename.json>` | Restores a previously saved JSON session (backward-compatible with v2.0/v2.1). |
| **Unlock Pro** | `/key [secret_key]` | Upgrades the current session from Free Trial to unlimited PREMIUM. |
| **System Status** | `/status` / `/help` | Displays the status bar or opens the interactive command reference table. |
| **Clear Memory** | `/reset` | Flushes context memory to start a fresh conversation. |
| **Safe Exit** | `esc` or `throat` | Gracefully closes the session. |
### ⚠️ Disclaimer & Acknowledgments

#### Disclaimer
> **Notice:** *LiemAI is an independent, non-commercial project developed by Lương Minh Quân solely for educational, research, and personal experimentation purposes.*

* This software is distributed on an **"AS IS" basis**, without warranties or conditions of any kind, either express or implied.
* The author assumes no liability or responsibility for API billing, rate limits, token consumption, or any outputs/hallucinations generated by the underlying AI models.
* Users are responsible for managing their own API credentials and adhering to the respective provider's terms of service.

#### Special Thanks
* Heartfelt gratitude to **Google** and the **Google AI Studio** team for providing the high-performance Gemini models and streaming infrastructure powering LiemAI.
* Big thanks to **Groq** for delivering the lightning-fast LPU inference engine that powered the original LiemAI 1.0 prototype and the Dual-Engine mode in v3.0.
* Appreciation to the open-source communities behind Python and web libraries including `rich`, `flask`, `requests`, `python-dotenv`, **Tailwind CSS**, **Marked.js**, and **Highlight.js**.

---
