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
| **v3.0 (Web)** | **`LiemAI Lumique 3`** | **Full-Stack Web Studio:** Modern browser-based GUI (Flask + Tailwind CSS + Marked.js + Highlight.js) with 1-click file attachments (`📎`), live streaming bubbles, interactive temperature slider, visual quota bar, and sidebar session manager. |

---

## 🖥️ Dual-Edition Overview

### 1. LiemAI Lumique 3 (Web Studio Edition)
Designed for a seamless, modern browser experience similar to leading web AI studios:
* **Real-Time Streaming Bubbles:** Instant token-by-token Markdown rendering with syntax-highlighted code blocks.
* **1-Click File Attachment (`📎`):** Attach source code (`.py`, `.java`, `.cpp`) or documents (`.md`, `.txt`, `.json`) directly from your file explorer without typing paths.
* **Hot-Swappable Model Header & Controls:** Switch between models and fine-tune creativity (`Temperature`) via live UI controls.
* **Sidebar Session & Export Manager:** Create new chats, load previous `.json` sessions with a single click, undo the last turn (`↩ Hoàn tác`), or download the entire conversation as a formatted `.md` report.
* **Visual Quota & Instant Premium Unlock:** Track Free Tier usage on a live progress bar and unlock **👑 PREMIUM** directly from the sidebar.

### 2. LiemAI 3.0 (Terminal Studio - CLI Edition)
Built for developers who love fast, keyboard-centric terminal workflows:
* **Object-Oriented Architecture:** Clean separation of concerns across `LiemConfig`, `SessionManager`, `AIClient`, and `LiemAIApp`.
* **Rich TUI Dashboard:** Live status bar displaying active user, current model engine, temperature, context turns, auto-save state, and account tier.
* **Graceful Stream Interruption:** Press `Ctrl + C` mid-generation to stop a lengthy response cleanly without crashing the application.
* **Smart Sliding Context Window:** Automatically trims old messages while preserving strict `user`/`assistant` role alignment.

---

## 🛠️ Installation & Quick Start

### 1. Prerequisites & Dependencies
Clone the repository and install the required Python packages:

```bash
pip install requests python-dotenv rich flask
```

### 2. Environment Configuration ( .env )

Create a `.env` file in the root directory with your API credentials:

GEMINI_API_KEY_V20=your_google_gemini_api_key_here
API_KEY_V10=your_groq_api_key_here

### 3. Launch Your Preferred Edition

* To launch LiemAI Lumique 3 (Web UI):
  ```bash
  python LiemAI_Lumique_3.py
  ```

  (Your default browser will automatically open at [http://127.0.0.1:5000](http://127.0.0.1:5000))

*  To launch LiemAI 3.0 (Terminal CLI)
```
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

> **💡 Mẹo nhỏ để đồng bộ tên gọi `LiemAI Lumique 3` trên bản Web:**
> *Trong file code Web ở lượt trước, bạn chỉ cần đổi tên file từ `LiemAI_3.0_Web.py` thành `LiemAI_Lumique_3.py` là được.*