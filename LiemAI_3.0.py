import json
import os
import shlex
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.status import Status
from rich.table import Table
from rich.text import Text

# Tương thích hoàn toàn với module input_handler.py sẵn có trong dự án
try:
    from input_handler import get_multiline_input
except ImportError:
    def get_multiline_input(prompt: str = "") -> str:
        return input(prompt).strip()

load_dotenv()
console = Console()


# =====================================================================
# 1. CLASS CẤU HÌNH HỆ THỐNG (CONFIG & MODELS REGISTRY)
# =====================================================================
class LiemConfig:
    APP_VERSION = "3.0"
    MAX_CONTEXT_TURNS = 16       # Lưu 8 cặp hỏi-đáp gần nhất
    QUESTION_LIMIT = 5           # Giới hạn bản Free
    SECRET_KEY = "key"           # Key kích hoạt Premium
    DEFAULT_PROMPT_FILE = "prompt_v30.txt"

    # Tự động đọc các key sẵn có từ file .env của bạn
    GEMINI_API_KEY = (
        os.getenv("GEMINI_API_KEY_V20")
        or os.getenv("GEMINI_API_KEY_V15")
        or os.getenv("GEMINI_API_KEY")
    )
    GROQ_API_KEY = os.getenv("API_KEY_V10") or os.getenv("GROQ_API_KEY")

    # Danh mục các Model từ bản 1.0 -> 2.1 để chuyển đổi tức thì
    MODELS: Dict[str, Dict[str, str]] = {
        "1": {
            "id": "gemini-4-argon",
            "provider": "gemini",
            "name": "Gemini 4 Argon (Siêu trí tuệ - Bản 2.0)",
        },
        "2": {
            "id": "gemini-3.6-flash",
            "provider": "gemini",
            "name": "Gemini 3.6 Flash (Tốc độ cao - Bản 1.5)",
        },
        "3": {
            "id": "gemini-3.5-flash",
            "provider": "gemini",
            "name": "Gemini 3.5 Flash (Ổn định - Bản 2.1)",
        },
        "4": {
            "id": "qwen/qwen3.8-27b",
            "provider": "groq",
            "name": "Qwen 3.8 27B qua Groq (Siêu tốc - Bản 1.0)",
        },
    }


# =====================================================================
# 2. CLASS QUẢN LÝ PHIÊN & LỊCH SỬ HỘI THOẠI (SESSION MANAGER)
# =====================================================================
class SessionManager:
    """Quản lý bộ nhớ ngữ cảnh, lưu/nạp JSON và xuất báo cáo Markdown."""

    def __init__(self, user_name: str, is_persistent: bool = False):
        self.user_name = user_name
        self.is_persistent = is_persistent
        self.session_file = f"session_{user_name}.json" if is_persistent else ""
        # Chuẩn hóa lưu trữ nội bộ: [{"role": "user"|"assistant", "content": "...", "time": "HH:MM"}]
        self.history: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        self.history.append({
            "role": role,
            "content": content,
            "time": datetime.now().strftime("%H:%M:%S")
        })
        if self.is_persistent and self.session_file and role == "assistant":
            self.save_to_json(self.session_file, silent=True)

    def pop_last_user_message(self) -> None:
        if self.history and self.history[-1]["role"] == "user":
            self.history.pop()

    def undo_last_turn(self) -> bool:
        """Xóa cặp hỏi - đáp gần nhất (Tính năng mới ở 3.0)."""
        if len(self.history) >= 2 and self.history[-1]["role"] == "assistant":
            self.history.pop()
            self.history.pop()
            return True
        elif self.history:
            self.history.pop()
            return True
        return False

    def clear(self) -> None:
        self.history.clear()

    def get_trimmed_history(self, max_turns: int) -> List[Dict[str, str]]:
        """Lấy cửa sổ ngữ cảnh trượt, đảm bảo luôn bắt đầu bằng role 'user'."""
        sliced = self.history[-max_turns:] if len(self.history) > max_turns else list(self.history)
        while sliced and sliced[0]["role"] != "user":
            sliced.pop(0)
        return sliced

    def to_gemini_contents(self, max_turns: int) -> List[Dict]:
        """Chuyển đổi lịch sử sang định dạng chuẩn của Google Gemini API."""
        gemini_list = []
        for msg in self.get_trimmed_history(max_turns):
            role = "user" if msg["role"] == "user" else "model"
            gemini_list.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })
        return gemini_list

    def to_groq_messages(self, sys_prompt: str, max_turns: int) -> List[Dict]:
        """Chuyển đổi lịch sử sang định dạng chuẩn của OpenAI / Groq API."""
        messages = [{"role": "system", "content": sys_prompt}]
        for msg in self.get_trimmed_history(max_turns):
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        return messages

    def save_to_json(self, filename: str, silent: bool = False) -> None:
        if not filename.endswith(".json"):
            filename += ".json"
        try:
            # Lưu theo định dạng tương thích ngược với LiemAI 2.0/2.1
            export_data = [
                {
                    "role": "user" if m["role"] == "user" else "model",
                    "parts": [{"text": m["content"]}],
                    "time": m.get("time", "")
                }
                for m in self.history
            ]
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            if not silent:
                console.print(f"[bold green]✔ Đã lưu hội thoại vào '{filename}'[/bold green]")
        except Exception as e:
            console.print(f"[bold red]✖ Lỗi lưu file: {e}[/bold red]")

    def load_from_json(self, filename: str) -> None:
        if not filename.endswith(".json"):
            filename += ".json"
        if not os.path.isfile(filename):
            console.print(f"[bold red]✖ Không tìm thấy file '{filename}'[/bold red]")
            return
        try:
            with open(filename, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            loaded_history = []
            for item in raw_data:
                # Hỗ trợ đọc cả file JSON từ bản 2.0/2.1 (parts) lẫn chuẩn OpenAI (content)
                role_raw = item.get("role", "user")
                role = "user" if role_raw == "user" else "assistant"
                if "parts" in item and item["parts"]:
                    content = item["parts"][0].get("text", "")
                else:
                    content = item.get("content", "")
                loaded_history.append({
                    "role": role,
                    "content": content,
                    "time": item.get("time", "--:--")
                })
            self.history = loaded_history
            console.print(f"[bold green]✔ Đã nạp {len(self.history)} tin nhắn từ '{filename}' thành công![/bold green]")
        except Exception as e:
            console.print(f"[bold red]✖ Lỗi nạp file: {e}[/bold red]")

    def export_to_markdown(self, filename: str) -> None:
        """Tính năng mới: Xuất toàn bộ cuộc trò chuyện ra file Markdown (.md) để đọc/nộp báo cáo."""
        if not filename.endswith(".md"):
            filename += ".md"
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"# Lịch sử trò chuyện LiemAI 3.0 - Người dùng: {self.user_name}\n")
                f.write(f"*Xuất lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n---\n\n")
                for m in self.history:
                    sender = f"🧑 **{self.user_name}**" if m["role"] == "user" else "🤖 **LiemAI**"
                    f.write(f"### {sender} `[{m.get('time', '')}]`\n\n{m['content']}\n\n---\n\n")
            console.print(f"[bold green]✔ Đã xuất bản ghi Markdown ra file '{filename}'![/bold green]")
        except Exception as e:
            console.print(f"[bold red]✖ Lỗi xuất Markdown: {e}[/bold red]")


# =====================================================================
# 3. CLASS GIAO TIẾP API ĐA NỀN TẢNG (AI CLIENT - GEMINI & GROQ)
# =====================================================================
class AIClient:
    """Xử lý kết nối HTTP bền vững (Connection Pooling) và Streaming cho cả Gemini & Groq."""

    def __init__(self):
        self.http_session = requests.Session()
        self.temperature = 0.7
        self.current_model_key = "3"  # Mặc định dùng Gemini 3.5 Flash (như bản 2.1)

    @property
    def current_model_info(self) -> Dict[str, str]:
        return LiemConfig.MODELS[self.current_model_key]

    def stream_chat(self, session: SessionManager, sys_prompt: str) -> Tuple[bool, str, float]:
        model_info = self.current_model_info
        provider = model_info["provider"]
        model_id = model_info["id"]

        if provider == "gemini":
            return self._stream_gemini(model_id, session, sys_prompt)
        else:
            return self._stream_groq(model_id, session, sys_prompt)

    def _stream_gemini(self, model_id: str, session: SessionManager, sys_prompt: str) -> Tuple[bool, str, float]:
        if not LiemConfig.GEMINI_API_KEY:
            console.print("[bold red]✖ Thiếu GEMINI_API_KEY trong file .env![/bold red]")
            return False, "", 0.0

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:streamGenerateContent?alt=sse"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": LiemConfig.GEMINI_API_KEY
        }
        data = {
            "contents": session.to_gemini_contents(LiemConfig.MAX_CONTEXT_TURNS),
            "systemInstruction": {"parts": [{"text": sys_prompt}]},
            "generationConfig": {"temperature": self.temperature}
        }
        return self._execute_sse_stream(url, headers, data, provider="gemini")

    def _stream_groq(self, model_id: str, session: SessionManager, sys_prompt: str) -> Tuple[bool, str, float]:
        if not LiemConfig.GROQ_API_KEY:
            console.print("[bold red]✖ Thiếu API_KEY_V10 (Groq Key) trong file .env![/bold red]")
            return False, "", 0.0

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {LiemConfig.GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model_id,
            "messages": session.to_groq_messages(sys_prompt, LiemConfig.MAX_CONTEXT_TURNS),
            "temperature": self.temperature,
            "stream": True
        }
        return self._execute_sse_stream(url, headers, data, provider="groq")

    def _execute_sse_stream(self, url: str, headers: Dict, data: Dict, provider: str) -> Tuple[bool, str, float]:
        start_time = time.time()
        full_answer = ""
        model_label = self.current_model_info["id"]

        try:
            # 1. Hiển thị Spinner trong lúc chờ phản hồi đầu tiên từ Server
            with Status(f"[bold cyan]LiemAI ({model_label}) đang suy nghĩ...[/bold cyan]", console=console, spinner="dots"):
                response = self.http_session.post(url, headers=headers, json=data, stream=True, timeout=(10, 60))

            if response.status_code != 200:
                console.print(f"[bold red]✖ Lỗi phản hồi từ API (HTTP {response.status_code})[/bold red]")
                if input("Bạn có muốn xem chi tiết mã lỗi? [y/n]: ").strip().lower() == "y":
                    console.print(Panel(response.text, title="Chi tiết lỗi API", border_style="red"))
                return False, "", 0.0

            # 2. Stream trực tiếp bên trong khung Panel bo góc đẹp mắt
            with Live(
                Panel(Markdown("..."), title=f"[bold cyan]🤖 LiemAI • {model_label}[/bold cyan]", border_style="cyan"),
                console=console,
                refresh_per_second=18,
                vertical_overflow="visible"
            ) as live:
                for line in response.iter_lines():
                    if not line:
                        continue
                    decoded_line = line.decode("utf-8", errors="ignore").strip()
                    if not decoded_line.startswith("data: "):
                        continue

                    json_str = decoded_line[6:].strip()
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
                            elapsed = time.time() - start_time
                            word_count = len(full_answer.split())
                            live.update(
                                Panel(
                                    Markdown(full_answer),
                                    title=f"[bold cyan]🤖 LiemAI • {model_label}[/bold cyan]",
                                    subtitle=f"[dim]⏱ {elapsed:.1f}s | ~{word_count} từ | Nhấn Ctrl+C để dừng[/dim]",
                                    border_style="cyan",
                                    padding=(0, 1)
                                )
                            )
                    except (KeyError, IndexError, json.JSONDecodeError):
                        pass

            total_time = time.time() - start_time
            return True, full_answer, total_time

        except KeyboardInterrupt:
            # Cho phép người dùng bấm Ctrl+C để ngắt câu trả lời đang chạy mà không sập chương trình
            console.print("\n[bold yellow]⚠ Đã dừng sinh văn bản theo yêu cầu của bạn.[/bold yellow]")
            total_time = time.time() - start_time
            return (len(full_answer) > 0), full_answer, total_time

        except requests.exceptions.Timeout:
            console.print("[bold red]✖ Quá thời gian chờ kết nối (Timeout)! Vui lòng thử lại.[/bold red]")
            return False, "", 0.0

        except Exception as e:
            console.print(f"[bold red]✖ Lỗi kết nối mạng: {e}[/bold red]")
            return False, "", 0.0


# =====================================================================
# 4. CLASS ĐIỀU KHIỂN CHÍNH & GIAO DIỆN TUI (LIEMAI 3.0 APP)
# =====================================================================
class LiemAIApp:
    def __init__(self):
        self.user_name = "User"
        self.sys_prompt = ""
        self.is_premium = False
        self.num_of_questions = 0
        self.client = AIClient()
        self.session: Optional[SessionManager] = None

    @staticmethod
    def load_prompt_file(file_path: str, user_name: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                template = f.read()
            return template.replace("{name}", user_name)
        except FileNotFoundError:
            return f"You are LiemAI 3.0, an intelligent, witty AI collaborator chatting with {user_name}."

    def render_banner(self) -> None:
        os.system("cls" if os.name == "nt" else "clear")
        banner_text = Text.from_markup(
            "[bold cyan]╔══════════════════════════════════════════════════════════════╗\n"
            "║[/bold cyan]        [bold magenta]✨ LiemAI 3.0 — MULTI-MODEL TERMINAL STUDIO ✨[/bold magenta]       [bold cyan]║\n"
            "║[/bold cyan]      [dim]Kiến trúc OOP • Dual-Engine Streaming • Smart Context[/dim]   [bold cyan]║\n"
            "╚══════════════════════════════════════════════════════════════╝[/bold cyan]"
        )
        console.print(banner_text)

    def render_help_table(self) -> None:
        table = Table(title="[bold cyan]BẢNG ĐIỀU KHIỂN LỆNH (SLASH COMMANDS)[/bold cyan]", border_style="cyan", show_lines=False)
        table.add_column("Lệnh", style="bold yellow", no_wrap=True)
        table.add_column("Chức năng", style="white")
        table.add_column("Điểm mới ở 3.0", style="italic green")

        table.add_row('/file "<path>" [yêu cầu]', "Đính kèm file code/văn bản vào câu hỏi", "Hỗ trợ tên file có dấu cách")
        table.add_row("/model", "Chuyển đổi nóng giữa Gemini 4 / 3.6 / 3.5 & Groq", "MỚI (Dual-Engine)")
        table.add_row("/temp <0.0 - 2.0>", "Chỉnh độ sáng tạo (Temperature) của AI", "MỚI")
        table.add_row("/undo", "Thu hồi câu hỏi & câu trả lời gần nhất", "MỚI")
        table.add_row("/history", "Xem tóm tắt các lượt hội thoại trong bộ nhớ", "MỚI")
        table.add_row("/export [tên_file.md]", "Xuất toàn bộ cuộc trò chuyện ra file Markdown", "MỚI")
        table.add_row("/save [tên_file.json]", "Lưu phiên trò chuyện ra định dạng JSON", "Tương thích bản 2.x")
        table.add_row("/load <tên_file.json>", "Khôi phục phiên trò chuyện từ file JSON", "Tự nhận diện chuẩn cũ/mới")
        table.add_row("/key [mã_key]", "Nâng cấp lên LiemAI PREMIUM không giới hạn", "Kế thừa bản 2.1")
        table.add_row("/status | /help", "Xem trạng thái hệ thống hoặc mở lại bảng này", "Nâng cấp giao diện")
        table.add_row("/reset | esc", "Xóa sạch bộ nhớ ngữ cảnh | Thoát chương trình", "Kế thừa bản 2.x")

        console.print(table)

    def render_status_bar(self) -> None:
        plan_badge = (
            "[bold magenta]👑 PREMIUM[/bold magenta]"
            if self.is_premium
            else f"[bold yellow]⚡ FREE ({self.num_of_questions}/{LiemConfig.QUESTION_LIMIT})[/bold yellow]"
        )
        persist_badge = "[green]Auto-Save: ON[/green]" if self.session.is_persistent else "[dim]Auto-Save: OFF[/dim]"
        model_id = self.client.current_model_info["id"]
        ctx_count = len(self.session.history) // 2

        status_line = (
            f"[bold green]👤 {self.user_name}[/bold green]  │  "
            f"[bold cyan]🧠 {model_id}[/bold cyan] (Temp: {self.client.temperature})  │  "
            f"💬 Ngữ cảnh: {ctx_count} lượt  │  {persist_badge}  │  {plan_badge}"
        )
        console.print(Rule(status_line, style="dim cyan"))

    def handle_slash_command(self, text: str) -> bool:
        """Trả về True nếu input là lệnh hệ thống (không tính vào câu hỏi gửi AI)."""
        cmd = text.strip()

        if cmd == "/help":
            self.render_help_table()
            return True

        if cmd == "/reset":
            self.session.clear()
            console.print("[bold green]✔ Đã dọn sạch bộ nhớ ngữ cảnh![/bold green]")
            return True

        if cmd == "/undo":
            if self.session.undo_last_turn():
                console.print("[bold green]✔ Đã thu hồi lượt trò chuyện gần nhất![/bold green]")
            else:
                console.print("[bold yellow]⚠ Bộ nhớ đang trống, không có gì để thu hồi.[/bold yellow]")
            return True

        if cmd == "/history":
            if not self.session.history:
                console.print("[dim]Chưa có tin nhắn nào trong phiên hiện tại.[/dim]")
                return True
            hist_table = Table(title="Lịch sử bộ nhớ hiện tại", border_style="blue")
            hist_table.add_column("#", style="dim", width=4)
            hist_table.add_column("Thời gian", style="cyan", width=10)
            hist_table.add_column("Vai trò", style="bold", width=12)
            hist_table.add_column("Nội dung tóm tắt", style="white")
            for idx, msg in enumerate(self.session.history, 1):
                role_label = f"[green]{self.user_name}[/green]" if msg["role"] == "user" else "[cyan]LiemAI[/cyan]"
                preview = msg["content"].replace("\n", " ")
                if len(preview) > 75:
                    preview = preview[:75] + "..."
                hist_table.add_row(str(idx), msg.get("time", ""), role_label, preview)
            console.print(hist_table)
            return True

        if cmd.startswith("/save"):
            parts = cmd.split(maxsplit=1)
            filename = parts[1].strip() if len(parts) > 1 else (self.session.session_file or f"session_{self.user_name}.json")
            self.session.save_to_json(filename)
            return True

        if cmd.startswith("/load"):
            parts = cmd.split(maxsplit=1)
            if len(parts) < 2:
                console.print("[bold yellow]⚠ Cú pháp: /load <tên_file.json>[/bold yellow]")
                return True
            self.session.load_from_json(parts[1].strip())
            return True

        if cmd.startswith("/export"):
            parts = cmd.split(maxsplit=1)
            filename = parts[1].strip() if len(parts) > 1 else f"chat_{self.user_name}_{datetime.now().strftime('%H%M')}.md"
            self.session.export_to_markdown(filename)
            return True

        if cmd.startswith("/model"):
            console.print("\n[bold cyan]--- CHỌN MÔ HÌNH AI (DUAL-ENGINE) ---[/bold cyan]")
            for k, v in LiemConfig.MODELS.items():
                marker = " [bold green](Đang dùng)[/bold green]" if k == self.client.current_model_key else ""
                console.print(f"  [bold yellow]{k}[/bold yellow]. {v['name']} [dim]({v['id']})[/dim]{marker}")
            choice = input("-> Chọn số [1-4] (Enter để giữ nguyên): ").strip()
            if choice in LiemConfig.MODELS:
                self.client.current_model_key = choice
                console.print(f"[bold green]✔ Đã chuyển sang: {LiemConfig.MODELS[choice]['name']}[/bold green]")
            return True

        if cmd.startswith("/temp"):
            parts = cmd.split(maxsplit=1)
            if len(parts) < 2:
                console.print(f"[bold cyan]Nhiệt độ (Temperature) hiện tại: {self.client.temperature}[/bold cyan] (Nhập: /temp <0.0 - 2.0>)")
                return True
            try:
                val = float(parts[1].strip())
                if 0.0 <= val <= 2.0:
                    self.client.temperature = val
                    console.print(f"[bold green]✔ Đã cập nhật Temperature = {val}[/bold green]")
                else:
                    console.print("[bold red]✖ Giá trị phải nằm trong khoảng 0.0 đến 2.0![/bold red]")
            except ValueError:
                console.print("[bold red]✖ Vui lòng nhập một số hợp lệ![/bold red]")
            return True

        if cmd.startswith("/key"):
            parts = cmd.split(maxsplit=1)
            input_k = parts[1].strip() if len(parts) > 1 else input("🔑 Nhập key kích hoạt Premium: ").strip()
            if input_k == LiemConfig.SECRET_KEY:
                self.is_premium = True
                console.print(Panel.fit("[bold magenta]🎉 Kích hoạt LiemAI 3.0 PREMIUM thành công! Không giới hạn câu hỏi.[/bold magenta]", border_style="magenta"))
            else:
                console.print("[bold red]✖ Key không hợp lệ![/bold red]")
            return True

        if cmd == "/status":
            self.render_status_bar()
            return True

        return False

    @staticmethod
    def parse_file_command(text: str) -> Optional[str]:
        """Xử lý lệnh /file thông minh bằng shlex, hỗ trợ đường dẫn có khoảng trắng."""
        clean_text = text.strip()
        if not clean_text.startswith("/file"):
            return text

        try:
            tokens = shlex.split(clean_text)
        except ValueError:
            tokens = clean_text.split(maxsplit=2)

        if len(tokens) < 2:
            console.print('[bold yellow]⚠ Cú pháp: /file "<đường_dẫn_file>" [yêu cầu phân tích][/bold yellow]')
            return None

        file_path = tokens[1]
        user_prompt = " ".join(tokens[2:]) if len(tokens) > 2 else "Hãy tóm tắt và phân tích chi tiết nội dung file này."

        if not os.path.isfile(file_path):
            console.print(f"[bold red]✖ Không tìm thấy file tại: '{file_path}'[/bold red]")
            return None

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            line_count = content.count("\n") + 1
            size_kb = os.path.getsize(file_path) / 1024
            console.print(f"[bold green]📎 Đã đính kèm file:[/bold green] [cyan]{file_path}[/cyan] ({line_count} dòng, {size_kb:.1f} KB)")
            return f"[Tài liệu đính kèm: {file_path}]\n```\n{content}\n```\n\nYêu cầu của người dùng: {user_prompt}"
        except Exception as e:
            console.print(f"[bold red]✖ Lỗi khi đọc file: {e}[/bold red]")
            return None

    def check_quota_gate(self) -> bool:
        """Kiểm tra giới hạn câu hỏi bản Free. Trả về False nếu người dùng chọn thoát."""
        while not self.is_premium and self.num_of_questions >= LiemConfig.QUESTION_LIMIT:
            console.print(
                Panel(
                    f"[bold red]Bạn đã sử dụng hết {LiemConfig.QUESTION_LIMIT}/{LiemConfig.QUESTION_LIMIT} lượt hỏi miễn phí![/bold red]\n"
                    "[yellow]Hãy nhập mã khóa để nâng cấp lên [bold magenta]LiemAI 3.0 PREMIUM[/bold magenta] (Mặc định: 'key').[/yellow]",
                    title="🔒 GIỚI HẠN BẢN DÙNG THỬ",
                    border_style="yellow"
                )
            )
            key_input = input("🔑 Nhập key kích hoạt (hoặc gõ 'esc' để thoát): ").strip()
            if key_input.lower() == "esc":
                return False
            if key_input == LiemConfig.SECRET_KEY:
                self.is_premium = True
                console.print(Panel.fit("[bold magenta]🎉 Chào mừng bạn đến với LiemAI 3.0 PREMIUM![/bold magenta]", border_style="magenta"))
                return True
            console.print("[bold red]✖ Key không chính xác, vui lòng thử lại![/bold red]\n")
        return True

    def run(self) -> None:
        self.render_banner()
        self.user_name = input("\n👤 Nhập tên của bạn (Enter = User): ").strip() or "User"

        storage_choice = input("💾 Tự động lưu lịch sử phiên trò chuyện này? (Y/N): ").strip().lower()
        is_persistent = storage_choice == "y"

        self.session = SessionManager(self.user_name, is_persistent=is_persistent)
        self.sys_prompt = self.load_prompt_file(LiemConfig.DEFAULT_PROMPT_FILE, self.user_name)

        print()
        self.render_help_table()

        if is_persistent:
            console.print(f"[dim]➜ Đã bật tự động lưu vào '{self.session.session_file}'[/dim]")

        # Chào người dùng ngay lập tức mà KHÔNG tốn 1 API call lãng phí như bản 1.5 - 2.1
        greeting_msg = f"Hello **{self.user_name}**, how can I help you today? *(Gõ `/help` để xem các tính năng mới của LiemAI 3.0)*"
        console.print(
            Panel(
                Markdown(greeting_msg),
                title=f"[bold cyan]🤖 LiemAI • {self.client.current_model_info['id']}[/bold cyan]",
                border_style="cyan"
            )
        )

        # VÒNG LẶP SỰ KIỆN CHÍNH
        while True:
            if not self.check_quota_gate():
                console.print("[bold blue]👋 Tạm biệt! Hẹn gặp lại bạn ở LiemAI 3.0![/bold blue]")
                break

            print()
            self.render_status_bar()
            console.print(f"[bold green]🧑 {self.user_name}:[/bold green]")
            user_input = get_multiline_input("")

            if not user_input:
                continue

            if user_input.strip().lower() in ("esc", "thoat"):
                console.print("[bold blue]👋 Tạm biệt! Hẹn gặp lại bạn![/bold blue]")
                break

            # 1. Kiểm tra các lệnh hệ thống (/model, /save, /load, /undo, /export...)
            if self.handle_slash_command(user_input):
                continue

            # 2. Kiểm tra lệnh nạp file (/file)
            processed_prompt = self.parse_file_command(user_input)
            if not processed_prompt:
                continue

            # 3. Lưu câu hỏi và gọi AI Streaming
            self.session.add_message("user", processed_prompt)
            success, answer_text, _ = self.client.stream_chat(self.session, self.sys_prompt)

            if success and answer_text:
                self.session.add_message("assistant", answer_text)
                if not self.is_premium:
                    self.num_of_questions += 1
            else:
                # Thu hồi câu hỏi của user nếu gọi API thất bại để giữ sạch ngữ cảnh
                self.session.pop_last_user_message()


if __name__ == "__main__":
    app = LiemAIApp()
    app.run()