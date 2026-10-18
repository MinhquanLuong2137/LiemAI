import json
import os
import requests
from dotenv import load_dotenv

# Thư viện Rich & Handler
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from input_handler import get_multiline_input

console = Console()
load_dotenv()

# --- CẤU HÌNH HỆ THỐNG ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_V20")
MAX_CONTEXT_TURNS = 12  # Giới hạn 12 message gần nhất gửi đi để tiết kiệm token
QUESTION_LIMIT = 5      # Giới hạn bản Free
SECRET_KEY = "key"      # Key kích hoạt Premium

chat_history = []
SYS_PROMPT = ""
is_persistent = False
session_file = ""
is_premium = False
num_of_questions = 0


def load_prompt(file_path, user_name):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            template = f.read()
        return template.replace("{name}", user_name)
    except FileNotFoundError:
        return f"You are LiemAI, a helpful AI assistant chatting with {user_name}."


def save_chat_to_file(filename):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(chat_history, f, ensure_ascii=False, indent=2)
        console.print(f"[bold green][Đã lưu hội thoại vào '{filename}'][/bold green]")
    except Exception as e:
        console.print(f"[bold red][Lỗi lưu file: {e}][/bold red]")


def handle_session_commands(text):
    global chat_history, is_premium
    cmd = text.strip()

    if cmd == "/reset":
        chat_history.clear()
        console.print("[bold green][Đã dọn sạch bộ nhớ ngữ cảnh!][/bold green]")
        return True

    if cmd.startswith("/save"):
        parts = cmd.split(maxsplit=1)
        filename = parts[1].strip() if len(parts) > 1 else (session_file or "chat_session.json")
        if not filename.endswith(".json"):
            filename += ".json"
        save_chat_to_file(filename)
        return True

    if cmd.startswith("/load"):
        parts = cmd.split(maxsplit=1)
        if len(parts) < 2:
            console.print("[bold yellow][Lỗi cú pháp] Nhập: /load <tên_file>[/bold yellow]")
            return True
        filename = parts[1].strip()
        if not filename.endswith(".json"):
            filename += ".json"
        if not os.path.isfile(filename):
            console.print(f"[bold red][Lỗi] Không tìm thấy file '{filename}'[/bold red]")
            return True
        try:
            with open(filename, "r", encoding="utf-8") as f:
                chat_history = json.load(f)
            console.print(f"[bold green][Đã nạp lịch sử từ '{filename}' thành công!][/bold green]")
        except Exception as e:
            console.print(f"[bold red][Lỗi nạp file: {e}][/bold red]")
        return True

    if cmd.startswith("/key"):
        parts = cmd.split(maxsplit=1)
        input_k = parts[1].strip() if len(parts) > 1 else input("Nhập key Premium: ").strip()
        if input_k == SECRET_KEY:
            is_premium = True
            console.print(Panel.fit("[bold magenta]Kích hoạt LiemAI PREMIUM thành công![/bold magenta]", border_style="magenta"))
        else:
            console.print("[bold red]Key không hợp lệ![/bold red]")
        return True

    if cmd == "/status":
        plan = "[bold magenta]PREMIUM (Không giới hạn)[/bold magenta]" if is_premium else f"[yellow]FREE ({num_of_questions}/{QUESTION_LIMIT} câu)[/yellow]"
        console.print(f"Trạng thái tài khoản: {plan}")
        return True

    return False


def handle_file_command(text):
    clean_text = text.strip()
    if clean_text.startswith("/file"):
        parts = clean_text.split(maxsplit=2)
        if len(parts) < 2:
            console.print("[bold yellow]Lỗi cú pháp! Nhập: /file <đường_dẫn_file> [yêu cầu][/bold yellow]")
            return None
        
        file_path = parts[1]
        user_prompt = parts[2] if len(parts) > 2 else "Hãy tóm tắt và phân tích nội dung file này."

        if not os.path.isfile(file_path):
            console.print(f"[bold red]Lỗi: Không tìm thấy file tại '{file_path}'[/bold red]")
            return None

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            return f"[File: {file_path}]\n```\n{content}\n```\n\n{user_prompt}"
        except Exception as e:
            console.print(f"[bold red]Lỗi khi đọc file: {e}[/bold red]")
            return None
    return text


def AI_response(question):
    # Bảo mật API Key qua Header thay vì để trên URL params
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:streamGenerateContent?alt=sse"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }

    # Ghi nhận lượt hỏi của user
    chat_history.append({
        "role": "user",
        "parts": [{"text": question}]
    })

    # Cắt tỉa ngữ cảnh (Sliding Window) để chống tràn token & tiết kiệm chi phí
    trimmed_context = chat_history[-MAX_CONTEXT_TURNS:]

    data = {
        "contents": trimmed_context,
        "systemInstruction": {
            "parts": [{"text": SYS_PROMPT}]
        },
        "generationConfig": {"temperature": 0.7}
    }

    console.print(Rule(style="cyan"))

    try:
        response = requests.post(url, headers=headers, json=data, stream=True)

        if response.status_code == 200:
            full_answer_text = ""
            with Live(Markdown(full_answer_text), console=console, refresh_per_second=15, vertical_overflow="visible") as live:
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith("data: "):
                            try:
                                chunk = json.loads(decoded_line[6:])
                                text_piece = chunk['candidates'][0]['content']['parts'][0]['text']
                                full_answer_text += text_piece
                                live.update(Markdown(full_answer_text))
                            except (KeyError, IndexError, json.JSONDecodeError):
                                pass

            console.print(Rule(style="cyan"))
            
            chat_history.append({
                "role": "model",
                "parts": [{"text": full_answer_text}]
            })

            if is_persistent and session_file:
                save_chat_to_file(session_file)
            return True

        else:
            console.print(f"[bold red]Lỗi phản hồi (HTTP {response.status_code})[/bold red]")
            if input("Xem chi tiết lỗi? [y/n]: ").lower() == 'y':
                console.print(f"[red]{response.text}[/red]")
            chat_history.pop()  # Thu hồi message nếu gọi thất bại
            return False

    except Exception as e:
        console.print(f"[bold red]Lỗi mạng/kết nối: {e}[/bold red]")
        chat_history.pop()  # Thu hồi message nếu rớt mạng
        return False


if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    console.print(Panel.fit("[bold red]Welcome to LiemAI 2.5[/bold red]", border_style="red"))

    name = input("Your name: ").strip() or "User"

    HELP_MENU = """
    [bold yellow]/file <path> [prompt][/bold yellow] : Nạp file văn bản/code vào câu hỏi
    [bold yellow]/save [tên_file][/bold yellow]       : Lưu lịch sử trò chuyện ra file .json
    [bold yellow]/load <tên_file>[/bold yellow]       : Nạp lại lịch sử trò chuyện từ file .json
    [bold yellow]/key [mã_key][/bold yellow]          : Nâng cấp tài khoản lên bản Premium
    [bold yellow]/status[/bold yellow]                 : Kiểm tra số câu hỏi còn lại
    [bold yellow]/reset[/bold yellow]                  : Dọn sạch ngữ cảnh bộ nhớ
    [bold yellow]esc[/bold yellow]                     : Thoát chương trình
    """
    console.print(Panel(HELP_MENU.strip(), title="[bold cyan]HƯỚNG DẪN LỆNH[/bold cyan]", border_style="cyan"))

    storage_choice = input("Bạn có muốn tự động lưu phiên này không? (Y/N): ").strip().lower()
    if storage_choice == 'y':
        is_persistent = True
        session_file = f"session_{name}.json"
        console.print(f"[dim]-> Đã bật tự động lưu vào '{session_file}'[/dim]\n")

    SYS_PROMPT = load_prompt("prompt_v20.txt", name)

    # Chào khởi tạo
    first_greeting = f'Please acknowledge my presence by responding EXACTLY with: "Hello {name}, how can I help you ?" and nothing else.'
    AI_response(first_greeting)

    # VÒNG LẶP CHÍNH DUY NHẤT
    while True:
        # Kiểm tra quota bản Free
        if not is_premium and num_of_questions >= QUESTION_LIMIT:
            console.print("\n[bold yellow]" + "=" * 50 + "[/bold yellow]")
            console.print("[bold red]Bạn đã hết lượt hỏi miễn phí![/bold red]")
            console.print(f"[yellow]Bản dùng thử chỉ hỗ trợ {QUESTION_LIMIT} câu hỏi.[/yellow]")
            key_input = input("Nhập key kích hoạt Premium (hoặc gõ 'esc' để thoát): ").strip()
            
            if key_input.lower() == 'esc':
                console.print("[bold blue]Tạm biệt![/bold blue]")
                break
            elif key_input == SECRET_KEY:
                is_premium = True
                console.print(Panel.fit("[bold magenta]Chào mừng bạn đến với LiemAI PREMIUM![/bold magenta]", border_style="magenta"))
            else:
                console.print("[bold red]Key không chính xác![/bold red]")
                continue

        # Nhập tin nhắn
        user_question = get_multiline_input(f"{name}: ")

        if not user_question:
            continue

        if user_question.lower() == 'esc':
            console.print("[bold blue]Tạm biệt! Hẹn gặp lại![/bold blue]")
            break

        # Xử lý các Slash Commands
        if handle_session_commands(user_question):
            continue

        # Xử lý lệnh nạp file
        processed_input = handle_file_command(user_question)
        if not processed_input:
            continue

        # Gửi request sang AI
        success = AI_response(processed_input)
        if success and not is_premium:
            num_of_questions += 1