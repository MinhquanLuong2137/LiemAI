import json
import os
import requests

# Module nhập liệu đa dòng
from input_handler import get_multiline_input

from dotenv import load_dotenv

# Thêm các module từ thư viện rich để render UI và Markdown
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule

# Khởi tạo đối tượng Console để quản lý hiển thị và màu sắc
console = Console()

chat_history = []

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_V20")
SYS_PROMPT = ""  # Khai báo biến toàn cục chứa luật AI



# Khai báo cấu hình quản lý phiên (session)
is_persistent = False
session_file = ""

#Đọc và nạp biến {name} trong injection prompt
def load_prompt(file_path, user_name):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            template = f.read()
        # Thay thế placeholder {name} bằng tên người dùng nhập vào
        return template.replace("{name}", user_name)
    except FileNotFoundError:
        return f"You are LiemAI chatting with {user_name}."
#############################################################

# Hàm lưu lịch sử ra file JSON
def save_chat_to_file(filename):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(chat_history, f, ensure_ascii=False, indent=2)
        # Sử dụng thẻ màu của rich để báo hiệu trạng thái
        console.print(f"[bold green]\[Đã lưu hội thoại vào '{filename}'][/bold green]")
    except Exception as e:
        console.print(f"[bold red]\[Lỗi lưu file: {e}][/bold red]")

def handle_session_commands(text):
    global chat_history
    cmd = text.strip()

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
            console.print("[bold yellow]\[Lỗi cú pháp] Vui lòng nhập: /load <tên_file>[/bold yellow]")
            return True
        filename = parts[1].strip()
        if not filename.endswith(".json"):
            filename += ".json"
        if not os.path.isfile(filename):
            console.print(f"[bold red]\[Lỗi] Không tìm thấy file '{filename}'[/bold red]")
            return True
        try:
            with open(filename, "r", encoding="utf-8") as f:
                chat_history = json.load(f)
            console.print(f"[bold green]\[Đã tải lịch sử trò truyện từ '{filename}' thành công!][/bold green]")
        except Exception as e:
            console.print(f"[bold red]\[Lỗi nạp file: {e}][/bold red]")
        return True

    if cmd == "/reset":
        chat_history.clear()
        console.print("[bold green]\[Đã dọn sạch bộ nhớ hội thoại!][/bold green]")
        return True

    return False

def handle_file_command(text):
    clean_text = text.strip()
    if clean_text.startswith("/file"):
        parts = clean_text.split(maxsplit=2)
        if len(parts) < 2:
            console.print("[bold yellow]Lỗi cú pháp! Vui lòng nhập: /file <đường_dẫn_file> [câu hỏi kèm theo][/bold yellow]")
            return None
        
        file_path = parts[1]
        user_prompt = parts[2] if len(parts) > 2 else "Hãy phân tích nội dung file này."

        if not os.path.isfile(file_path):
            console.print(f"[bold red]Lỗi: Không tìm thấy file tại '{file_path}'[/bold red]")
            return None

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            return f"[Nội dung file: {file_path}]\n```\n{content}\n```\n\n{user_prompt}"
        except Exception as e:
            console.print(f"[bold red]Lỗi khi đọc file: {e}[/bold red]")
            return None
    return text



def AI_response(question):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.7-flash:streamGenerateContent?alt=sse&key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}

    chat_history.append({
        "role": "user",
        "parts": [{"text": question}]
    })

    # Đính kèm systemInstruction ở TẤT CẢ các lượt gọi để AI luôn nhớ vai
    data = {
        "contents": chat_history,
        "systemInstruction": {
            "parts": [{"text": SYS_PROMPT}]
        },
        "generationConfig": {"temperature": 0.7}
    }

    # Thêm dòng kẻ ngang (Rule) phân cách các lượt chat cho đẹp mắt
    console.print(Rule(style="cyan"))

    try:
        response = requests.post(url, headers=headers, json=data, stream=True)

        if response.status_code == 200:
            full_answer_text = ""
            
            # Khởi tạo Live context để cập nhật Markdown liên tục không làm trôi dòng
            with Live(Markdown(full_answer_text), console=console, refresh_per_second=15, vertical_overflow="visible") as live:
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith("data: "):
                            chunk = json.loads(decoded_line[6:])
                            try:
                                text_piece = chunk['candidates'][0]['content']['parts'][0]['text']
                                full_answer_text += text_piece
                                # Đẩy chuỗi cập nhật vào Markdown để render lại ngay lập tức
                                live.update(Markdown(full_answer_text))
                            except (KeyError, IndexError):
                                pass

            console.print(Rule(style="cyan"))
            
            chat_history.append({
                "role": "model",
                "parts": [{"text": full_answer_text}]
            })

            if is_persistent and session_file:
                save_chat_to_file(session_file)
        else:
            console.print("[bold red]Limits reach![/bold red]")
            if input("Wanna see Error code ? [y/n] ").lower() == 'y':
                console.print(f"[bold red]Error: {response.status_code}: {response.text}[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Network error: {e}[/bold red]")
        chat_history.pop()

if __name__ == "__main__":

    # Dùng os để clear terminal sạch sẽ lúc vừa khởi động
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Bọc tiêu đề ứng dụng vào một khung chữ nhật (Panel) để nổi bật
    console.print(Panel.fit("[bold red]Welcome to LiemAI 2.0[/bold red]", border_style="red"))


    name = input("Your name: ").strip()


    HELP_MENU = """
    [bold yellow]/file <path> [prompt][/bold yellow] : Nạp file văn bản/code vào câu hỏi
    [bold yellow]/save [tên_file][/bold yellow]       : Lưu lịch sử trò chuyện ra file .json
    [bold yellow]/load <tên_file>[/bold yellow]       : Nạp lại lịch sử trò chuyện từ file .json
    [bold yellow]/reset[/bold yellow]                 : Dọn sạch ngữ cảnh bộ nhớ
    [bold yellow]esc[/bold yellow]                    : Thoát chương trình
    """

    # Dán dòng này ở __main__ (ngay trước vòng lặp chat) để hiện menu:
    console.print(Panel(HELP_MENU.strip(), title="[bold cyan]HƯỚNG DẪN PHÍM TẮT[/bold cyan]", border_style="cyan"))
   


    # Hỏi chế độ lưu trữ trước khi bắt đầu phiên trò chuyện
    storage_choice = input("Bạn có muốn lưu trữ cuộc trò chuyện lâu dài không? (Y/N): ").strip().lower()
    if storage_choice == 'y':
        is_persistent = True
        session_file = f"session_{name}.json"
        console.print(f"[dim]-> Chế độ: Lưu trữ lâu dài (Tự động ghi vào '{session_file}')[/dim]\n")
    else:
        is_persistent = False
        console.print("[dim]-> Chế độ: Tạm thời (Không tự động lưu khi thoát)[/dim]\n")
    
    # Tạo prompt định hình vai trò
    SYS_PROMPT = load_prompt("prompt_v20.txt", name)

    # Lượt chào khởi tạo đầu tiên
    first_greeting = f'Please acknowledge my presence by responding EXACTLY with: "Hello {name}, how can I help you ?" and nothing else.'
    AI_response(first_greeting)
    
    NumOfQuestions = 0  
    QuestionLimit = 5
    check = 0

 
    while True:
        # Kiểm tra nếu vượt quá số câu hỏi ở bản Free
        if NumOfQuestions >= QuestionLimit:
            console.print("\n[bold yellow]" + "*" * 15 + "[/bold yellow]")
            console.print("[bold red]Limits reach...[/bold red]")
            console.print(f"[yellow]The trial version only allows you to ask {QuestionLimit} questions.[/yellow]")
            console.print("[yellow]Please subscribe to the Premium plan to receive the extension key. ![/yellow]")
            console.print("[bold yellow]<=======================================================================================>[/bold yellow]\n")

            if input("Input key: ").strip() == "key":
                console.print(Panel.fit("[bold magenta]Welcome to LiemAI 2.0 PREMIUM[/bold magenta]", border_style="magenta"))
                check = 1
                while True:
                    console.print(f"[bold green]{name}:[/bold green] ", end="")
                    user_question = get_multiline_input("")
        
                    if user_question.lower() == 'esc':
                        console.print("[bold blue]Good bye! See you soon![/bold blue]")
                        break
                    if user_question:
                        if handle_session_commands(user_question):
                            continue
                        processed_input = handle_file_command(user_question)
                        if processed_input:
                            AI_response(processed_input)
            
            if check == 1:
                break
            
            console.print("[bold red]Inappropriate Key! Make sure your key is correct before \"Enter\"[/bold red]")
            input("\"Enter\" to escape...")
            break
            
        # Chế độ dùng thử
        user_question = get_multiline_input(f"{name}: ")
        
        if user_question.lower() == 'esc':
            print("Good bye! See you soon!")
            break
            
        if user_question:  # Bỏ qua nếu tin nhắn rỗng
            # Kiểm tra và thực thi các slash commands trước khi gọi AI
            if handle_session_commands(user_question):
                continue

            # Kiểm tra xem người dùng có nạp file qua lệnh /file hay không
            processed_input = handle_file_command(user_question)
            if processed_input:
                AI_response(processed_input)
                NumOfQuestions += 1
