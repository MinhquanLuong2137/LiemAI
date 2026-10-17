import os
import requests
from dotenv import load_dotenv

# 1. Nạp file .env vào môi trường
load_dotenv()

# 2. Lấy API key từ biến môi trường
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_V15")

# Kiểm tra xem key đã được nạp thành công chưa
if not GEMINI_API_KEY:
    print("Lỗi: Không tìm thấy GEMINI_API_KEY! Hãy kiểm tra lại file .env của bạn.")
else:
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        res = response.json()
        
        print("Các model hỗ trợ generateContent:")
        for model in res.get("models", []):
            if "generateContent" in model.get("supportedGenerationMethods", []):
                print(f"- {model['name'].replace('models/', '')}")
                
    except requests.exceptions.RequestException as e:
        print(f"Đã xảy ra lỗi kết nối: {e}")