import os
from groq import Groq

def check_groq_models():

    try:
        client = Groq()
    except Exception as e:
        print(f"Lỗi khởi tạo client (Hãy kiểm tra biến môi trường GROQ_API_KEY): {e}")
        return

    print("Đang kết nối tới Groq API để lấy danh sách model...\n")
    
    try:

        models_response = client.models.list()
        

        print(f"{'STT':<5} | {'Tên Model (ID)':<35} | {'Được sở hữu bởi':<15}")
        print("-" * 65)
        
        for idx, model in enumerate(models_response.data, 1):
            model_id = model.id
            owned_by = getattr(model, 'owned_by', 'N/A')
            print(f"{idx:<5} | {model_id:<35} | {owned_by:<15}")
            
        print(f"\nTổng số model đang khả dụng: {len(models_response.data)}")

    except Exception as e:
        print(f"Đã xảy ra lỗi khi gọi API Groq: {e}")

if __name__ == "__main__":

    os.environ["GROQ_API_KEY"] = "gsk_myapi"
    
    check_groq_models()
