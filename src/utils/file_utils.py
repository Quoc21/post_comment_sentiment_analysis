import json
import os

def load_data_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_data_json(obj, path):
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)  # Tạo folder nếu chưa tồn tại

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=4)
        print(f"Đã lưu JSON thành công tại: {path}")
    except Exception as e:
        print("Lỗi khi lưu JSON:", e)
