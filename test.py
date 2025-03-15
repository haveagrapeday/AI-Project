from PIL import Image
import os

image_folder = r"C:\Users\uoobu\Desktop\Final\Fianl-project-AI\datasources\princess"

for root, dirs, files in os.walk(image_folder):
    for file in files:
        file_path = os.path.join(root, file)
        try:
            img = Image.open(file_path)
            img.verify()  # ตรวจสอบว่าเป็นไฟล์ภาพที่เปิดได้
        except Exception as e:
            print(f"ไฟล์ {file_path} เปิดไม่ได้: {e}")


            for root, dirs, files in os.walk(image_folder):
    for file in files:
        file_path = os.path.join(root, file)
        try:
            img = Image.open(file_path).convert("RGB")  # แปลงเป็น RGB
            new_file_path = os.path.splitext(file_path)[0] + ".jpg"
            img.save(new_file_path, "JPEG")
            print(f"แปลง {file_path} → {new_file_path}")
        except Exception as e:
            print(f"ไม่สามารถแปลง {file_path}: {e}")

