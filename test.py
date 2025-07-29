import os

def list_folder_contents(folder_path):
    """
    ตรวจสอบและแสดงรายการไฟล์และโฟลเดอร์ย่อยทั้งหมดภายในโฟลเดอร์ที่ระบุ
    """
    print(f"--- ตรวจสอบเนื้อหาภายในโฟลเดอร์: {folder_path} ---")
    try:
        # ตรวจสอบว่า path ที่ให้มาเป็นโฟลเดอร์จริงหรือไม่
        if not os.path.isdir(folder_path):
            print(f"ข้อผิดพลาด: '{folder_path}' ไม่ใช่โฟลเดอร์ หรือไม่พบโฟลเดอร์นี้")
            return

        # ใช้ os.listdir() เพื่อรับรายชื่อไฟล์และโฟลเดอร์ย่อยทั้งหมด
        contents = os.listdir(folder_path)

        if not contents:
            print("โฟลเดอร์นี้ว่างเปล่า")
            return

        files = []
        subfolders = []

        # แยกไฟล์และโฟลเดอร์ย่อยออกจากกัน
        for item in contents:
            item_path = os.path.join(folder_path, item) # สร้าง full path ของแต่ละรายการ
            if os.path.isfile(item_path):
                files.append(item)
            elif os.path.isdir(item_path):
                subfolders.append(item)

        print(f"จำนวนรายการทั้งหมดในโฟลเดอร์: {len(contents)}")

        if subfolders:
            print(f"\n--- โฟลเดอร์ย่อย ({len(subfolders)}): ---")
            for sf in subfolders:
                print(f"  [DIR] {sf}")
        else:
            print("\nไม่มีโฟลเดอร์ย่อยในโฟลเดอร์นี้")

        if files:
            print(f"\n--- ไฟล์ ({len(files)}): ---")
            for f in files:
                print(f"  [FILE] {f}")
        else:
            print("\nไม่มีไฟล์ในโฟลเดอร์นี้")

    except FileNotFoundError:
        print(f"ข้อผิดพลาด: ไม่พบโฟลเดอร์ที่ {folder_path}")
    except PermissionError:
        print(f"ข้อผิดพลาด: ไม่มีสิทธิ์เข้าถึงโฟลเดอร์ {folder_path}")
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการอ่านโฟลเดอร์: {e}")

# --- ตัวอย่างการใช้งาน ---

# 1. สร้างโครงสร้างโฟลเดอร์และไฟล์จำลองเพื่อทดสอบ (ไม่จำเป็นต้องรันในโปรดักชัน)
# คุณสามารถสร้างด้วยมือ หรือใช้โค้ดนี้สร้างให้
# if not os.path.exists("my_test_folder"):
#     os.makedirs("my_test_folder/sub_dir_a")
#     os.makedirs("my_test_folder/sub_dir_b")
#     with open("my_test_folder/file1.txt", "w") as f:
#         f.write("This is file1.")
#     with open("my_test_folder/sub_dir_a/nested_file.csv", "w") as f:
#         f.write("col1,col2\n1,2")
#     print("สร้างโครงสร้างโฟลเดอร์ทดสอบ 'my_test_folder' เรียบร้อยแล้ว")
#     print("-" * 50)


# ตัวอย่างที่ 1: ตรวจสอบโฟลเดอร์ที่มีอยู่จริง (เปลี่ยนเป็น path ของคุณ)
# เช่น list_folder_contents("/Users/YourName/Documents/MyProject")
# list_folder_contents("my_test_folder")

# print("\n" + "=" * 50 + "\n")

# # ตัวอย่างที่ 2: ตรวจสอบโฟลเดอร์ย่อย
list_folder_contents("my_test_folder/sub_dir_b")

# print("\n" + "=" * 50 + "\n")

# # ตัวอย่างที่ 3: ตรวจสอบโฟลเดอร์ที่ไม่มีอยู่
# list_folder_contents("non_existent_folder")

# print("\n" + "=" * 50 + "\n")

# # ตัวอย่างที่ 4: ตรวจสอบ path ที่เป็นไฟล์แทนที่จะเป็นโฟลเดอร์
# list_folder_contents("my_test_folder/file1.txt")