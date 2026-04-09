import cv2
import os
import uuid

def add_employee():
    name = input("Enter employee name: ").strip()
    emp_id = input("Enter employee ID: ").strip()
    dept = input("Enter department (optional): ").strip()
    folder_name = f"{name}_{emp_id}"
    save_dir = os.path.join('dataset', folder_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Capturing images for {name} ({emp_id})...")
    cap = cv2.VideoCapture(0)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to access webcam.")
            break
        cv2.imshow("Capture Face - Press 'c' to capture, 'q' to quit", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('c'):
            img_path = os.path.join(save_dir, f"{uuid.uuid4().hex}.jpg")
            cv2.imwrite(img_path, frame)
            count += 1
            print(f"Captured image {count}")
        elif key & 0xFF == ord('q'):
            break
        if count >= 10:
            print("Captured 10 images. Done.")
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"Images saved in {save_dir}")
