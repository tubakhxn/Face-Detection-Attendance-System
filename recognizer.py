import cv2
import face_recognition
import pickle
import numpy as np
import os
from datetime import datetime
from attendance import mark_attendance


def start_attendance_system():
    if not os.path.exists('encodings.pkl'):
        print("No encodings found. Please train the model first.")
        return
    with open('encodings.pkl', 'rb') as f:
        data = pickle.load(f)
    known_encodings = data['encodings']
    known_names = data['names']
    known_ids = data['ids']
    marked_today = set()
    cap = cv2.VideoCapture(0)
    print("Starting attendance system. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Webcam error.")
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, boxes)
        for (box, encoding) in zip(boxes, encodings):
            matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
            face_dist = face_recognition.face_distance(known_encodings, encoding)
            name = "Unknown"
            emp_id = ""
            color = (0,0,255)
            conf = 0.0
            if True in matches:
                best_match = np.argmin(face_dist)
                name = known_names[best_match]
                emp_id = known_ids[best_match]
                color = (0,255,0)
                conf = 1 - face_dist[best_match]
                key = (name, emp_id)
                if key not in marked_today:
                    if mark_attendance(name, emp_id):
                        marked_today.add(key)
                        label = f"{name} ({emp_id})"
                    else:
                        label = f"{name} ({emp_id}) - Already Marked"
                else:
                    label = f"{name} ({emp_id}) - Already Marked"
            else:
                label = "Unknown"
            top, right, bottom, left = box
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"Conf: {conf:.2f}", (left, bottom+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.imshow("Attendance System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
