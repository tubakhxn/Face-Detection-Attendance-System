"""
AI Attendance System using MediaPipe Face Detection
Run directly - auto-installs all required packages.
"""

# ─── AUTO-INSTALLER ───────────────────────────────────────────────────────────
import subprocess, sys

def _install(pkg):
    print(f"  Installing {pkg} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

def _ensure_deps():
    try:
        import cv2
        if not hasattr(cv2, 'VideoCapture'):
            raise ImportError
    except Exception:
        print("  Fixing OpenCV...")
        subprocess.call([sys.executable, "-m", "pip", "uninstall",
                         "opencv-python-headless", "opencv-python", "-y", "-q"])
        _install("opencv-python")
    try:
        import mediapipe as mp
        _ = mp.solutions.face_detection
    except Exception:
        _install("mediapipe")
    try:
        import numpy
    except Exception:
        _install("numpy")

_ensure_deps()
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import mediapipe as mp
import numpy as np
import os, csv, json, time
from datetime import datetime, date

# ─── MediaPipe setup ──────────────────────────────────────────────────────────
mp_face_detection = mp.solutions.face_detection
mp_face_mesh      = mp.solutions.face_mesh

# ─── Config ───────────────────────────────────────────────────────────────────
STUDENTS_FILE        = "students.json"
ATTENDANCE_FILE      = f"attendance_{date.today().isoformat()}.csv"
CONFIDENCE_THRESHOLD = 0.60
COOLDOWN_SECONDS     = 5
IDENTIFY_EVERY_N     = 8      # run face-ID every N frames (keeps it smooth)

# ─── Persistent FaceMesh (created ONCE, reused every frame = fast) ────────────
_face_mesh = None

def get_face_mesh():
    global _face_mesh
    if _face_mesh is None:
        _face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _face_mesh

# ─── Student DB ───────────────────────────────────────────────────────────────
def load_students():
    if os.path.exists(STUDENTS_FILE):
        with open(STUDENTS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_students(students):
    with open(STUDENTS_FILE, "w") as f:
        json.dump(students, f, indent=2)

# ─── Attendance CSV ───────────────────────────────────────────────────────────
def init_attendance_csv():
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["Student ID", "Name", "Date", "Time", "Status"])

def mark_attendance(student_id, name):
    now = datetime.now()
    with open(ATTENDANCE_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            student_id, name,
            now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), "Present"
        ])
    print(f"  PRESENT: {name} ({student_id})  {now.strftime('%H:%M:%S')}")

# ─── Face Features ────────────────────────────────────────────────────────────
def extract_features(image_bgr):
    rgb     = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = get_face_mesh().process(rgb)
    if not results.multi_face_landmarks:
        return None
    lm      = results.multi_face_landmarks[0].landmark
    indices = np.linspace(0, len(lm) - 1, 64, dtype=int)
    feat    = []
    for i in indices:
        feat.extend([lm[i].x, lm[i].y])
    arr  = np.array(feat)
    norm = np.linalg.norm(arr)
    return (arr / norm).tolist() if norm else None

def cosine_sim(a, b):
    a, b  = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0

def identify(features, students):
    best_id, best_name, best_score = "unknown", "Unknown", 0.0
    for sid, info in students.items():
        s = cosine_sim(features, info["features"])
        if s > best_score:
            best_score, best_id, best_name = s, sid, info["name"]
    if best_score >= CONFIDENCE_THRESHOLD:
        return best_id, best_name, best_score
    return "unknown", "Unknown", best_score

# ─── Drawing helpers ──────────────────────────────────────────────────────────
def draw_face_box(display, x, y, x2, y2, name, color):
    cv2.rectangle(display, (x, y), (x2, y2), color, 2)
    box_y2 = min(y2 + 30, display.shape[0])
    cv2.rectangle(display, (x, y2), (x2, box_y2), color, -1)
    cv2.putText(display, name, (x + 6, y2 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

def draw_info_panel(display, student_id, name):
    """Top-right card: name + ID + PRESENT stamp (like the reference image)."""
    ih, iw = display.shape[:2]
    pw, ph = 230, 115
    px, py = iw - pw - 10, 50

    overlay = display.copy()
    cv2.rectangle(overlay, (px - 4, py - 4), (px + pw + 4, py + ph + 4), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.60, display, 0.40, 0, display)
    cv2.rectangle(display, (px, py), (px + pw, py + ph), (0, 200, 70), 2)

    # Name (large green)
    cv2.putText(display, name,
                (px + 10, py + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 255, 100), 2)

    # Student ID
    cv2.putText(display, f"ID : {student_id}",
                (px + 10, py + 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

    # PRESENT stamp (solid green)
    cv2.rectangle(display, (px + 10, py + 68), (px + pw - 10, py + ph - 8),
                  (0, 160, 55), -1)
    cv2.putText(display, "PRESENT",
                (px + 48, py + ph - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)

def draw_present_list(display, marked_today, students):
    """Scrolling list of present students below the info panel."""
    if not marked_today:
        return
    ih, iw = display.shape[:2]
    pw     = 230
    px     = iw - pw - 10
    py     = 175
    lh     = 24
    n      = len(marked_today)
    bh     = 20 + n * lh

    overlay = display.copy()
    cv2.rectangle(overlay, (px - 4, py), (px + pw + 4, py + bh), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.60, display, 0.40, 0, display)

    cv2.putText(display, f"Present today  ({n})",
                (px + 8, py + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (100, 220, 100), 1)

    for i, sid in enumerate(marked_today):
        nm = students.get(sid, {}).get("name", sid)
        cv2.putText(display, f"  + {nm}",
                    (px + 8, py + 16 + (i + 1) * lh),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 255, 180), 1)

# ─── 1. Register ──────────────────────────────────────────────────────────────
def register_student_from_webcam():
    students = load_students()
    print("\n── Register New Student ──")
    student_id   = input("  Enter student ID  : ").strip()
    student_name = input("  Enter student name: ").strip()
    if not student_id or not student_name:
        print("  ID and name cannot be empty.")
        return

    print("  Position your face in the frame.")
    print("  Press SPACE to capture, ESC to cancel ...\n")

    cap      = cv2.VideoCapture(0)
    features = None

    with mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5) as fd:

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res     = fd.process(rgb)
            display = frame.copy()
            ih, iw  = frame.shape[:2]
            found   = False

            if res.detections:
                for det in res.detections:
                    bb = det.location_data.relative_bounding_box
                    x  = max(0, int(bb.xmin * iw))
                    y  = max(0, int(bb.ymin * ih))
                    x2 = min(iw, x + int(bb.width  * iw))
                    y2 = min(ih, y + int(bb.height * ih))
                    cv2.rectangle(display, (x, y), (x2, y2), (0, 255, 100), 2)
                    found = True

            msg = "Face detected!  Press SPACE to capture" if found \
                  else "No face found – align your face"
            col = (0, 255, 100) if found else (0, 80, 255)
            cv2.rectangle(display, (0, 0), (iw, 40), (20, 20, 20), -1)
            cv2.putText(display, msg, (10, 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.70, col, 2)
            cv2.putText(display, f"Registering: {student_name}",
                        (10, ih - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (200, 200, 200), 1)

            cv2.imshow("Register  –  SPACE=capture   ESC=cancel", display)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break
            if key == 32 and found:
                print("  Extracting features ...", end="", flush=True)
                features = extract_features(frame)
                if features:
                    print(" done!")
                    green = display.copy()
                    cv2.rectangle(green, (0, 0), (iw, ih), (0, 255, 80), 10)
                    cv2.imshow("Register  –  SPACE=capture   ESC=cancel", green)
                    cv2.waitKey(700)
                    break
                else:
                    print(" could not extract – try again.")

    cap.release()
    cv2.destroyAllWindows()

    if features:
        students[student_id] = {"name": student_name, "features": features}
        save_students(students)
        print(f"\n  Registered: {student_name}  (ID: {student_id})")
    else:
        print("\n  Registration cancelled.")

# ─── 2. Attendance (webcam) ───────────────────────────────────────────────────
def run_attendance():
    students = load_students()
    if not students:
        print("  No students registered. Please use option 1 first.")
        return

    init_attendance_csv()
    print(f"\n── Attendance Mode  ({len(students)} students) ──")
    print("  Press Q to quit.\n")

    marked_today   = {}   # sid -> timestamp
    id_cache       = {}   # face_index -> (sid, name, conf)
    frame_count    = 0
    last_panel_sid = None

    cap = cv2.VideoCapture(0)

    with mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5) as fd:

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame_count += 1
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fd_res  = fd.process(rgb)
            display = frame.copy()
            ih, iw  = frame.shape[:2]

            # top bar
            cv2.rectangle(display, (0, 0), (iw, 40), (20, 20, 20), -1)
            cv2.putText(display,
                        f"AI Attendance  |  {datetime.now().strftime('%H:%M:%S')}  |  "
                        f"Present: {len(marked_today)}/{len(students)}",
                        (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 1)

            if fd_res.detections:
                for idx, det in enumerate(fd_res.detections):
                    bb  = det.location_data.relative_bounding_box
                    x   = max(0, int(bb.xmin  * iw))
                    y   = max(0, int(bb.ymin  * ih) + 40)
                    x2  = min(iw, x + int(bb.width  * iw))
                    y2  = min(ih, y + int(bb.height * ih))

                    # Re-identify every IDENTIFY_EVERY_N frames
                    if frame_count % IDENTIFY_EVERY_N == 0 or idx not in id_cache:
                        crop = frame[max(0, y - 40):y2, x:x2]
                        if crop.size > 0:
                            feats = extract_features(crop)
                            id_cache[idx] = identify(feats, students) \
                                            if feats else ("unknown", "Unknown", 0.0)
                        else:
                            id_cache[idx] = ("unknown", "Unknown", 0.0)

                    sid, sname, conf = id_cache.get(idx, ("unknown", "Unknown", 0.0))

                    if sid != "unknown":
                        color = (0, 220, 80)
                        label = f"{sname}  {conf:.0%}"
                        last_panel_sid = sid
                        now_t = time.time()
                        if sid not in marked_today or \
                           (now_t - marked_today[sid]) > COOLDOWN_SECONDS:
                            mark_attendance(sid, sname)
                            marked_today[sid] = now_t
                    else:
                        color = (60, 60, 220)
                        label = "Unknown"

                    draw_face_box(display, x, y, x2, y2, label, color)

            # info panel + present list (top-right)
            if last_panel_sid and last_panel_sid in students:
                draw_info_panel(display, last_panel_sid,
                                students[last_panel_sid]["name"])
            draw_present_list(display, marked_today, students)

            cv2.imshow("AI Attendance System  –  press Q to quit", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n  Attendance saved -> {ATTENDANCE_FILE}")

# ─── 3. Static image ──────────────────────────────────────────────────────────
def run_on_image(image_path):
    students = load_students()
    init_attendance_csv()
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"  Cannot read image: {image_path}")
        return

    marked = []
    with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5) as fd:
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res     = fd.process(rgb)
        display = frame.copy()
        ih, iw  = frame.shape[:2]

        if res.detections:
            for det in res.detections:
                bb  = det.location_data.relative_bounding_box
                x   = max(0, int(bb.xmin  * iw))
                y   = max(0, int(bb.ymin  * ih))
                x2  = min(iw, x + int(bb.width  * iw))
                y2  = min(ih, y + int(bb.height * ih))
                crop = frame[y:y2, x:x2]
                label, color = "Unknown", (60, 60, 220)
                if crop.size > 0 and students:
                    feats = extract_features(crop)
                    if feats:
                        sid, sname, conf = identify(feats, students)
                        if sid != "unknown":
                            label = f"{sname} ({conf:.0%})"
                            color = (0, 200, 80)
                            mark_attendance(sid, sname)
                            marked.append(sname)
                draw_face_box(display, x, y, x2, y2, label, color)

            print(f"  Found {len(res.detections)} face(s). Identified: {marked or ['none']}")
        else:
            print("  No faces detected.")

    out = "annotated_" + os.path.basename(image_path)
    cv2.imwrite(out, display)
    print(f"  Saved -> {out}")

# ─── Main menu ────────────────────────────────────────────────────────────────
def main():
    print("╔══════════════════════════════════════╗")
    print("║   AI Attendance System (MediaPipe)   ║")
    print("╚══════════════════════════════════════╝")
    while True:
        print("\n  1. Register new student (webcam)")
        print("  2. Run attendance (webcam)")
        print("  3. Run attendance on image file")
        print("  4. Show today's attendance")
        print("  5. List registered students")
        print("  0. Exit")
        choice = input("\n  Choice: ").strip()

        if choice == "1":
            register_student_from_webcam()
        elif choice == "2":
            run_attendance()
        elif choice == "3":
            path = input("  Image path: ").strip().strip('"')
            run_on_image(path)
        elif choice == "4":
            if os.path.exists(ATTENDANCE_FILE):
                with open(ATTENDANCE_FILE) as f:
                    print("\n" + f.read())
            else:
                print("  No attendance recorded today.")
        elif choice == "5":
            s = load_students()
            if s:
                print(f"\n  {len(s)} registered students:")
                for sid, info in s.items():
                    print(f"    {sid:12s} -> {info['name']}")
            else:
                print("  No students registered.")
        elif choice == "0":
            print("  Goodbye!")
            break
        else:
            print("  Invalid choice.")

if __name__ == "__main__":
    main()