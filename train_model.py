import os
import face_recognition
import pickle

def train_model():
    dataset_dir = 'dataset'
    known_encodings = []
    known_names = []
    known_ids = []
    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        name_id = folder.split('_')
        name = name_id[0]
        emp_id = name_id[1] if len(name_id) > 1 else ''
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            image = face_recognition.load_image_file(img_path)
            boxes = face_recognition.face_locations(image)
            encodings = face_recognition.face_encodings(image, boxes)
            for encoding in encodings:
                known_encodings.append(encoding)
                known_names.append(name)
                known_ids.append(emp_id)
    data = {'encodings': known_encodings, 'names': known_names, 'ids': known_ids}
    with open('encodings.pkl', 'wb') as f:
        pickle.dump(data, f)
    print(f"Training complete. {len(known_encodings)} faces encoded.")
