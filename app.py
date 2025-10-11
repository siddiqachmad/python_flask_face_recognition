import os
import pickle
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
import base64
import io
import time

app = Flask(__name__)

# --- Konfigurasi ---
UPLOAD_FOLDER = 'static/uploads'
DATASET_FOLDER = 'dataset'
MODELS_FOLDER = 'models'
CASCADE_FILE = os.path.join(MODELS_FOLDER, 'haarcascade_frontalface_default.xml')

# File untuk model LBPH dan label
LBPH_MODEL_FILE = os.path.join(MODELS_FOLDER, 'lbph_model.yml')
LABELS_FILE = os.path.join(MODELS_FOLDER, 'labels.pkl')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# --- Rute Halaman ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train')
def train_page():
    return render_template('train.html')

# --- Rute API ---
@app.route('/api/training-status', methods=['GET'])
def training_status():
    """Memeriksa apakah dataset siap untuk training."""
    try:
        subdirs = [d for d in os.scandir(DATASET_FOLDER) if d.is_dir()]
        num_classes = len(subdirs)
        ready_to_train = num_classes >= 2

        if ready_to_train:
            message = f"Dataset siap. Ditemukan data untuk {num_classes} orang."
        elif num_classes == 1:
            message = f"Dataset belum siap. Diperlukan data dari setidaknya 2 orang. Baru ada {num_classes}."
        else:
            message = "Dataset kosong. Tambahkan gambar untuk setidaknya 2 orang."

        return jsonify({
            'ready_to_train': ready_to_train,
            'num_classes': num_classes,
            'message': message
        })
    except Exception as e:
        return jsonify({'error': f'Gagal memeriksa status training: {str(e)}'}), 500

def _save_image_from_base64(base64_string, folder_path):
    """Mendekode string base64 dan menyimpannya sebagai file gambar."""
    try:
        # Pisahkan header (e.g., "data:image/jpeg;base64,")
        header, encoded = base64_string.split(",", 1)
        image_data = base64.b64decode(encoded)

        # Buat nama file unik berdasarkan timestamp
        filename = f"face_{int(time.time() * 1000)}.jpg"
        filepath = os.path.join(folder_path, filename)

        with open(filepath, "wb") as f:
            f.write(image_data)

        return filepath
    except Exception as e:
        print(f"Error decoding base64 string: {e}")
        return None

@app.route('/api/add-face', methods=['POST'])
def add_face():
    """Menyimpan gambar wajah baru dari data base64 ke dataset."""
    if 'image' not in request.form or 'name' not in request.form:
        return jsonify({'error': 'Permintaan tidak lengkap (membutuhkan nama dan gambar).'}), 400

    name = request.form.get('name', '').strip()
    base64_image = request.form.get('image')

    if not name or not base64_image:
        return jsonify({'error': 'Nama dan gambar tidak boleh kosong.'}), 400

    person_folder = os.path.join(DATASET_FOLDER, name)
    os.makedirs(person_folder, exist_ok=True)

    filepath = _save_image_from_base64(base64_image, person_folder)

    if filepath:
        return jsonify({'success': f'Gambar untuk "{name}" berhasil ditambahkan.'})
    else:
        return jsonify({'error': 'Gagal menyimpan gambar dari data base64.'}), 500

@app.route('/api/train-model', methods=['POST'])
def train_model():
    """Melatih model LBPH dari semua gambar di dataset."""
    detector = cv2.CascadeClassifier(CASCADE_FILE)
    if detector.empty():
        return jsonify({'error': 'Kesalahan Internal: Tidak dapat memuat file cascade classifier.'}), 500

    faces, ids = [], []
    label_map = {}
    current_id = 0

    for person_name in sorted(os.listdir(DATASET_FOLDER)):
        person_folder = os.path.join(DATASET_FOLDER, person_name)
        if not os.path.isdir(person_folder):
            continue

        if person_name not in label_map:
            label_map[person_name] = current_id
            current_id += 1

        person_id = label_map[person_name]

        for filename in os.listdir(person_folder):
            filepath = os.path.join(person_folder, filename)
            try:
                pil_image = Image.open(filepath).convert('L') # Konversi ke grayscale
                image_np = np.array(pil_image, 'uint8')

                detected_faces = detector.detectMultiScale(image_np)
                for (x, y, w, h) in detected_faces:
                    faces.append(image_np[y:y+h, x:x+w])
                    ids.append(person_id)
            except Exception as e:
                print(f"Gagal memproses {filepath}: {e}")

    if not faces:
        return jsonify({'error': 'Tidak ada wajah yang terdeteksi di dalam dataset.'}), 400

    if len(label_map) < 2:
        return jsonify({'error': 'Training memerlukan data dari setidaknya dua orang yang berbeda.'}), 400

    # Latih model LBPH
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))

    # Simpan model dan mapping label
    recognizer.write(LBPH_MODEL_FILE)
    with open(LABELS_FILE, 'wb') as f:
        # Kita perlu membalik map untuk lookup nanti (id -> nama)
        id_to_name_map = {v: k for k, v in label_map.items()}
        pickle.dump(id_to_name_map, f)

    return jsonify({'success': f'Model berhasil dilatih dengan {len(faces)} wajah dari {len(label_map)} orang.'})


def _image_from_base64(base64_string):
    """Membaca gambar dari string base64 ke dalam format numpy array."""
    try:
        header, encoded = base64_string.split(",", 1)
        image_data = base64.b64decode(encoded)

        # Buka gambar dari byte stream
        pil_image = Image.open(io.BytesIO(image_data)).convert('L') # Langsung grayscale
        return np.array(pil_image, 'uint8')
    except Exception as e:
        print(f"Error decoding image from base64: {e}")
        return None

@app.route('/api/recognize', methods=['POST'])
def recognize():
    """Mengenali wajah dari gambar base64 menggunakan model LBPH."""
    if not all(os.path.exists(f) for f in [LBPH_MODEL_FILE, LABELS_FILE]):
        return jsonify({'error': 'Model belum dilatih. Latih model terlebih dahulu.'}), 400

    if 'image' not in request.form:
        return jsonify({'error': 'Permintaan tidak lengkap (membutuhkan gambar).'}), 400

    base64_image = request.form.get('image')
    if not base64_image:
        return jsonify({'error': 'Data gambar tidak boleh kosong.'}), 400

    detector = cv2.CascadeClassifier(CASCADE_FILE)
    if detector.empty():
        return jsonify({'error': 'Kesalahan Internal: Tidak dapat memuat cascade classifier.'}), 500

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(LBPH_MODEL_FILE)
    with open(LABELS_FILE, 'rb') as f:
        id_to_name_map = pickle.load(f)

    try:
        image_np = _image_from_base64(base64_image)
        if image_np is None:
            return jsonify({'error': 'Format gambar tidak valid.'}), 400

        detected_faces = detector.detectMultiScale(image_np)
        if len(detected_faces) == 0:
            return jsonify({'name': 'Unknown', 'message': 'Wajah tidak terdeteksi.'})

        # Ambil wajah terbesar jika ada lebih dari satu
        (x, y, w, h) = sorted(detected_faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        face = image_np[y:y+h, x:x+w]

        label_id, confidence = recognizer.predict(face)

        if confidence < 100:
            recognized_name = id_to_name_map.get(label_id, "Unknown")
        else:
            recognized_name = "Unknown"

        return jsonify({'name': recognized_name, 'confidence': f"{confidence:.2f}"})

    except Exception as e:
        return jsonify({'error': f'Terjadi kesalahan saat pengenalan: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')