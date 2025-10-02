import os
import pickle
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from PIL import Image
from time import time

app = Flask(__name__)

# --- Konfigurasi ---
UPLOAD_FOLDER = 'static/uploads'
DATASET_FOLDER = 'dataset'
MODELS_FOLDER = 'models'
CASCADE_FILE = os.path.join(MODELS_FOLDER, 'haarcascade_frontalface_default.xml')

# Dimensi untuk resize gambar wajah
IMG_WIDTH, IMG_HEIGHT = 100, 100

# Komponen PCA
N_COMPONENTS = 15

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Pastikan direktori ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Path untuk file model yang akan disimpan
PCA_MODEL_FILE = os.path.join(MODELS_FOLDER, 'pca.pkl')
SVM_MODEL_FILE = os.path.join(MODELS_FOLDER, 'svm.pkl')
TARGET_NAMES_FILE = os.path.join(MODELS_FOLDER, 'target_names.pkl')

# Inisialisasi detektor wajah
detector = cv2.CascadeClassifier(CASCADE_FILE)

# --- Rute Halaman ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train')
def train_page():
    return render_template('train.html')

# --- Rute API ---
@app.route('/api/add-face', methods=['POST'])
def add_face():
    """Menyimpan gambar wajah baru ke dataset untuk training."""
    if 'file' not in request.files or 'name' not in request.form:
        return jsonify({'error': 'Permintaan tidak lengkap'}), 400

    file = request.files['file']
    name = request.form.get('name', '').strip()

    if file.filename == '' or name == '':
        return jsonify({'error': 'File atau nama tidak boleh kosong'}), 400

    person_folder = os.path.join(DATASET_FOLDER, name)
    os.makedirs(person_folder, exist_ok=True)

    # Simpan file ke folder orang tersebut
    filepath = os.path.join(person_folder, file.filename)
    file.save(filepath)

    return jsonify({'success': f'Gambar untuk "{name}" berhasil ditambahkan.'})


@app.route('/api/train-model', methods=['POST'])
def train_model():
    """Melatih model dari semua gambar di dataset."""
    t0 = time()
    print("Mulai proses training model...")

    X, y, target_names = [], [], []
    label_count = 0

    # Membaca semua gambar dari dataset
    for person_name in os.listdir(DATASET_FOLDER):
        person_folder = os.path.join(DATASET_FOLDER, person_name)
        if not os.path.isdir(person_folder):
            continue

        target_names.append(person_name)
        for filename in os.listdir(person_folder):
            filepath = os.path.join(person_folder, filename)
            try:
                # Buka gambar, konversi ke grayscale
                pil_image = Image.open(filepath).convert('L')
                image_np = np.array(pil_image, 'uint8')

                # Deteksi wajah
                faces = detector.detectMultiScale(image_np)
                for (x, y_face, w, h) in faces:
                    # Crop dan resize wajah, lalu flatten
                    face_resized = cv2.resize(image_np[y_face:y_face+h, x:x+w], (IMG_WIDTH, IMG_HEIGHT))
                    X.append(face_resized.flatten())
                    y.append(label_count)
            except Exception as e:
                print(f"Gagal memproses {filepath}: {e}")
        label_count += 1

    if not X:
        return jsonify({'error': 'Dataset kosong atau tidak ada wajah yang terdeteksi. Tambahkan beberapa wajah terlebih dahulu.'}), 400

    n_samples = len(X)
    n_classes = len(target_names)
    print(f"Total sampel: {n_samples}, Total kelas: {n_classes}")

    # 1. Latih PCA
    print("Melatih PCA...")
    # Sesuaikan n_components jika jumlah sampel lebih sedikit
    actual_n_components = min(N_COMPONENTS, n_samples, len(X[0]))
    pca = PCA(n_components=actual_n_components, whiten=True).fit(X)

    # 2. Transformasi data menggunakan PCA
    X_pca = pca.transform(X)

    # 3. Latih SVM Classifier
    print("Melatih SVM...")
    # Mungkin perlu GridSearchCV di sini untuk hasil terbaik, tapi kita gunakan parameter default untuk kecepatan
    clf = SVC(kernel='rbf', class_weight='balanced', probability=True)
    clf.fit(X_pca, y)

    # 4. Simpan model
    with open(PCA_MODEL_FILE, 'wb') as f:
        pickle.dump(pca, f)
    with open(SVM_MODEL_FILE, 'wb') as f:
        pickle.dump(clf, f)
    with open(TARGET_NAMES_FILE, 'wb') as f:
        pickle.dump(target_names, f)

    print(f"Training selesai dalam {time() - t0:.3f}s")
    return jsonify({'success': f'Model berhasil dilatih dengan {n_samples} sampel dari {n_classes} orang.'})


@app.route('/api/recognize', methods=['POST'])
def recognize():
    """Mengenali wajah dari gambar menggunakan model yang telah dilatih."""
    if 'file' not in request.files:
        return jsonify({'error': 'File tidak ada'}), 400

    # Periksa apakah model sudah ada
    if not all(os.path.exists(f) for f in [PCA_MODEL_FILE, SVM_MODEL_FILE, TARGET_NAMES_FILE]):
        return jsonify({'error': 'Model belum dilatih. Silakan latih model terlebih dahulu.'}), 400

    # Muat model
    with open(PCA_MODEL_FILE, 'rb') as f:
        pca = pickle.load(f)
    with open(SVM_MODEL_FILE, 'rb') as f:
        clf = pickle.load(f)
    with open(TARGET_NAMES_FILE, 'rb') as f:
        target_names = pickle.load(f)

    file = request.files['file']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    try:
        file.save(filepath)

        pil_image = Image.open(filepath).convert('L')
        image_np = np.array(pil_image, 'uint8')

        faces = detector.detectMultiScale(image_np)
        if len(faces) == 0:
            return jsonify({'name': 'Unknown', 'message': 'Wajah tidak terdeteksi.'})

        # Ambil wajah pertama yang terdeteksi
        (x, y, w, h) = faces[0]
        face_resized = cv2.resize(image_np[y:y+h, x:x+w], (IMG_WIDTH, IMG_HEIGHT))

        # Transformasi dan prediksi
        face_pca = pca.transform(face_resized.flatten().reshape(1, -1))
        prediction = clf.predict(face_pca)

        recognized_name = target_names[prediction[0]]

        return jsonify({'name': recognized_name})

    except Exception as e:
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')