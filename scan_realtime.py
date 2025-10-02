import cv2
import numpy as np
import pickle
import os

# --- Konfigurasi ---
MODELS_FOLDER = 'models'
CASCADE_FILE = 'haarcascade_frontalface_default.xml'
IMG_WIDTH, IMG_HEIGHT = 100, 100 # Harus sama dengan yang di app.py

# Path ke file model
PCA_MODEL_FILE = os.path.join(MODELS_FOLDER, 'pca.pkl')
SVM_MODEL_FILE = os.path.join(MODELS_FOLDER, 'svm.pkl')
TARGET_NAMES_FILE = os.path.join(MODELS_FOLDER, 'target_names.pkl')

def run_realtime_scanner():
    """
    Fungsi utama untuk menjalankan pemindai wajah real-time.
    """
    # 1. Periksa apakah semua model yang diperlukan ada
    if not all(os.path.exists(f) for f in [PCA_MODEL_FILE, SVM_MODEL_FILE, TARGET_NAMES_FILE, CASCADE_FILE]):
        print("Kesalahan: Model atau file cascade tidak ditemukan.")
        print("Pastikan Anda telah melatih model melalui antarmuka web ('/train') dan file haarcascade_frontalface_default.xml ada di direktori yang sama.")
        return

    # 2. Muat model yang telah dilatih
    try:
        with open(PCA_MODEL_FILE, 'rb') as f:
            pca = pickle.load(f)
        with open(SVM_MODEL_FILE, 'rb') as f:
            clf = pickle.load(f)
        with open(TARGET_NAMES_FILE, 'rb') as f:
            target_names = pickle.load(f)

        detector = cv2.CascadeClassifier(CASCADE_FILE)
        print("Model berhasil dimuat.")
    except Exception as e:
        print(f"Gagal memuat model: {e}")
        return

    # 3. Inisialisasi webcam
    # Angka 0 berarti webcam default. Ubah jika Anda memiliki beberapa kamera.
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Kesalahan: Tidak dapat membuka webcam.")
        return

    print("Memulai pemindaian real-time... Tekan 'q' untuk keluar.")

    while True:
        # Tangkap frame per frame dari webcam
        ret, frame = video_capture.read()
        if not ret:
            print("Tidak dapat menerima frame. Keluar...")
            break

        # Konversi frame ke grayscale untuk deteksi wajah
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Deteksi wajah dalam frame
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Loop melalui setiap wajah yang terdeteksi
        for (x, y, w, h) in faces:
            # Crop wajah dari frame grayscale
            face_gray = gray[y:y+h, x:x+w]

            # Resize wajah agar sesuai dengan ukuran input model
            face_resized = cv2.resize(face_gray, (IMG_WIDTH, IMG_HEIGHT))

            # Ubah wajah menjadi vektor fitur dan lakukan transformasi PCA
            face_vector = face_resized.flatten().reshape(1, -1)
            face_pca = pca.transform(face_vector)

            # Lakukan prediksi menggunakan model SVM
            prediction = clf.predict(face_pca)
            recognized_name = target_names[prediction[0]]

            # Ambil probabilitas prediksi
            # proba = clf.predict_proba(face_pca)
            # confidence = proba.max() * 100
            # text_to_display = f"{recognized_name} ({confidence:.2f}%)"

            # Gambar kotak di sekitar wajah yang terdeteksi pada frame asli
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Tulis nama yang dikenali di bawah kotak
            cv2.putText(frame, recognized_name, (x, y+h+20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        # Tampilkan frame yang hasilnya
        cv2.imshow('Video Pengenalan Wajah', frame)

        # Hentikan loop jika tombol 'q' ditekan
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Setelah selesai, lepaskan webcam dan tutup semua jendela
    video_capture.release()
    cv2.destroyAllWindows()
    print("Pemindaian dihentikan.")

if __name__ == '__main__':
    run_realtime_scanner()