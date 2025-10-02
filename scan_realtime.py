import cv2
import numpy as np
import pickle
import os

# --- Konfigurasi ---
MODELS_FOLDER = 'models'
CASCADE_FILE = os.path.join(MODELS_FOLDER, 'haarcascade_frontalface_default.xml')
LBPH_MODEL_FILE = os.path.join(MODELS_FOLDER, 'lbph_model.yml')
LABELS_FILE = os.path.join(MODELS_FOLDER, 'labels.pkl')

# Threshold kepercayaan (confidence) untuk LBPH. Nilai yang lebih rendah berarti lebih percaya diri.
# Nilai di bawah 100 umumnya dianggap sebagai kecocokan yang wajar.
CONFIDENCE_THRESHOLD = 100

def run_realtime_scanner():
    """
    Fungsi utama untuk menjalankan pemindai wajah real-time menggunakan model LBPH.
    """
    # 1. Periksa apakah semua model yang diperlukan ada
    if not all(os.path.exists(f) for f in [LBPH_MODEL_FILE, LABELS_FILE, CASCADE_FILE]):
        print("Kesalahan: Model atau file cascade tidak ditemukan.")
        print("Pastikan Anda telah melatih model melalui antarmuka web ('/train').")
        return

    # 2. Muat model LBPH dan label
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(LBPH_MODEL_FILE)
        with open(LABELS_FILE, 'rb') as f:
            id_to_name_map = pickle.load(f)

        detector = cv2.CascadeClassifier(CASCADE_FILE)
        if detector.empty():
            print(f"Kesalahan: Gagal memuat file cascade classifier dari {CASCADE_FILE}")
            return

        print("Model LBPH berhasil dimuat.")
    except Exception as e:
        print(f"Gagal memuat model: {e}")
        return

    # 3. Inisialisasi webcam
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Kesalahan: Tidak dapat membuka webcam.")
        return

    print("Memulai pemindaian real-time... Tekan 'q' untuk keluar.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Prediksi wajah yang terdeteksi
            face_roi = gray[y:y+h, x:x+w]
            label_id, confidence = recognizer.predict(face_roi)

            # Tentukan nama berdasarkan tingkat kepercayaan
            if confidence < CONFIDENCE_THRESHOLD:
                name = id_to_name_map.get(label_id, "Unknown")
                display_text = f"{name} ({confidence:.2f})"
            else:
                name = "Unknown"
                display_text = f"{name} ({confidence:.2f})"

            # Gambar kotak dan teks di sekitar wajah
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, display_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow('Video Pengenalan Wajah (LBPH)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    print("Pemindaian dihentikan.")

if __name__ == '__main__':
    run_realtime_scanner()