import cv2
import numpy as np
import pickle
import os
import mediapipe as mp
from scipy.spatial import distance as dist

# --- Konfigurasi ---
MODELS_FOLDER = 'models'
LBPH_MODEL_FILE = os.path.join(MODELS_FOLDER, 'lbph_model.yml')
LABELS_FILE = os.path.join(MODELS_FOLDER, 'labels.pkl')

# --- Konfigurasi Liveness Detection ---
# Threshold untuk Eye Aspect Ratio (EAR)
EAR_THRESH = 0.25
# Jumlah frame berturut-turut di mana EAR harus di bawah threshold untuk dianggap kedipan
EAR_CONSEC_FRAMES = 3

def eye_aspect_ratio(eye):
    """Menghitung Eye Aspect Ratio (EAR) dari landmark mata."""
    # Jarak vertikal antara kelopak mata
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Jarak horizontal antara sudut mata
    C = dist.euclidean(eye[0], eye[3])
    # Hitung EAR
    ear = (A + B) / (2.0 * C)
    return ear

def run_realtime_scanner():
    """
    Fungsi utama untuk menjalankan pemindai wajah real-time dengan deteksi kedipan.
    """
    # 1. Muat model LBPH dan label
    if not all(os.path.exists(f) for f in [LBPH_MODEL_FILE, LABELS_FILE]):
        print("Kesalahan: Model LBPH tidak ditemukan.")
        print("Pastikan Anda telah melatih model melalui antarmuka web.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(LBPH_MODEL_FILE)
    with open(LABELS_FILE, 'rb') as f:
        id_to_name_map = pickle.load(f)
    print("Model LBPH berhasil dimuat.")

    # 2. Inisialisasi MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Landmark mata kiri dan kanan dari MediaPipe
    (lStart, lEnd) = mp_face_mesh.FACEMESH_LEFT_EYE_TESSELATION[0]
    (rStart, rEnd) = mp_face_mesh.FACEMESH_RIGHT_EYE_TESSELATION[0]
    # Sederhanakan: kita hanya butuh 6 titik utama per mata untuk EAR
    # [33, 160, 158, 133, 153, 144] -> Kiri
    # [362, 385, 387, 263, 373, 380] -> Kanan

    # 3. Inisialisasi variabel liveness
    blink_counter = 0
    liveness_detected = False
    recognition_display_timer = 0
    recognized_name = "Unknown"

    # 4. Inisialisasi webcam
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Kesalahan: Tidak dapat membuka webcam.")
        return
    print("Memulai pemindaian real-time dengan deteksi liveness... Tekan 'q' untuk keluar.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        h, w, _ = frame.shape

        # Konversi BGR ke RGB untuk MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark

            # Ekstrak 6 titik landmark untuk setiap mata
            left_eye_pts = np.array([[face_landmarks[p].x * w, face_landmarks[p].y * h] for p in [33, 160, 158, 133, 153, 144]], dtype=np.int32)
            right_eye_pts = np.array([[face_landmarks[p].x * w, face_landmarks[p].y * h] for p in [362, 385, 387, 263, 373, 380]], dtype=np.int32)

            # Hitung EAR
            left_ear = eye_aspect_ratio(left_eye_pts)
            right_ear = eye_aspect_ratio(right_eye_pts)
            ear = (left_ear + right_ear) / 2.0

            # Logika deteksi kedipan
            if ear < EAR_THRESH:
                blink_counter += 1
            else:
                if blink_counter >= EAR_CONSEC_FRAMES:
                    liveness_detected = True
                    recognition_display_timer = 30 # Tampilkan hasil selama 30 frame
                blink_counter = 0

            # Jika liveness terdeteksi, lakukan pengenalan
            if liveness_detected:
                # Dapatkan bounding box dari landmark
                x_coords = [lm.x * w for lm in face_landmarks]
                y_coords = [lm.y * h for lm in face_landmarks]
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))

                face_roi = rgb_frame[y_min:y_max, x_min:x_max]

                if face_roi.size != 0:
                    label_id, confidence = recognizer.predict(face_roi)
                    if confidence < 100:
                        recognized_name = f"{id_to_name_map.get(label_id, 'Unknown')} ({confidence:.2f})"
                    else:
                        recognized_name = f"Unknown ({confidence:.2f})"

            # Tampilkan status di layar
            if liveness_detected:
                cv2.putText(frame, "Liveness Terdeteksi", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Hasil: {recognized_name}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                recognition_display_timer -= 1
                if recognition_display_timer <= 0:
                    liveness_detected = False # Reset setelah timer habis
            else:
                cv2.putText(frame, "Menunggu Kedipan...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Tidak ada wajah terdeteksi", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        cv2.imshow('Liveness Detection & Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("Pemindaian dihentikan.")

if __name__ == '__main__':
    run_realtime_scanner()