# Aplikasi Pengenalan Wajah dengan Flask

Aplikasi web sederhana yang dibangun menggunakan Python, Flask, OpenCV, dan Scikit-learn untuk melakukan pengenalan wajah. Aplikasi ini menyediakan antarmuka web untuk menambahkan data wajah (training) dan untuk mengenali wajah dari sebuah gambar.

## Fitur

-   **Antarmuka Web Sederhana**: Halaman terpisah untuk training dan pengenalan.
-   **Training Berbasis Dataset**: Tambahkan gambar wajah untuk setiap orang yang ingin dikenali.
-   **Model PCA + SVM**: Menggunakan Principal Component Analysis (PCA) untuk ekstraksi fitur dan Support Vector Machine (SVM) untuk klasifikasi.
-   **API Endpoints**: Endpoint yang jelas untuk menambahkan wajah, melatih model, dan melakukan pengenalan.

## Prasyarat

-   Python 3.7+
-   `pip` untuk manajemen paket

## Instalasi

1.  **Clone Repositori**
    ```bash
    git clone <url-repositori-anda>
    cd <nama-direktori-repositori>
    ```

2.  **Buat dan Aktifkan Lingkungan Virtual (Direkomendasikan)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Di Windows, gunakan `venv\Scripts\activate`
    ```

3.  **Instal Dependensi**
    Pastikan Anda berada di direktori utama proyek, lalu jalankan:
    ```bash
    pip install -r requirements.txt
    ```
    Ini akan menginstal Flask, Scikit-learn, OpenCV, dan Numpy.

4.  **Jalankan Aplikasi**
    ```bash
    python app.py
    ```
    Aplikasi akan berjalan di `http://127.0.0.1:5000`.

## Cara Penggunaan

Aplikasi ini memiliki alur kerja dua langkah utama: **Training** dan **Pengenalan**.

### 1. Training Model

Sebelum bisa mengenali wajah, Anda harus melatih model terlebih dahulu.

1.  **Buka Halaman Training**: Buka browser dan navigasikan ke `http://127.0.0.1:5000/train`.

2.  **Tambah Wajah ke Dataset**:
    -   Di bawah "Langkah 1: Tambah Wajah", masukkan nama orang yang ingin Anda tambahkan.
    -   Pilih file gambar wajah orang tersebut.
    -   Klik tombol "Tambah Wajah".
    -   Ulangi proses ini untuk beberapa gambar dari orang yang sama untuk hasil yang lebih baik. Anda juga bisa menambahkan orang lain dengan cara yang sama. Setiap gambar yang diunggah akan disimpan di folder `dataset/<nama_orang>/`.

3.  **Latih Model**:
    -   Setelah Anda selesai menambahkan semua gambar ke dataset, klik tombol "Latih Model Sekarang" di bawah "Langkah 2: Latih Model".
    -   Proses ini akan membaca semua gambar di folder `dataset`, mendeteksi wajah, melatih model PCA dan SVM, lalu menyimpan model yang telah dilatih ke dalam folder `models/`.
    -   Tunggu hingga pesan sukses muncul.

### 2. Pengenalan Wajah

Setelah model dilatih, Anda dapat menggunakannya untuk mengenali wajah.

1.  **Buka Halaman Utama**: Navigasikan ke `http://127.0.0.1:5000/`.

2.  **Unggah Gambar**:
    -   Pilih gambar yang berisi wajah seseorang yang sudah Anda latih sebelumnya.
    -   Klik tombol "Kenali".

3.  **Lihat Hasil**: Aplikasi akan menampilkan nama orang yang dikenali di bawah bagian "Hasil". Jika wajah tidak dikenali atau tidak terdeteksi, pesan yang sesuai akan ditampilkan.

### 3. Pengenalan Real-time (via Webcam)

Setelah model Anda dilatih, Anda dapat menjalankan skrip pemindaian real-time untuk mengenali wajah langsung dari webcam Anda.

1.  Pastikan Anda telah melatih model Anda menggunakan antarmuka web.
2.  Jalankan skrip berikut dari terminal di direktori utama proyek:
    ```bash
    python scan_realtime.py
    ```
3.  Sebuah jendela video akan muncul, menampilkan feed dari webcam Anda. Jika wajah yang telah dilatih terdeteksi, nama orang tersebut akan muncul di bawah kotak yang mengelilingi wajah.
4.  Tekan tombol **'q'** pada keyboard Anda untuk menutup jendela dan menghentikan skrip.

## API Endpoints

Aplikasi ini juga menyediakan beberapa endpoint API:

-   `POST /api/add-face`
    -   Menambahkan gambar baru ke dataset training.
    -   **Body**: `form-data` dengan field `name` (string) dan `file` (file gambar).

-   `POST /api/train-model`
    -   Memicu proses training model pada seluruh dataset.
    -   Tidak memerlukan body.

-   `POST /api/recognize`
    -   Mengenali wajah dari gambar yang diunggah.
    -   **Body**: `form-data` dengan field `file` (file gambar).