# 🚗 Used Car Price Prediction using Deep Artificial Neural Networks (ANN)

Repositori ini berisi implementasi *end-to-end machine learning* dan *deep learning* untuk memprediksi harga jual mobil bekas (*selling price*) berdasarkan spesifikasi teknis dan riwayat kepemilikan kendaraan menggunakan **TensorFlow / Keras**.

---

## 📌 Ringkasan Proyek
Penentuan harga pasar mobil bekas dipengaruhi oleh berbagai faktor non-linear seperti tahun produksi, jarak tempuh, tenaga mesin (*max power*), serta torsi dan RPM. Proyek ini berfokus pada:
1. **Data Cleaning & Regex Feature Extraction**: Mengurai kolom gabungan torsi menjadi variabel numerik terpisah (`torque_nm` dan `torque_rpm`).
2. **Preprocessing**: Imputasi nilai hilang (*missing values*), *Label Encoding* fitur kategorikal, dan standarisasi fitur numerik menggunakan `StandardScaler`.
3. **Model Comparison**: Membandingkan performa arsitektur *Deep ANN* bertingkat (aktivasi Sigmoid) terhadap arsitektur modifikasi (aktivasi ReLU, Dropout, Early Stopping, dan ReduceLROnPlateau).

---

## 📊 Dataset & Fitur
Data bersumber dari file tabular `2B.parquet` yang memuat informasi spesifikasi mobil:
- **Target:** `selling_price` (harga jual mobil)
- **Fitur Utama:** `name`, `year`, `mileage`, `engine`, `max_power`, `seats`, `fuel`, `transmission`, `owner`, `seller_type`, `Region`, `State or Province`, `City`.
- **Fitur Baru (Hasil Ekstraksi):**
  - `torque_nm`: Besaran torsi mesin (dalam Nm).
  - `torque_rpm`: Putaran mesin saat torsi puncak tercapai (dalam rpm).

---

## 🛠️ Alur Preprocessing
1. **Pemeriksaan & Imputasi Missing Values:** Median untuk variabel numerik dan modus untuk variabel kategorikal.
2. **Pembersihan Torsi:** Menggunakan ekspresi reguler (*regex*) `r'(\d+)\s*Nm'` dan `r'@\s*(\d+)\s*rpm'`.
3. **Encoding:** `LabelEncoder` untuk seluruh kolom kategorikal.
4. **Data Splitting:** Pembagian dataset menjadi 70% Train, 15% Validation, dan 15% Test.
5. **Normalisasi:** `StandardScaler` diaplikasikan secara terpisah pada fitur independen ($X$) dan target ($y$).

---

## 🧠 Eksperimen & Arsitektur Model

### 1. Baseline Model (Sigmoid Deep ANN)
- **Input:** 14 fitur
- **Hidden Layers:**
  - Dense (512 neuron, aktivasi `sigmoid`)
  - Dense (256 neuron, aktivasi `sigmoid`)
  - Dense (128 neuron, aktivasi `sigmoid`)
- **Output:** Dense (1 neuron)
- **Optimizer & Loss:** Adam ($lr = 0.01$), Loss: MSE, Metric: MAE
- **Epochs:** 30

### 2. Modified Model (ReLU + Regularization)
- **Input:** 14 fitur
- **Hidden Layers:**
  - Dense(256, `relu`) + Dropout(0.2)
  - Dense(128, `relu`) + Dropout(0.2)
  - Dense(64, `relu`) + Dropout(0.2)
  - Dense(32, `relu`)
- **Output:** Dense (1 neuron, linear)
- **Call-backs:** `EarlyStopping` (patience=5) & `ReduceLROnPlateau` (factor=0.5, patience=3)
- **Optimizer & Loss:** Adam ($lr = 0.0001$), Loss: MAE
- **Epochs:** 100

---

## 📈 Hasil Evaluasi Performa (Test Set)

| Metrik Evaluasi | Baseline Model (Sigmoid) | Modified Model (ReLU + Dropout) |
| :--- | :---: | :---: |
| **Test Loss** | **0.0646** | 0.1973 |
| **Test MAE** | **0.1644** | 0.1973 |
| **$R^2$ Score** | **0.9277 (92.8%)** | 0.8352 (83.5%) |

> **Analisis:** Model dasar dengan aktivasi bertingkat menghasilkan akurasi yang lebih unggul ($R^2 \approx 92.8\%$) dalam menangkap dinamika regresi harga mobil dibandingkan model modifikasi pada konfigurasi parameter ini.

---

## 📁 Struktur File
```text
├── 2B.parquet              # Dataset mentah
├── 2BSolution.ipynb        # Jupyter Notebook eksperimen
└── README.md               # Dokumentasi proyek
```
