"""
========================================================
TUGAS KELOMPOK 5 — BAGIAN KAYOO
Implementasi Deteksi Tepi: Sobel, Roberts, dan Kompas
Mata Kuliah: Analisis dan Desain Citra Berbasis Komputasi
========================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ============================================================
# KONFIGURASI CITRA INPUT
# ============================================================
# Ganti path sesuai lokasi citra yang digunakan kelompok
IMAGE1_PATH = 'images/citra1.jpg'   # Ganti dengan citra pertama
IMAGE2_PATH = 'images/citra2.jpg'   # Ganti dengan citra kedua

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# FUNGSI BANTUAN: LOAD & KONVERSI CITRA
# ============================================================

def load_grayscale(path):
    """Membaca citra dan mengkonversi ke grayscale"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Citra tidak ditemukan: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


# ============================================================
# METODE 1: DETEKSI TEPI SOBEL
# ============================================================

def sobel_detection(gray_img):
    """
    Deteksi tepi menggunakan operator Sobel.
    
    Kernel Sobel:
        Gx = [-1  0  +1]    Gy = [-1  -2  -1]
             [-2  0  +2]         [ 0   0   0]
             [-1  0  +1]         [+1  +2  +1]
    
    Magnitude: G = sqrt(Gx^2 + Gy^2)
    """
    # Konversi ke float64 untuk presisi perhitungan
    gray_float = gray_img.astype(np.float64)
    
    # Hitung gradien arah X dan Y
    sobel_x = cv2.Sobel(gray_float, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal
    sobel_y = cv2.Sobel(gray_float, cv2.CV_64F, 0, 1, ksize=3)  # Vertikal
    
    # Magnitude total gradien
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalisasi ke rentang 0-255
    sobel_magnitude = np.clip(sobel_magnitude, 0, 255).astype(np.uint8)
    
    return sobel_magnitude, sobel_x, sobel_y


def visualize_sobel(gray1, gray2, result1_mag, result1_x, result1_y,
                    result2_mag, result2_x, result2_y):
    """Visualisasi hasil deteksi tepi Sobel untuk kedua citra"""
    
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('DETEKSI TEPI SOBEL\nKelompok 5 — Bagian Kayoo', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.3)
    
    titles_row = ['Citra Asli (Grayscale)', 'Sobel-X (Horizontal)', 
                  'Sobel-Y (Vertikal)', 'Hasil Sobel (Magnitude)']
    
    # Baris 1: Citra 1
    for j, (img, title) in enumerate(zip(
        [gray1, np.abs(result1_x).clip(0,255).astype(np.uint8),
         np.abs(result1_y).clip(0,255).astype(np.uint8), result1_mag],
        titles_row
    )):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(img, cmap='gray')
        ax.set_title(f'CITRA 1\n{title}', fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # Baris 2: Citra 2
    for j, (img, title) in enumerate(zip(
        [gray2, np.abs(result2_x).clip(0,255).astype(np.uint8),
         np.abs(result2_y).clip(0,255).astype(np.uint8), result2_mag],
        titles_row
    )):
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(img, cmap='gray')
        ax.set_title(f'CITRA 2\n{title}', fontsize=10, fontweight='bold')
        ax.axis('off')
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'hasil_sobel.png'), 
                dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Hasil Sobel disimpan: output/hasil_sobel.png")


# ============================================================
# METODE 2: DETEKSI TEPI ROBERTS
# ============================================================

def roberts_detection(gray_img):
    """
    Deteksi tepi menggunakan operator Roberts Cross.
    
    Kernel Roberts:
        Gx = [+1   0]    Gy = [ 0  +1]
             [ 0  -1]         [-1   0]
    
    Magnitude: G = sqrt(Gx^2 + Gy^2)
    """
    gray_float = gray_img.astype(np.float64)
    
    # Definisi kernel Roberts Cross
    kernel_x = np.array([[1,  0],
                         [0, -1]], dtype=np.float64)
    kernel_y = np.array([[0,  1],
                         [-1, 0]], dtype=np.float64)
    
    # Konvolusi dengan kernel Roberts
    roberts_x = cv2.filter2D(gray_float, cv2.CV_64F, kernel_x)
    roberts_y = cv2.filter2D(gray_float, cv2.CV_64F, kernel_y)
    
    # Magnitude gradien
    roberts_magnitude = np.sqrt(roberts_x**2 + roberts_y**2)
    roberts_magnitude = np.clip(roberts_magnitude, 0, 255).astype(np.uint8)
    
    return roberts_magnitude, roberts_x, roberts_y


def visualize_roberts(gray1, gray2, result1_mag, result1_x, result1_y,
                      result2_mag, result2_x, result2_y):
    """Visualisasi hasil deteksi tepi Roberts untuk kedua citra"""
    
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('DETEKSI TEPI ROBERTS\nKelompok 5 — Bagian Kayoo', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.3)
    
    titles_row = ['Citra Asli (Grayscale)', 'Roberts-X (Diagonal 45°)', 
                  'Roberts-Y (Diagonal 135°)', 'Hasil Roberts (Magnitude)']
    
    # Baris 1: Citra 1
    for j, (img, title) in enumerate(zip(
        [gray1, np.abs(result1_x).clip(0,255).astype(np.uint8),
         np.abs(result1_y).clip(0,255).astype(np.uint8), result1_mag],
        titles_row
    )):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(img, cmap='gray')
        ax.set_title(f'CITRA 1\n{title}', fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # Baris 2: Citra 2
    for j, (img, title) in enumerate(zip(
        [gray2, np.abs(result2_x).clip(0,255).astype(np.uint8),
         np.abs(result2_y).clip(0,255).astype(np.uint8), result2_mag],
        titles_row
    )):
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(img, cmap='gray')
        ax.set_title(f'CITRA 2\n{title}', fontsize=10, fontweight='bold')
        ax.axis('off')
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'hasil_roberts.png'), 
                dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Hasil Roberts disimpan: output/hasil_roberts.png")


# ============================================================
# METODE 3: DETEKSI TEPI KOMPAS (KIRSCH COMPASS)
# ============================================================

def compass_detection(gray_img):
    """
    Deteksi tepi menggunakan operator Kompas (Kirsch Compass Operator).
    
    Menggunakan 8 kernel yang merepresentasikan 8 arah mata angin:
    N (Utara), NE (Timur Laut), E (Timur), SE (Tenggara),
    S (Selatan), SW (Barat Daya), W (Barat), NW (Barat Laut)
    
    Tepi = nilai maksimum dari semua 8 respons arah
    """
    gray_float = gray_img.astype(np.float64)
    
    # 8 Kernel Kompas Kirsch — berurutan N, NE, E, SE, S, SW, W, NW
    kernels = {
        'N  (Utara)':       np.array([[ 5,  5,  5],
                                       [-3,  0, -3],
                                       [-3, -3, -3]], dtype=np.float64),
        'NE (Timur Laut)':  np.array([[-3,  5,  5],
                                       [-3,  0,  5],
                                       [-3, -3, -3]], dtype=np.float64),
        'E  (Timur)':       np.array([[-3, -3,  5],
                                       [-3,  0,  5],
                                       [-3, -3,  5]], dtype=np.float64),
        'SE (Tenggara)':    np.array([[-3, -3, -3],
                                       [-3,  0,  5],
                                       [-3,  5,  5]], dtype=np.float64),
        'S  (Selatan)':     np.array([[-3, -3, -3],
                                       [-3,  0, -3],
                                       [ 5,  5,  5]], dtype=np.float64),
        'SW (Barat Daya)':  np.array([[-3, -3, -3],
                                       [ 5,  0, -3],
                                       [ 5,  5, -3]], dtype=np.float64),
        'W  (Barat)':       np.array([[ 5, -3, -3],
                                       [ 5,  0, -3],
                                       [ 5, -3, -3]], dtype=np.float64),
        'NW (Barat Laut)':  np.array([[ 5,  5, -3],
                                       [ 5,  0, -3],
                                       [-3, -3, -3]], dtype=np.float64),
    }
    
    responses = {}
    response_arrays = []
    
    # Konvolusi dengan setiap kernel
    for direction, kernel in kernels.items():
        response = cv2.filter2D(gray_float, cv2.CV_64F, kernel)
        abs_response = np.abs(response)
        responses[direction] = abs_response
        response_arrays.append(abs_response)
    
    # Ambil nilai maksimum dari semua 8 arah sebagai hasil akhir
    compass_result = np.maximum.reduce(response_arrays)
    compass_result = np.clip(compass_result, 0, 255).astype(np.uint8)
    
    return compass_result, responses


def visualize_compass(gray1, gray2, result1, responses1, result2, responses2):
    """Visualisasi hasil deteksi tepi Kompas — termasuk 8 arah"""
    
    # --- Visualisasi 8 arah untuk Citra 1 ---
    fig1, axes1 = plt.subplots(2, 5, figsize=(25, 10))
    fig1.suptitle('DETEKSI TEPI KOMPAS — CITRA 1 (8 Arah Mata Angin)\nKelompok 5 — Bagian Kayoo',
                  fontsize=14, fontweight='bold')
    
    axes1[0, 0].imshow(gray1, cmap='gray')
    axes1[0, 0].set_title('Citra 1 Asli\n(Grayscale)', fontweight='bold')
    axes1[0, 0].axis('off')
    
    for idx, (direction, resp) in enumerate(responses1.items()):
        row = (idx + 1) // 5
        col = (idx + 1) % 5
        resp_norm = np.clip(resp, 0, 255).astype(np.uint8)
        axes1[row, col].imshow(resp_norm, cmap='gray')
        axes1[row, col].set_title(f'Arah {direction}', fontsize=9)
        axes1[row, col].axis('off')
    
    axes1[1, 4].imshow(result1, cmap='gray')
    axes1[1, 4].set_title('HASIL AKHIR\n(Maksimum 8 Arah)', fontweight='bold', color='red')
    axes1[1, 4].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'kompas_8arah_citra1.png'), 
                dpi=150, bbox_inches='tight')
    plt.show()
    
    # --- Visualisasi 8 arah untuk Citra 2 ---
    fig2, axes2 = plt.subplots(2, 5, figsize=(25, 10))
    fig2.suptitle('DETEKSI TEPI KOMPAS — CITRA 2 (8 Arah Mata Angin)\nKelompok 5 — Bagian Kayoo',
                  fontsize=14, fontweight='bold')
    
    axes2[0, 0].imshow(gray2, cmap='gray')
    axes2[0, 0].set_title('Citra 2 Asli\n(Grayscale)', fontweight='bold')
    axes2[0, 0].axis('off')
    
    for idx, (direction, resp) in enumerate(responses2.items()):
        row = (idx + 1) // 5
        col = (idx + 1) % 5
        resp_norm = np.clip(resp, 0, 255).astype(np.uint8)
        axes2[row, col].imshow(resp_norm, cmap='gray')
        axes2[row, col].set_title(f'Arah {direction}', fontsize=9)
        axes2[row, col].axis('off')
    
    axes2[1, 4].imshow(result2, cmap='gray')
    axes2[1, 4].set_title('HASIL AKHIR\n(Maksimum 8 Arah)', fontweight='bold', color='red')
    axes2[1, 4].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'kompas_8arah_citra2.png'), 
                dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Hasil Kompas disimpan: output/kompas_8arah_citra1.png & citra2.png")


# ============================================================
# PERBANDINGAN AKHIR: SOBEL vs ROBERTS vs KOMPAS
# ============================================================

def visualize_comparison(gray1, gray2, sobel1, roberts1, compass1,
                         sobel2, roberts2, compass2):
    """Visualisasi perbandingan seluruh metode bagian Kayoo"""
    
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    fig.suptitle('PERBANDINGAN METODE DETEKSI TEPI — BAGIAN KAYOO\n'
                 'Sobel | Roberts | Kompas\n'
                 'Kelompok 5', 
                 fontsize=15, fontweight='bold', y=0.99)
    
    methods = ['Citra Asli (Grayscale)', 'Sobel', 'Roberts', 'Kompas']
    images_row1 = [gray1, sobel1, roberts1, compass1]
    images_row2 = [gray2, sobel2, roberts2, compass2]
    
    colors = ['black', '#1a6fb5', '#c0392b', '#27ae60']
    
    for j, (img, method, color) in enumerate(zip(images_row1, methods, colors)):
        axes[0, j].imshow(img, cmap='gray')
        axes[0, j].set_title(f'CITRA 1\n{method}', fontsize=12, 
                              fontweight='bold', color=color)
        axes[0, j].axis('off')
    
    for j, (img, method, color) in enumerate(zip(images_row2, methods, colors)):
        axes[1, j].imshow(img, cmap='gray')
        axes[1, j].set_title(f'CITRA 2\n{method}', fontsize=12, 
                              fontweight='bold', color=color)
        axes[1, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'perbandingan_kayoo.png'), 
                dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Perbandingan disimpan: output/perbandingan_kayoo.png")


def print_analysis(gray1_name, gray2_name, sobel1, roberts1, compass1,
                   sobel2, roberts2, compass2):
    """Analisis statistik kuantitatif hasil deteksi tepi"""
    
    print("\n" + "="*60)
    print("ANALISIS KUANTITATIF — BAGIAN KAYOO")
    print("="*60)
    
    methods = ['Sobel', 'Roberts', 'Kompas']
    results_1 = [sobel1, roberts1, compass1]
    results_2 = [sobel2, roberts2, compass2]
    
    print(f"\n{'Metode':<12} {'Citra':<10} {'Rata-Rata':<12} {'Std Dev':<12} {'Piksel Tepi':<15}")
    print("-"*60)
    
    for method, r1, r2 in zip(methods, results_1, results_2):
        # Ambang batas sederhana untuk menghitung piksel tepi
        threshold = 50
        edge_pixels1 = np.sum(r1 > threshold)
        edge_pixels2 = np.sum(r2 > threshold)
        
        print(f"{method:<12} {gray1_name:<10} {r1.mean():<12.2f} {r1.std():<12.2f} {edge_pixels1:<15}")
        print(f"{method:<12} {gray2_name:<10} {r2.mean():<12.2f} {r2.std():<12.2f} {edge_pixels2:<15}")
        print()
    
    print("="*60)
    print("KESIMPULAN ANALISIS:")
    print("- Sobel: Tepi jelas dan tahan noise (kernel 3×3 berbobot)")
    print("- Roberts: Tepi tipis dan tajam, sensitif noise (kernel 2×2)")
    print("- Kompas: Mendeteksi tepi 8 arah, paling detail namun lebih lambat")
    print("="*60)


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("TUGAS KELOMPOK 5 — BAGIAN KAYOO")
    print("Deteksi Tepi: Sobel, Roberts, Kompas")
    print("="*60)
    
    # --- Load Citra ---
    print("\n[1] Memuat citra input...")
    try:
        img1_bgr, gray1 = load_grayscale(IMAGE1_PATH)
        img2_bgr, gray2 = load_grayscale(IMAGE2_PATH)
        print(f"    ✅ Citra 1: {IMAGE1_PATH} — ukuran {gray1.shape}")
        print(f"    ✅ Citra 2: {IMAGE2_PATH} — ukuran {gray2.shape}")
    except FileNotFoundError as e:
        print(f"    ❌ Error: {e}")
        print("    → Pastikan folder 'images/' berisi citra1.jpg dan citra2.jpg")
        exit(1)
    
    # --- Sobel ---
    print("\n[2] Menjalankan Deteksi Tepi SOBEL...")
    sobel1, sx1, sy1 = sobel_detection(gray1)
    sobel2, sx2, sy2 = sobel_detection(gray2)
    visualize_sobel(gray1, gray2, sobel1, sx1, sy1, sobel2, sx2, sy2)
    
    # --- Roberts ---
    print("\n[3] Menjalankan Deteksi Tepi ROBERTS...")
    roberts1, rx1, ry1 = roberts_detection(gray1)
    roberts2, rx2, ry2 = roberts_detection(gray2)
    visualize_roberts(gray1, gray2, roberts1, rx1, ry1, roberts2, rx2, ry2)
    
    # --- Kompas ---
    print("\n[4] Menjalankan Deteksi Tepi KOMPAS (8 Arah)...")
    compass1, resp1 = compass_detection(gray1)
    compass2, resp2 = compass_detection(gray2)
    visualize_compass(gray1, gray2, compass1, resp1, compass2, resp2)
    
    # --- Perbandingan Akhir ---
    print("\n[5] Membuat visualisasi perbandingan akhir...")
    visualize_comparison(gray1, gray2, sobel1, roberts1, compass1,
                         sobel2, roberts2, compass2)
    
    # --- Analisis ---
    print_analysis('Citra1', 'Citra2', sobel1, roberts1, compass1,
                   sobel2, roberts2, compass2)
    
    print("\n✅ SELESAI! Semua output disimpan di folder 'output/'")
