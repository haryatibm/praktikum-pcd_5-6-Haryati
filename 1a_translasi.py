# pyright: reportUnknownMemberType=false

from pathlib import Path

import imageio.v3 as img
import matplotlib.pyplot as plt
import numpy as np


def translasi(citra, geser_x, geser_y):
    tinggi, lebar = citra.shape[:2]

    # Siapkan gambar hasil dengan latar hitam.
    hasil = np.zeros_like(citra)

    # Koordinat setiap piksel pada gambar hasil.
    y, x = np.indices((tinggi, lebar))

    # Posisi piksel pada gambar sumber.
    sumber_x = x - geser_x
    sumber_y = y - geser_y

    valid = (
        (sumber_x >= 0)
        & (sumber_x < lebar)
        & (sumber_y >= 0)
        & (sumber_y < tinggi)
    )

    hasil[valid] = citra[
        sumber_y[valid],
        sumber_x[valid]
    ]

    return hasil


def main():
    folder = Path(__file__).resolve().parent
    file_gambar = folder / "gambar.png"

    if not file_gambar.exists():
        raise FileNotFoundError(
            "File gambar.png tidak ditemukan. "
            "Simpan foto di folder yang sama dengan program."
        )

    citra_asli = img.imread(file_gambar, mode="RGB")

    # Sesuai contoh tugas:
    # 50 piksel ke kanan dan 300 piksel ke atas.
    citra_hasil = translasi(citra_asli, 50, -300)

    folder_hasil = folder / "hasil_1a"
    folder_hasil.mkdir(exist_ok=True)

    img.imwrite(
        folder_hasil / "hasil_translasi.png",
        citra_hasil
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].imshow(citra_asli)
    axes[0].set_title("Citra Asli")
    axes[0].axis("off")

    axes[1].imshow(citra_hasil)
    axes[1].set_title(
        "Translasi: 50 Piksel ke Kanan dan 300 Piksel ke Atas"
    )
    axes[1].axis("off")

    fig.tight_layout()

    fig.savefig(
        folder_hasil / "perbandingan_translasi.png",
        dpi=150
    )

    print("Translasi berhasil.")
    print("Hasil disimpan di:", folder_hasil)

    plt.show()


if __name__ == "__main__":
    main()