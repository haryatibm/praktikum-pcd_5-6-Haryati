# pyright: reportUnknownMemberType=false

from pathlib import Path

import imageio.v3 as img
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def rotasi(
    citra: NDArray[np.uint8],
    sudut: float
) -> NDArray[np.uint8]:
    tinggi, lebar = citra.shape[:2]

    radian = np.radians(sudut)
    cosinus = np.cos(radian)
    sinus = np.sin(radian)

    # Kanvas persegi berdasarkan panjang diagonal citra.
    ukuran = int(np.ceil(np.hypot(tinggi, lebar)))

    hasil = np.zeros(
        (ukuran, ukuran, 3),
        dtype=np.uint8
    )

    # Pivot pada pusat gambar sumber.
    pusat_x = (lebar - 1) / 2
    pusat_y = (tinggi - 1) / 2

    # Posisi pusat pada kanvas hasil.
    pusat_hasil = (ukuran - 1) / 2

    y, x = np.indices((ukuran, ukuran))

    dx = x - pusat_hasil
    dy = y - pusat_hasil

    # Inverse mapping: cari piksel sumber untuk setiap piksel hasil.
    sumber_x = np.rint(
        cosinus * dx + sinus * dy + pusat_x
    ).astype(int)

    sumber_y = np.rint(
        -sinus * dx + cosinus * dy + pusat_y
    ).astype(int)

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


def main() -> None:
    folder = Path(__file__).resolve().parent
    file_gambar = folder / "gambar.png"

    if not file_gambar.exists():
        raise FileNotFoundError(
            "Letakkan gambar.png di folder yang sama "
            "dengan 1c_rotasi.py."
        )

    citra_asli = np.asarray(
        img.imread(file_gambar, mode="RGB"),
        dtype=np.uint8
    )

    sudut = 45
    citra_hasil = rotasi(citra_asli, sudut)

    folder_hasil = folder / "hasil_1c"
    folder_hasil.mkdir(exist_ok=True)

    img.imwrite(
        folder_hasil / "hasil_rotasi.png",
        citra_hasil
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(citra_asli)
    axes[0].set_title("Citra Asli")
    axes[0].axis("off")

    axes[1].imshow(citra_hasil)
    axes[1].set_title(
        "Rotasi 45 Derajat — Pivot Tengah"
    )
    axes[1].axis("off")

    fig.tight_layout()

    fig.savefig(
        folder_hasil / "perbandingan_rotasi.png",
        dpi=150
    )

    print("Rotasi berhasil.")
    print("Sudut: 45 derajat searah jarum jam")
    print("Pivot: titik tengah gambar")
    print("Hasil disimpan di:", folder_hasil)

    plt.show()


if __name__ == "__main__":
    main()