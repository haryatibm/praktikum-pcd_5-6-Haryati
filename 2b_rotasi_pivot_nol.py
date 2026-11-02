# pyright: reportUnknownMemberType=false

from pathlib import Path

import imageio.v3 as img
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def rotasi_pivot_nol(
    citra: NDArray[np.uint8],
    sudut: float
) -> NDArray[np.uint8]:
    tinggi, lebar = citra.shape[:2]

    radian = np.radians(sudut)
    cosinus = np.cos(radian)
    sinus = np.sin(radian)

    # Ukuran kanvas tetap.
    hasil = np.zeros_like(citra)

    # Koordinat gambar hasil.
    # Pivot (0,0) berada di pojok kiri atas.
    y, x = np.indices((tinggi, lebar))

    # Inverse mapping untuk rotasi searah jarum jam.
    sumber_x = np.rint(
        cosinus * x + sinus * y
    ).astype(int)

    sumber_y = np.rint(
        -sinus * x + cosinus * y
    ).astype(int)

    # Pilih koordinat yang berada dalam gambar sumber.
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
            "dengan 2b_rotasi_pivot_nol.py."
        )

    citra_asli = np.asarray(
        img.imread(file_gambar, mode="RGB"),
        dtype=np.uint8
    )

    sudut = 45
    citra_hasil = rotasi_pivot_nol(citra_asli, sudut)

    folder_hasil = folder / "hasil_2b"
    folder_hasil.mkdir(exist_ok=True)

    img.imwrite(
        folder_hasil / "hasil_rotasi_pivot_nol.png",
        citra_hasil
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].imshow(citra_asli)
    axes[0].set_title("Citra Asli")
    axes[0].axis("off")

    axes[1].imshow(citra_hasil)
    axes[1].set_title(
        "Rotasi 45 Derajat — Pivot (0,0) Pojok Kiri Atas"
    )
    axes[1].axis("off")

    fig.tight_layout()

    fig.savefig(
        folder_hasil / "perbandingan_rotasi_pivot_nol.png",
        dpi=150
    )

    print("Rotasi berhasil.")
    print("Sudut: 45 derajat searah jarum jam")
    print("Pivot: (0,0), pojok kiri atas")
    print("Ukuran kanvas tetap.")
    print("Hasil disimpan di:", folder_hasil)

    plt.show()


if __name__ == "__main__":
    main()