# pyright: reportUnknownMemberType=false

from pathlib import Path

import imageio.v3 as img
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def zooming(
    citra: NDArray[np.uint8],
    faktor: float
) -> NDArray[np.uint8]:
    if not np.isfinite(faktor) or faktor <= 0:
        raise ValueError("Faktor zoom harus positif dan berhingga.")

    tinggi, lebar = citra.shape[:2]

    tinggi_baru = max(1, int(tinggi * faktor))
    lebar_baru = max(1, int(lebar * faktor))

    # Tentukan piksel sumber dengan metode nearest neighbor.
    sumber_y = np.minimum(
        (np.arange(tinggi_baru) / faktor).astype(int),
        tinggi - 1
    )

    sumber_x = np.minimum(
        (np.arange(lebar_baru) / faktor).astype(int),
        lebar - 1
    )

    hasil = citra[
        sumber_y[:, None],
        sumber_x[None, :]
    ].copy()

    return hasil


def main() -> None:
    folder = Path(__file__).resolve().parent
    file_gambar = folder / "gambar.png"

    if not file_gambar.exists():
        raise FileNotFoundError(
            "Letakkan gambar.png di folder yang sama "
            "dengan 1d_zooming.py."
        )

    citra_asli = np.asarray(
        img.imread(file_gambar, mode="RGB"),
        dtype=np.uint8
    )

    faktor = 2.0
    citra_hasil = zooming(citra_asli, faktor)

    folder_hasil = folder / "hasil_1d"
    folder_hasil.mkdir(exist_ok=True)

    img.imwrite(
        folder_hasil / "hasil_zoom_plus.png",
        citra_hasil
    )

    tinggi, lebar = citra_asli.shape[:2]
    tinggi_baru, lebar_baru = citra_hasil.shape[:2]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].imshow(citra_asli)
    axes[0].set_title(
        f"Citra Asli — {lebar} x {tinggi} Piksel"
    )
    axes[0].axis("off")

    axes[1].imshow(citra_hasil)
    axes[1].set_title(
        f"Zoom Plus 2 Kali — "
        f"{lebar_baru} x {tinggi_baru} Piksel"
    )
    axes[1].axis("off")

    fig.tight_layout()

    fig.savefig(
        folder_hasil / "perbandingan_zoom_plus.png",
        dpi=150
    )

    print("Zoom plus berhasil.")
    print(f"Ukuran asli : {lebar} x {tinggi} piksel")
    print(f"Ukuran hasil: {lebar_baru} x {tinggi_baru} piksel")
    print("Hasil disimpan di:", folder_hasil)

    plt.show()


if __name__ == "__main__":
    main()