# pyright: reportUnknownMemberType=false

from pathlib import Path

import imageio.v3 as img
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def zoom_minus(
    citra: NDArray[np.uint8],
    faktor: float = 0.5
) -> NDArray[np.uint8]:
    if not np.isfinite(faktor) or not 0 < faktor < 1:
        raise ValueError(
            "Faktor zoom minus harus lebih dari 0 dan kurang dari 1."
        )

    tinggi, lebar = citra.shape[:2]

    tinggi_baru = max(1, int(tinggi * faktor))
    lebar_baru = max(1, int(lebar * faktor))

    # Pilih piksel sumber menggunakan nearest neighbor.
    sumber_y = np.minimum(
        (np.arange(tinggi_baru) / faktor).astype(int),
        tinggi - 1
    )

    sumber_x = np.minimum(
        (np.arange(lebar_baru) / faktor).astype(int),
        lebar - 1
    )

    return citra[
        sumber_y[:, None],
        sumber_x[None, :]
    ].copy()


def main() -> None:
    folder = Path(__file__).resolve().parent
    file_gambar = folder / "gambar.png"

    if not file_gambar.exists():
        raise FileNotFoundError(
            "Letakkan gambar.png di folder yang sama "
            "dengan 2c_zoom_minus.py."
        )

    citra_asli = np.asarray(
        img.imread(file_gambar, mode="RGB"),
        dtype=np.uint8
    )

    faktor = 0.5
    citra_hasil = zoom_minus(citra_asli, faktor)

    folder_hasil = folder / "hasil_2c"
    folder_hasil.mkdir(exist_ok=True)

    img.imwrite(
        folder_hasil / "hasil_zoom_minus.png",
        citra_hasil
    )

    tinggi, lebar = citra_asli.shape[:2]
    tinggi_baru, lebar_baru = citra_hasil.shape[:2]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].imshow(citra_asli)
    axes[0].set_title(
        f"Citra Asli — {lebar} x {tinggi} Piksel"
    )

    axes[1].imshow(citra_hasil)
    axes[1].set_title(
        f"Zoom Minus 0,5 Kali — "
        f"{lebar_baru} x {tinggi_baru} Piksel"
    )

    # Gunakan rentang koordinat yang sama agar pengecilan terlihat.
    for ax in axes:
        ax.set_xlim(-0.5, lebar - 0.5)
        ax.set_ylim(tinggi - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.tight_layout()

    fig.savefig(
        folder_hasil / "perbandingan_zoom_minus.png",
        dpi=150
    )

    print("Zoom minus berhasil.")
    print(f"Ukuran asli : {lebar} x {tinggi} piksel")
    print(f"Ukuran hasil: {lebar_baru} x {tinggi_baru} piksel")
    print("Hasil disimpan di:", folder_hasil)

    plt.show()


if __name__ == "__main__":
    main()