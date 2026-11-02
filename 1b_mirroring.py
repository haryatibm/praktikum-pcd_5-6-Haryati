# pyright: reportUnknownMemberType=false

from pathlib import Path

import imageio.v3 as img
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def mirroring_horizontal(
    citra: NDArray[np.uint8]
) -> NDArray[np.uint8]:
    # Membalik posisi kiri dan kanan.
    return citra[:, ::-1].copy()


def mirroring_vertikal(
    citra: NDArray[np.uint8]
) -> NDArray[np.uint8]:
    # Membalik posisi atas dan bawah.
    return citra[::-1, :].copy()


def main() -> None:
    folder = Path(__file__).resolve().parent
    file_gambar = folder / "gambar.png"

    if not file_gambar.exists():
        raise FileNotFoundError(
            "Letakkan gambar.png di folder yang sama "
            "dengan 1b_mirroring.py."
        )

    citra_asli = np.asarray(
        img.imread(file_gambar, mode="RGB"),
        dtype=np.uint8
    )

    horizontal = mirroring_horizontal(citra_asli)
    vertikal = mirroring_vertikal(citra_asli)

    folder_hasil = folder / "hasil_1b"
    folder_hasil.mkdir(exist_ok=True)

    img.imwrite(
        folder_hasil / "mirroring_horizontal.png",
        horizontal
    )

    img.imwrite(
        folder_hasil / "mirroring_vertikal.png",
        vertikal
    )

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    axes[0].imshow(citra_asli)
    axes[0].set_title("Citra Asli")
    axes[0].axis("off")

    axes[1].imshow(horizontal)
    axes[1].set_title("Mirroring Horizontal — Kiri dan Kanan")
    axes[1].axis("off")

    axes[2].imshow(vertikal)
    axes[2].set_title("Mirroring Vertikal — Atas dan Bawah")
    axes[2].axis("off")

    fig.tight_layout()

    fig.savefig(
        folder_hasil / "perbandingan_mirroring.png",
        dpi=150
    )

    print("Mirroring berhasil.")
    print("Hasil disimpan di:", folder_hasil)

    plt.show()


if __name__ == "__main__":
    main()