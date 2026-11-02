# pyright: reportUnknownMemberType=false

from pathlib import Path

import imageio.v3 as img
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def mirroring_gabungan(
    citra: NDArray[np.uint8]
) -> NDArray[np.uint8]:
    # Balik kiri dan kanan.
    horizontal = citra[:, ::-1]

    # Balik atas dan bawah pada hasil horizontal.
    gabungan = horizontal[::-1, :]

    return gabungan.copy()


def main() -> None:
    folder = Path(__file__).resolve().parent
    file_gambar = folder / "gambar.png"

    if not file_gambar.exists():
        raise FileNotFoundError(
            "Letakkan gambar.png di folder yang sama "
            "dengan 2a_mirroring_gabungan.py."
        )

    citra_asli = np.asarray(
        img.imread(file_gambar, mode="RGB"),
        dtype=np.uint8
    )

    citra_hasil = mirroring_gabungan(citra_asli)

    folder_hasil = folder / "hasil_2a"
    folder_hasil.mkdir(exist_ok=True)

    img.imwrite(
        folder_hasil / "hasil_mirroring_gabungan.png",
        citra_hasil
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].imshow(citra_asli)
    axes[0].set_title("Citra Asli")
    axes[0].axis("off")

    axes[1].imshow(citra_hasil)
    axes[1].set_title(
        "Mirroring Horizontal dan Vertikal Bersamaan"
    )
    axes[1].axis("off")

    fig.tight_layout()

    fig.savefig(
        folder_hasil / "perbandingan_mirroring_gabungan.png",
        dpi=150
    )

    print("Mirroring gabungan berhasil.")
    print("Gambar dibalik kiri-kanan sekaligus atas-bawah.")
    print("Ukuran gambar tetap.")
    print("Hasil disimpan di:", folder_hasil)

    plt.show()


if __name__ == "__main__":
    main()