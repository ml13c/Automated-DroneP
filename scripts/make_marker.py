"""Print-ready AprilTag (tag36h11) for the user's wrist marker.

    python scripts/make_marker.py                    # ID 0, 12 cm (default wrist tag)
    python scripts/make_marker.py --size-cm 6        # close-range tests

Print at 100 % / "actual size" (no fit-to-page), check the 5 cm ruler, cut along the dashed
line (keeps the white border the detector needs), glue flat on cardboard, strap to the wrist.
The ID must be listed in configs/tracking.yaml -> marker.apriltag.ids.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from dronevision.config import REPO_ROOT

CELLS = 8  # tag36h11: 6x6 data cells + 1-cell black border each side
PAGE_IN = (8.5, 11.0)  # US Letter; fits A4 too (content is centred and small)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--id", type=int, default=0)
    p.add_argument("--size-cm", type=float, default=12.0, help="black square edge length")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "markers")
    p.add_argument("--png", action="store_true", help="also write a 300-dpi PNG of the page")
    args = p.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tag = cv2.aruco.generateImageMarker(d, args.id, CELLS * 50)

    pw, ph = PAGE_IN
    side_in = args.size_cm / 2.54
    cell_in = side_in / CELLS
    x0, y0 = (pw - side_in) / 2, (ph - side_in) / 2 + 1.0

    fig = plt.figure(figsize=PAGE_IN)
    ax = fig.add_axes((x0 / pw, y0 / ph, side_in / pw, side_in / ph))
    ax.imshow(tag, cmap="gray", interpolation="nearest", vmin=0, vmax=255)
    ax.set_axis_off()

    page = fig.add_axes((0, 0, 1, 1), facecolor="none")
    page.set_xlim(0, pw)
    page.set_ylim(0, ph)
    page.set_axis_off()
    m = 2 * cell_in  # white quiet zone kept when cutting
    page.add_patch(
        plt.Rectangle(
            (x0 - m, y0 - m),
            side_in + 2 * m,
            side_in + 2 * m,
            fill=False,
            ls="--",
            lw=0.6,
            ec="0.5",
        )
    )
    ruler_y = y0 - m - 0.6
    five_cm = 5 / 2.54
    page.plot([pw / 2 - five_cm / 2, pw / 2 + five_cm / 2], [ruler_y, ruler_y], "k-", lw=1)
    for x in (pw / 2 - five_cm / 2, pw / 2 + five_cm / 2):
        page.plot([x, x], [ruler_y - 0.06, ruler_y + 0.06], "k-", lw=1)
    page.text(
        pw / 2, ruler_y - 0.2, "this line must measure 5.0 cm", ha="center", va="top", fontsize=8
    )
    page.text(
        pw / 2,
        y0 + side_in + m + 0.25,
        f"tag36h11  ID {args.id}  -  black square {args.size_cm:.1f} cm  -  print at 100 %",
        ha="center",
        fontsize=9,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"tag36h11_id{args.id}_{args.size_cm:g}cm.pdf"
    fig.savefig(out)
    if args.png:
        fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
