from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "assets" / "examples"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _draw_case("normal_search_pattern.png", finding=None)
    _draw_case("right_lower_opacity.png", finding="right_lower_opacity")
    _draw_case("effusion_pattern.png", finding="effusion")
    _draw_case("pneumothorax_check.png", finding="pneumothorax")


def _draw_case(filename: str, finding: str | None) -> None:
    width = height = 900
    image = Image.new("L", (width, height), 18)
    draw = ImageDraw.Draw(image, "L")

    for i in range(80):
        shade = 42 + int(i * 0.7)
        draw.rounded_rectangle(
            (100 + i, 70 + i // 2, 800 - i, 850 - i // 3),
            radius=90,
            outline=shade,
            width=2,
        )

    _draw_lungs(draw)
    _draw_mediastinum(draw)
    _draw_ribs(draw)
    _draw_diaphragm(draw)

    if finding == "right_lower_opacity":
        _draw_opacity(draw, center=(610, 535), radius=78)
    elif finding == "effusion":
        _draw_effusion(draw)
    elif finding == "pneumothorax":
        _draw_pneumothorax(draw)

    image = image.filter(ImageFilter.GaussianBlur(1.2))
    image.save(OUT_DIR / filename)


def _draw_lungs(draw: ImageDraw.ImageDraw) -> None:
    for box in [(155, 170, 430, 745), (470, 170, 745, 745)]:
        draw.ellipse(box, fill=88)


def _draw_mediastinum(draw: ImageDraw.ImageDraw) -> None:
    for box, shade in [((350, 355, 575, 760), 118), ((395, 130, 505, 610), 102)]:
        draw.ellipse(box, fill=shade)


def _draw_ribs(draw: ImageDraw.ImageDraw) -> None:
    for y in range(220, 650, 55):
        draw.arc((85, y - 120, 455, y + 120), 195, 345, fill=145, width=4)
        draw.arc((445, y - 120, 815, y + 120), 195, 345, fill=145, width=4)


def _draw_diaphragm(draw: ImageDraw.ImageDraw) -> None:
    for y in range(690, 760, 4):
        draw.arc((120, y - 190, 440, y + 90), 10, 175, fill=130, width=3)
        draw.arc((460, y - 190, 780, y + 90), 5, 170, fill=130, width=3)


def _draw_opacity(draw: ImageDraw.ImageDraw, center: tuple[int, int], radius: int) -> None:
    cx, cy = center
    for r in range(radius, 8, -8):
        shade = 132 + int((radius - r) * 0.5)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=shade, width=4)


def _draw_effusion(draw: ImageDraw.ImageDraw) -> None:
    for offset in range(0, 95, 5):
        draw.arc((100 + offset, 620 - offset // 2, 430 - offset // 3, 860), 0, 180, fill=162, width=4)
        draw.arc((470 + offset // 3, 620 - offset // 2, 795 - offset, 860), 0, 180, fill=150, width=3)


def _draw_pneumothorax(draw: ImageDraw.ImageDraw) -> None:
    draw.arc((555, 135, 785, 610), 105, 260, fill=178, width=5)
    draw.polygon([(650, 155), (742, 160), (755, 515), (690, 550)], fill=42)


if __name__ == "__main__":
    main()

