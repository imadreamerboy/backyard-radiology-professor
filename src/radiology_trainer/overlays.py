from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from radiology_trainer.domain import RegionBox
from radiology_trainer.preprocessing import prepare_image


def draw_regions(image: Image.Image, regions: list[RegionBox]) -> Image.Image:
    rendered = prepare_image(image).convert("RGB")
    draw = ImageDraw.Draw(rendered, "RGBA")
    width, height = rendered.size
    font = ImageFont.load_default()

    for region in regions:
        x1 = int(region.x1 * width)
        y1 = int(region.y1 * height)
        x2 = int(region.x2 * width)
        y2 = int(region.y2 * height)
        color = (55, 132, 255, 190)
        fill = (55, 132, 255, 32)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3, fill=fill)
        label = f"{region.label} {region.score:.2f}"
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        pad = 4
        draw.rectangle(
            (
                text_bbox[0] - pad,
                text_bbox[1] - pad,
                text_bbox[2] + pad,
                text_bbox[3] + pad,
            ),
            fill=(10, 14, 24, 220),
        )
        draw.text((x1, y1), label, fill=(255, 255, 255, 255), font=font)

    return rendered

