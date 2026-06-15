from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=not args.headed)
        context = browser.new_context(
            viewport={"width": 1440, "height": 900},
            record_video_dir=str(raw_dir),
            record_video_size={"width": 1440, "height": 900},
        )
        page = context.new_page()
        page.goto(args.app_url, wait_until="domcontentloaded", timeout=args.timeout_ms)
        page.locator("#case-list .case-item").first.wait_for(timeout=args.timeout_ms)

        tutorial_steps = page.locator(".walkthrough-step").count()
        for _ in range(tutorial_steps):
            page.wait_for_timeout(1100)
            page.locator("#walkthrough-next").click()

        page.wait_for_timeout(800)
        page.locator('[data-case="scoliosis"]').click()
        page.locator("#primary-canvas").wait_for(state="visible", timeout=args.timeout_ms)
        page.wait_for_timeout(1800)

        page.locator("#blind-read").fill(
            "PA chest radiograph. No focal airspace opacity. "
            "There may be mild thoracic spinal curvature."
        )
        page.wait_for_timeout(1000)
        page.locator("#commit-read").click()
        page.locator("#evidence-content").wait_for(state="visible", timeout=args.timeout_ms)
        page.wait_for_timeout(2200)

        page.locator('[data-panel="evidence"]').click()
        page.wait_for_timeout(2500)
        page.locator('[data-panel="professor"]').click()
        page.wait_for_timeout(1800)
        page.locator("#chat-input").fill("Explain the most important teaching point.")
        page.locator("#chat-send").click()
        page.wait_for_function(
            "() => document.querySelector('#professor-state')?.textContent === "
            "'Grounded in this session'",
            timeout=args.timeout_ms,
        )
        page.wait_for_timeout(3000)

        video = page.video
        context.close()
        browser.close()

    if video is None:
        raise RuntimeError("Playwright did not create a video.")
    webm_path = args.output_dir / "backyard-radiology-professor-demo.webm"
    video.save_as(str(webm_path))
    _convert_to_mp4(webm_path, args.output_dir)
    print(webm_path)


def _convert_to_mp4(webm_path: Path, output_dir: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return
    mp4_path = output_dir / "backyard-radiology-professor-demo.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(webm_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(mp4_path),
        ],
        check=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record the complete public demo workflow.")
    parser.add_argument(
        "--app-url",
        default="https://build-small-hackathon-backyard-radiology-professor.hf.space",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/video"))
    parser.add_argument("--timeout-ms", type=int, default=15 * 60 * 1000)
    parser.add_argument("--headed", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
