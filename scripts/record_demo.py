from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from playwright.sync_api import Locator, Page, sync_playwright


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    webm_path = args.output_dir / "backyard-radiology-professor-demo.webm"

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=not args.headed)
        context = browser.new_context(
            viewport={"width": 1440, "height": 900},
            record_video_dir=str(raw_dir),
            record_video_size={"width": 1440, "height": 900},
        )
        page = context.new_page()
        page.goto(args.app_url, wait_until="domcontentloaded", timeout=args.timeout_ms)
        if args.show_cursor:
            _install_cursor(page)
        page.locator("#case-list .case-item").first.wait_for(timeout=args.timeout_ms)

        tutorial_steps = page.locator(".walkthrough-step").count()
        for _ in range(tutorial_steps):
            _pause(page, args.step_pause_ms)
            _click(page, page.locator("#walkthrough-next"))

        _pause(page, args.step_pause_ms)
        _click(page, page.locator('[data-case="scoliosis"]'))
        page.locator("#primary-canvas").wait_for(state="visible", timeout=args.timeout_ms)
        _pause(page, args.step_pause_ms)

        _click(page, page.locator("#blind-read"))
        page.locator("#blind-read").fill(
            "PA chest radiograph. No focal airspace opacity. "
            "There may be mild thoracic spinal curvature."
        )
        _pause(page, args.step_pause_ms)
        _click(page, page.locator("#commit-read"))
        page.locator("#evidence-content").wait_for(state="visible", timeout=args.timeout_ms)
        _pause(page, args.step_pause_ms + 1200)

        _click(page, page.locator('[data-panel="evidence"]'))
        _pause(page, args.step_pause_ms + 1200)
        _click(page, page.locator('[data-panel="chat"]'))
        _pause(page, args.step_pause_ms)
        page.locator("#chat-input").fill("Explain the most important teaching point.")
        _click(page, page.locator("#chat-send"))
        page.wait_for_function(
            "() => document.querySelector('#professor-state')?.textContent === "
            "'Grounded in this session'",
            timeout=args.timeout_ms,
        )
        _pause(page, args.step_pause_ms + 1800)

        video = page.video
        context.close()
        if video is None:
            raise RuntimeError("Playwright did not create a video.")
        video.save_as(str(webm_path))
        browser.close()

    _convert_to_mp4(webm_path, args.output_dir)
    print(webm_path)


def _install_cursor(page: Page) -> None:
    page.add_style_tag(
        content="""
        .recording-cursor {
          position: fixed;
          z-index: 2147483647;
          width: 18px;
          height: 18px;
          border: 2px solid #55d4d1;
          border-radius: 999px;
          box-shadow: 0 0 0 3px rgba(85, 212, 209, .18);
          pointer-events: none;
          transform: translate(-50%, -50%);
          transition: width .12s ease, height .12s ease, background .12s ease;
        }
        .recording-cursor.down {
          width: 28px;
          height: 28px;
          background: rgba(85, 212, 209, .22);
        }
        """
    )
    page.evaluate(
        """
        () => {
          const cursor = document.createElement('div');
          cursor.className = 'recording-cursor';
          document.body.appendChild(cursor);
          window.addEventListener('mousemove', (event) => {
            cursor.style.left = `${event.clientX}px`;
            cursor.style.top = `${event.clientY}px`;
          }, { passive: true });
          window.addEventListener('mousedown', () => cursor.classList.add('down'), { passive: true });
          window.addEventListener('mouseup', () => cursor.classList.remove('down'), { passive: true });
        }
        """
    )


def _click(page: Page, locator: Locator) -> None:
    locator.wait_for(state="visible")
    box = locator.bounding_box()
    if box:
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=18)
        page.wait_for_timeout(350)
    locator.click()


def _pause(page: Page, milliseconds: int) -> None:
    page.wait_for_timeout(milliseconds)


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
    parser.add_argument("--step-pause-ms", type=int, default=1800)
    parser.add_argument("--show-cursor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--headed", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
