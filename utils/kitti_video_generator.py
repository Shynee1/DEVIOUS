"""Generate a video from a KITTI odometry sequence's image_2 frames using MoviePy.

Usage (from repository root):

	python -m utils.kitti_video_generator \
		--sequence /path/to/datasets/KITTI/odometry/sequences/00 \
		--output kitti_00.mp4 \
		--fps 10 \
		--resize 1280x384

Notes:
  * Expects the sequence directory to contain an `image_2` subfolder (standard KITTI odometry layout).
  * Requires moviepy:  pip install moviepy
  * If --resize is omitted, original frame size is preserved.
  * You can limit frames with --start / --end or --max-frames.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from tqdm import tqdm
from PIL import ImageOps

try:
	from moviepy.editor import ImageSequenceClip
except ImportError as e:  # pragma: no cover - informative error path
	raise SystemExit("moviepy is required. Install with: pip install moviepy") from e

try:
	from PIL import Image
except ImportError as e:  # pragma: no cover
	raise SystemExit("Pillow is required. Install with: pip install pillow") from e


def parse_resize(resize: Optional[str]) -> Optional[Tuple[int, int]]:
	if not resize:
		return None
	if "x" not in resize.lower():
		raise ValueError("--resize must be WxH, e.g. 1280x384")
	w_str, h_str = resize.lower().split("x", 1)
	return int(w_str), int(h_str)


def collect_frame_paths(sequence_dir: Path) -> List[Path]:
	image_dir = sequence_dir / "image_2"
	if not image_dir.is_dir():
		raise FileNotFoundError(f"Expected image directory at {image_dir}")
	frames = sorted(image_dir.glob("*.png"))
	if not frames:
		raise FileNotFoundError(f"No .png frames found in {image_dir}")
	return frames


def ensure_even(img: Image.Image, pad_color=(0, 0, 0)) -> Image.Image:
	"""Pad image (right/bottom) so width & height are even (required by libx264 with yuv420p)."""
	w, h = img.size
	new_w = w + (w % 2)
	new_h = h + (h % 2)
	if (new_w, new_h) == (w, h):
		return img
	return ImageOps.expand(img, border=(0, 0, new_w - w, new_h - h), fill=pad_color)


def frame_iterator(
	frame_paths: List[Path],
	resize: Optional[Tuple[int, int]] = None,
	start: int = 0,
	end: Optional[int] = None,
	max_frames: Optional[int] = None,
	force_even: bool = True,
) -> Iterable[Image.Image]:
	total = len(frame_paths)
	if end is None or end < 0:
		end = total  # exclusive
	end = min(end, total)
	selected = frame_paths[start:end]
	if max_frames is not None:
		selected = selected[:max_frames]

	for p in tqdm(selected, desc="Loading frames", unit="frame"):
		img = Image.open(p)
		if resize is not None:
			img = img.resize(resize, Image.BILINEAR)
		if force_even:
			img = ensure_even(img)
		yield img


def build_clip(
	frame_paths: List[Path],
	fps: int,
	resize: Optional[Tuple[int, int]] = None,
	start: int = 0,
	end: Optional[int] = None,
	max_frames: Optional[int] = None,
	force_even: bool = True,
):
	# We feed a list of numpy arrays to ImageSequenceClip to avoid keeping PIL Image objects open.
	import numpy as np

	frames_np = [
		np.array(img)
		for img in frame_iterator(
			frame_paths, resize=resize, start=start, end=end, max_frames=max_frames, force_even=force_even
		)
	]
	if not frames_np:
		raise ValueError("No frames selected for clip")
	return ImageSequenceClip(frames_np, fps=fps)


def main():
	parser = argparse.ArgumentParser(description="Concatenate KITTI frames into a video using MoviePy")
	parser.add_argument(
		"--sequence",
		required=True,
		type=Path,
		help="Path to KITTI odometry sequence directory (containing image_2).",
	)
	parser.add_argument(
		"--output",
		required=True,
		type=Path,
		help="Output video file path (.mp4 recommended)",
	)
	parser.add_argument("--fps", type=int, default=10, help="Frames per second for output video (default: 10)")
	parser.add_argument(
		"--resize",
		type=str,
		default=None,
		help="Resize frames to WxH (e.g. 1280x384). If omitted, original size is used.",
	)
	parser.add_argument("--start", type=int, default=0, help="Start frame index (inclusive, default 0)")
	parser.add_argument(
		"--end",
		type=int,
		default=-1,
		help="End frame index (exclusive). Use -1 or omit for all remaining frames.",
	)
	parser.add_argument(
		"--max-frames",
		type=int,
		default=None,
		help="Limit total number of frames (after start/end slicing).",
	)
	parser.add_argument(
		"--codec",
		type=str,
		default="libx264",
		help="FFmpeg video codec (default libx264)",
	)
	parser.add_argument(
		"--bitrate",
		type=str,
		default="4000k",
		help="Target bitrate passed to ffmpeg (default 4000k)",
	)
	parser.add_argument(
		"--overwrite", action="store_true", help="Overwrite existing output file without prompting"
	)
	parser.add_argument(
		"--no-force-even",
		action="store_true",
		help="Disable automatic padding to even dimensions (may cause encoder failure if dimensions are odd)",
	)

	args = parser.parse_args()

	if args.output.exists() and not args.overwrite:
		raise SystemExit(f"Output file {args.output} exists. Use --overwrite to replace.")

	resize = parse_resize(args.resize)

	frame_paths = collect_frame_paths(args.sequence)
	clip = build_clip(
		frame_paths,
		fps=args.fps,
		resize=resize,
		start=args.start,
		end=args.end,
		max_frames=args.max_frames,
		force_even=not args.no_force_even,
	)

	args.output.parent.mkdir(parents=True, exist_ok=True)
	print(f"Writing video to {args.output} ({len(frame_paths)} total frames, fps={args.fps})")
	clip.write_videofile(
		str(args.output),
		codec=args.codec,
		audio=False,
		bitrate=args.bitrate,
		ffmpeg_params=["-pix_fmt", "yuv420p"],
		verbose=False,
		logger=None,
	)
	print("Done.")


if __name__ == "__main__":  # pragma: no cover
	main()

