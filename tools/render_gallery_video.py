#!/usr/bin/env python3
"""Render the genus 0-5 OBJ frame sequences into a report-friendly MP4."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="out")
    parser.add_argument(
        "--output",
        default="websites/final_assets/gallery_descent.mp4",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def load_obj(path: Path, faces: bool = True) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[tuple[float, float, float]] = []
    tri_faces: list[tuple[int, int, int]] = []
    with path.open() as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z, *rest = line.split()
                vertices.append((float(x), float(y), float(z)))
            elif faces and line.startswith("f "):
                idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
                if len(idx) == 3:
                    tri_faces.append((idx[0], idx[1], idx[2]))
    return np.asarray(vertices, dtype=np.float64), np.asarray(tri_faces, dtype=np.int32)


def rotation_matrix(elev_degrees: float, azim_degrees: float) -> np.ndarray:
    elev = math.radians(elev_degrees)
    azim = math.radians(azim_degrees)
    rz = np.array(
        [
            [math.cos(azim), -math.sin(azim), 0.0],
            [math.sin(azim), math.cos(azim), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(elev), -math.sin(elev)],
            [0.0, math.sin(elev), math.cos(elev)],
        ],
        dtype=np.float64,
    )
    return rx @ rz


def frame_paths(input_root: Path, genus: int) -> list[Path]:
    return sorted((input_root / f"gallery_genus{genus}").glob("frame_*.obj"))


def compute_centers_and_scales(paths_by_genus: list[list[Path]]) -> tuple[list[np.ndarray], list[float]]:
    centers: list[np.ndarray] = []
    scales: list[float] = []
    for paths in paths_by_genus:
        mins = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
        maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
        for path in paths:
            vertices, _ = load_obj(path, faces=False)
            mins = np.minimum(mins, vertices.min(axis=0))
            maxs = np.maximum(maxs, vertices.max(axis=0))
        center = 0.5 * (mins + maxs)
        scale = float(np.max(maxs - mins))
        centers.append(center)
        scales.append(scale if scale > 0.0 else 1.0)
    return centers, scales


def font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_mesh(
    draw: ImageDraw.ImageDraw,
    vertices: np.ndarray,
    faces: np.ndarray,
    rect: tuple[int, int, int, int],
    center: np.ndarray,
    scale: float,
    rotation: np.ndarray,
) -> None:
    x0, y0, x1, y1 = rect
    width = x1 - x0
    height = y1 - y0
    cell_scale = 0.72 * min(width, height)

    v = (vertices - center) / scale
    p = v @ rotation.T
    xy = p[:, [0, 2]]
    depth = p[:, 1]

    screen = np.empty_like(xy)
    screen[:, 0] = x0 + 0.5 * width + cell_scale * xy[:, 0]
    screen[:, 1] = y0 + 0.54 * height - cell_scale * xy[:, 1]

    order = np.argsort(depth[faces].mean(axis=1))
    shade_raw = depth[faces].mean(axis=1)
    shade_min = float(shade_raw.min())
    shade_ptp = float(np.ptp(shade_raw)) + 1e-12

    for face_idx in order:
        face = faces[face_idx]
        pts = [(float(screen[i, 0]), float(screen[i, 1])) for i in face]
        shade = (float(shade_raw[face_idx]) - shade_min) / shade_ptp
        base = np.array([154, 177, 207], dtype=np.float64)
        light = np.array([235, 241, 248], dtype=np.float64)
        rgb = (0.75 - 0.15 * shade) * base + (0.25 + 0.15 * shade) * light
        fill = tuple(int(np.clip(c, 0, 255)) for c in rgb)
        draw.polygon(pts, fill=fill, outline=(55, 65, 78))


def render_frame(
    frame_index: int,
    paths_by_genus: list[list[Path]],
    centers: list[np.ndarray],
    scales: list[float],
    output_path: Path,
    width: int,
    height: int,
    title_font: ImageFont.ImageFont,
    label_font: ImageFont.ImageFont,
) -> None:
    bg = (240, 242, 245)
    surface = (230, 233, 238)
    border = (205, 210, 217)
    text = (30, 42, 56)
    dim = (74, 85, 104)

    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)

    header_h = 64
    draw.text((32, 22), "Repulsive Surfaces canonical embedding descent", fill=text, font=title_font)
    draw.text((width - 160, 27), f"frame {frame_index:04d}", fill=dim, font=label_font)

    cols = 3
    rows = 2
    gap = 16
    left = 24
    top = header_h
    cell_w = (width - 2 * left - (cols - 1) * gap) // cols
    cell_h = (height - top - 24 - (rows - 1) * gap) // rows

    rotations = [
        rotation_matrix(24, -42),
        rotation_matrix(24, -58),
        rotation_matrix(24, -42),
        rotation_matrix(24, -42),
        rotation_matrix(24, -42),
        rotation_matrix(24, -42),
    ]

    for genus in range(6):
        row = genus // cols
        col = genus % cols
        x0 = left + col * (cell_w + gap)
        y0 = top + row * (cell_h + gap)
        x1 = x0 + cell_w
        y1 = y0 + cell_h
        draw.rounded_rectangle((x0, y0, x1, y1), radius=8, fill=surface, outline=border)

        paths = paths_by_genus[genus]
        path = paths[min(frame_index, len(paths) - 1)]
        vertices, faces = load_obj(path, faces=True)
        draw_mesh(
            draw,
            vertices,
            faces,
            (x0 + 10, y0 + 24, x1 - 10, y1 - 8),
            centers[genus],
            scales[genus],
            rotations[genus],
        )
        draw.text((x0 + 12, y0 + 8), f"genus {genus}", fill=text, font=label_font)

    image.save(output_path, quality=95)


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    paths_by_genus = [frame_paths(input_root, genus) for genus in range(6)]
    if any(len(paths) == 0 for paths in paths_by_genus):
        raise RuntimeError("all gallery_genusN directories must contain frame_*.obj")

    n_frames = min(len(paths) for paths in paths_by_genus)
    sampled_indices = list(range(0, n_frames, args.stride))
    if sampled_indices[-1] != n_frames - 1:
        sampled_indices.append(n_frames - 1)

    centers, scales = compute_centers_and_scales(paths_by_genus)
    title_font = font(23)
    label_font = font(15)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found")

    with tempfile.TemporaryDirectory(prefix="rsh_gallery_video_") as tmp:
        frame_dir = Path(tmp)
        for out_idx, frame_idx in enumerate(sampled_indices):
            render_frame(
                frame_idx,
                paths_by_genus,
                centers,
                scales,
                frame_dir / f"frame_{out_idx:04d}.png",
                args.width,
                args.height,
                title_font,
                label_font,
            )
            if out_idx % 25 == 0:
                print(f"rendered {out_idx + 1}/{len(sampled_indices)} frames")

        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            str(args.fps),
            "-i",
            str(frame_dir / "frame_%04d.png"),
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "22",
            "-movflags",
            "+faststart",
            str(output),
        ]
        subprocess.run(cmd, check=True)
    print(output)


if __name__ == "__main__":
    main()
