#!/usr/bin/env python3
"""Render the ball-through-tube OBJ sequence and convergence plot."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="out/ball_tube_interp")
    parser.add_argument(
        "--video-output",
        default="websites/final_assets/ball_tube_interp.mp4",
    )
    parser.add_argument(
        "--plot-output",
        default="websites/final_assets/ball_tube_energy.svg",
    )
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    with path.open() as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z, *rest = line.split()
                vertices.append((float(x), float(y), float(z)))
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
                if len(idx) == 3:
                    faces.append((idx[0], idx[1], idx[2]))
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


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


def projection(vertices: np.ndarray, center: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    v = (vertices - center) / scale
    # Tube axis is x. View mostly along -y, but with enough radial rotation
    # that the hollow rings remain visible instead of collapsing to lines.
    angle = math.radians(-35.0)
    ca = math.cos(angle)
    sa = math.sin(angle)
    x = v[:, 0]
    y = ca * v[:, 2] - sa * v[:, 1]
    depth = sa * v[:, 2] + ca * v[:, 1]
    return np.column_stack([x, y]), depth


def to_screen(xy: np.ndarray, width: int, height: int) -> np.ndarray:
    margin_x = 96
    margin_y = 96
    sx = margin_x + (xy[:, 0] + 0.58) * (width - 2 * margin_x) / 1.16
    sy = height * 0.54 - xy[:, 1] * (height - 2 * margin_y) / 0.72
    return np.column_stack([sx, sy])


def draw_mesh(
    draw: ImageDraw.ImageDraw,
    vertices: np.ndarray,
    faces: np.ndarray,
    center: np.ndarray,
    scale: float,
    width: int,
    height: int,
) -> None:
    xy, depth = projection(vertices, center, scale)
    screen = to_screen(xy, width, height)
    order = np.argsort(depth[faces].mean(axis=1))
    shade_raw = depth[faces].mean(axis=1)
    shade_min = float(shade_raw.min())
    shade_ptp = float(np.ptp(shade_raw)) + 1e-12
    base = np.array([93, 133, 185], dtype=np.float64)
    light = np.array([228, 237, 247], dtype=np.float64)
    edge = (48, 62, 78)

    for face_idx in order:
        face = faces[face_idx]
        shade = (float(shade_raw[face_idx]) - shade_min) / shade_ptp
        rgb = (0.70 - 0.18 * shade) * base + (0.30 + 0.18 * shade) * light
        pts = [(float(screen[i, 0]), float(screen[i, 1])) for i in face]
        draw.polygon(pts, fill=tuple(int(c) for c in rgb), outline=edge)


def mesh_edges(faces: np.ndarray) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for a, b, c in faces:
        for i, j in ((a, b), (b, c), (c, a)):
            if i > j:
                i, j = j, i
            edges.add((int(i), int(j)))
    return sorted(edges)


def draw_wire(
    draw: ImageDraw.ImageDraw,
    vertices: np.ndarray,
    faces: np.ndarray,
    center: np.ndarray,
    scale: float,
    width: int,
    height: int,
) -> None:
    xy, _ = projection(vertices, center, scale)
    screen = to_screen(xy, width, height)
    color = (92, 101, 114)
    n_circ = vertices.shape[0] // 4
    if vertices.shape[0] == 4 * n_circ and n_circ >= 8:
        def idx(side: int, radius_id: int, i: int) -> int:
            return (side * 2 + radius_id) * n_circ + (i % n_circ)

        for side in range(2):
            for radius_id in range(2):
                pts = [
                    (float(screen[idx(side, radius_id, i), 0]),
                     float(screen[idx(side, radius_id, i), 1]))
                    for i in range(n_circ + 1)
                ]
                draw.line(pts, fill=color, width=3 if radius_id == 0 else 2)
        for radius_id in range(2):
            for i in range(0, n_circ, max(1, n_circ // 8)):
                p = (float(screen[idx(0, radius_id, i), 0]),
                     float(screen[idx(0, radius_id, i), 1]))
                q = (float(screen[idx(1, radius_id, i), 0]),
                     float(screen[idx(1, radius_id, i), 1]))
                draw.line([p, q], fill=color, width=2)
        return

    for i, j in mesh_edges(faces):
        p = (float(screen[i, 0]), float(screen[i, 1]))
        q = (float(screen[j, 0]), float(screen[j, 1]))
        draw.line([p, q], fill=color, width=2)


def render_video(input_dir: Path, output: Path, fps: int, width: int, height: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = sorted(input_dir.glob("frame_*.obj"))
    if not paths:
        raise RuntimeError(f"no frame_*.obj files found in {input_dir}")
    tube_path = input_dir / "obstacle.obj"
    if not tube_path.exists():
        raise RuntimeError(f"missing obstacle mesh: {tube_path}")

    tube_v, tube_f = load_obj(tube_path)
    ball_frames = [load_obj(path) for path in paths]
    mins = tube_v.min(axis=0)
    maxs = tube_v.max(axis=0)
    for vertices, _ in ball_frames:
        mins = np.minimum(mins, vertices.min(axis=0))
        maxs = np.maximum(maxs, vertices.max(axis=0))
    center = 0.5 * (mins + maxs)
    scale = 1.15 * float(np.max(maxs - mins))

    title_font = font(28)
    label_font = font(17)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found")

    with tempfile.TemporaryDirectory(prefix="rsh_ball_tube_") as tmp:
        frame_dir = Path(tmp)
        for idx, (vertices, faces) in enumerate(ball_frames):
            image = Image.new("RGB", (width, height), (240, 242, 245))
            draw = ImageDraw.Draw(image)
            draw.text((36, 28), "Ball-through-tube geodesic interpolation", fill=(30, 42, 56), font=title_font)
            draw.text((width - 160, 34), f"frame {idx:02d}/{len(paths)-1:02d}", fill=(74, 85, 104), font=label_font)
            draw.rounded_rectangle((44, 88, width - 44, height - 42), radius=8, fill=(230, 233, 238), outline=(205, 210, 217))
            draw_mesh(draw, vertices, faces, center, scale, width, height)
            draw_wire(draw, tube_v, tube_f, center, scale, width, height)
            image.save(frame_dir / f"frame_{idx:04d}.png", quality=95)

        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-framerate",
                str(fps),
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
            ],
            check=True,
        )


def render_plot(input_dir: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = input_dir / "energy.csv"
    rows: list[dict[str, str]]
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    it = np.asarray([float(r["iter"]) for r in rows], dtype=np.float64)
    total = np.asarray([float(r["total"]) for r in rows], dtype=np.float64)
    graph = np.asarray([float(r["repulsive_graph"]) for r in rows], dtype=np.float64)
    min_phi = np.asarray([float(r["min_phi_min"]) for r in rows], dtype=np.float64)

    plt.figure(figsize=(7.2, 4.0))
    ax1 = plt.gca()
    ax1.plot(it, total / total[0], label="total / total0", color="#405f8f", linewidth=2)
    ax1.plot(it, graph / graph[0], label="graph / graph0", color="#7b6d36", linewidth=2)
    ax1.set_xlabel("trust-region iteration")
    ax1.set_ylabel("normalized energy")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(it, min_phi, label="minimum SDF clearance", color="#7b3e48", linewidth=2)
    ax2.set_ylabel("minimum tube clearance")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right", frameon=False)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    render_video(input_dir, Path(args.video_output), args.fps, args.width, args.height)
    render_plot(input_dir, Path(args.plot_output))
    print(args.video_output)
    print(args.plot_output)


if __name__ == "__main__":
    main()
