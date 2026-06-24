import os
import subprocess
from pathlib import Path


def video_to_frames(video_path: str, output_folder: str, fps: float = 25.0) -> None:
    os.makedirs(output_folder, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y",
        "-fflags", "+genpts",
        "-i", video_path,
        "-map", "0:v:0",
        "-vf", f"fps={fps}",
        "-start_number", "1",
        os.path.join(output_folder, "frame_%06d.png"),
    ], check=True)


def frames_to_video(frames_folder: str, output_file: str, fps: float = 25.0) -> None:
    if not list(Path(frames_folder).glob("*.png")):
        raise ValueError(f"No PNG frames found in: {frames_folder}")

    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-pattern_type", "glob",
        "-i", os.path.join(frames_folder, "*.png"),
        "-c:v", "ffv1",
        "-pix_fmt", "rgb24",
        output_file,
    ], check=True)


def to_h264(video_path: str, output_folder: str, crf: int = 23, fps: float = 25.0) -> str:
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, Path(video_path).stem + ".mp4")
    subprocess.run([
        "ffmpeg", "-y",
        "-fflags", "+genpts",
        "-i", video_path,
        "-map", "0:v:0",
        "-r", str(fps),
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "slow",
        "-pix_fmt", "yuv420p",
        output_file,
    ], check=True)
    return output_file
