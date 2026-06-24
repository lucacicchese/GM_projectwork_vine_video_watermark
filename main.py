import argparse
import os
import shutil

from evaluate import compute_metrics_sampled
from video_handler import video_to_frames, frames_to_video, to_h264
from watermark_vine_new import text_to_bits, bits_to_text, watermark_multi_chunked, decode_multi_chunked


def clean_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def bit_accuracy(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return sum(x == y for x, y in zip(a[:n], b[:n])) / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_video", required=True)
    p.add_argument("--watermark_text", default="VINE2024")
    p.add_argument("--workdir", default="data/video_demo")
    p.add_argument("--target_fps", type=float, default=25.0)
    p.add_argument("--crf", type=int, default=23)
    p.add_argument("--sample_every_n", type=int, default=30)
    args = p.parse_args()

    wd = os.path.abspath(args.workdir)
    d = {
        "baseline_h264": os.path.join(wd, "0_baseline_h264"),
        "frames_orig": os.path.join(wd, "1_frames_orig"),
        "frames_wm": os.path.join(wd, "2_frames_wm"),
        "wm_raw": os.path.join(wd, "3_wm_raw"),
        "wm_h264": os.path.join(wd, "4_wm_h264"),
        "frames_wm_h264": os.path.join(wd, "5_frames_wm_h264"),
        "frames_base_h264": os.path.join(wd, "6_frames_base_h264"),
    }
    os.makedirs(wd, exist_ok=True)
    for v in d.values():
        clean_dir(v)

    print("1) Baseline H264")
    baseline_video = to_h264(args.input_video, d["baseline_h264"], crf=args.crf, fps=args.target_fps)

    print("2) Extract original frames")
    video_to_frames(args.input_video, d["frames_orig"], fps=args.target_fps)

    print("3) Watermark frames")
    bits = text_to_bits(args.watermark_text)
    info = watermark_multi_chunked(d["frames_orig"], d["frames_wm"], bits)
    print(f"   bits={len(bits)} chunks={info['chunks']} encoded_frames={info['encoded_frames']}")

    print("4) Rebuild watermarked video")
    wm_raw_video = os.path.join(d["wm_raw"], "reconstructed_wm.mkv")
    frames_to_video(d["frames_wm"], wm_raw_video, fps=args.target_fps)

    print("5) Compress watermarked video")
    wm_h264_video = to_h264(wm_raw_video, d["wm_h264"], crf=args.crf, fps=args.target_fps)

    print("6) Extract compressed watermarked frames")
    video_to_frames(wm_h264_video, d["frames_wm_h264"], fps=args.target_fps)

    print("7) Extract baseline compressed frames")
    video_to_frames(baseline_video, d["frames_base_h264"], fps=args.target_fps)

    print("8) Decode watermark")
    pred_bits = decode_multi_chunked(d["frames_wm_h264"], total_bits=len(bits))
    acc = bit_accuracy(bits, pred_bits)
    print(f"   bit_accuracy={acc:.4f}")
    print(f"   original_text={args.watermark_text}")
    print(f"   decoded_text={bits_to_text(pred_bits)}")

    print("9) Metrics original vs compressed_watermarked")
    psnr_w, ssim_w, lpips_w = compute_metrics_sampled(
        d["frames_orig"], d["frames_wm_h264"], every_n=args.sample_every_n
    )
    print(f"PSNR={psnr_w:.2f} SSIM={ssim_w:.4f} LPIPS={lpips_w:.4f}")

    print("10) Metrics original vs baseline_compressed")
    psnr_b, ssim_b, lpips_b = compute_metrics_sampled(
        d["frames_orig"], d["frames_base_h264"], every_n=args.sample_every_n
    )
    print(f"PSNR={psnr_b:.2f} SSIM={ssim_b:.4f} LPIPS={lpips_b:.4f}")


if __name__ == "__main__":
    main()

