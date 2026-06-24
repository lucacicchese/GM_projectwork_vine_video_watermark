import argparse
import math
import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from evaluate import compute_metrics_sampled
from video_handler import video_to_frames, frames_to_video, to_h264
from vine.src.stega_encoder_decoder import CustomConvNeXt
from vine.src.vine_turbo import VINE_Turbo


IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def clean_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def bit_accuracy(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return sum(x == y for x, y in zip(a[:n], b[:n])) / n


def text_to_bits(text: str):
    data = text.encode("utf-8")
    return [int(b) for byte in data for b in format(byte, "08b")]


def bits_to_text(bits):
    bits = bits[: len(bits) - (len(bits) % 8)]
    if not bits:
        return ""
    out = bytearray()
    for i in range(0, len(bits), 8):
        out.append(int("".join(str(x) for x in bits[i : i + 8]), 2))
    return out.decode("utf-8", errors="replace")


def load_finetuned_encoder(checkpoint_dir: str, device: torch.device) -> VINE_Turbo:
    required = ["ConditionAdaptor.pth", "UNet2DConditionModel.pth", "vae.pth"]
    for name in required:
        path = os.path.join(checkpoint_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required encoder file: {path}")
    encoder = VINE_Turbo(ckpt_path=checkpoint_dir, device=str(device))
    encoder.to(device)
    encoder.eval()
    return encoder


def load_finetuned_decoder(checkpoint_dir: str, base_decoder: str, device: torch.device) -> CustomConvNeXt:
    dec_path = os.path.join(checkpoint_dir, "CustomConvNeXt.pth")
    decoder = CustomConvNeXt.from_pretrained(base_decoder)
    if os.path.exists(dec_path):
        state = torch.load(dec_path, map_location="cpu")
        decoder.load_state_dict(state, strict=True)
    else:
        print("[load_finetuned_decoder] No CustomConvNeXt.pth found — decoder was frozen, using pretrained base weights.")
    decoder.to(device)
    decoder.eval()
    return decoder


def _bits100_tensor(bits100, device: torch.device):
    if len(bits100) > 100:
        raise ValueError("chunk > 100 bits")
    padded = bits100 + [0] * (100 - len(bits100))
    return torch.tensor(padded, dtype=torch.float).unsqueeze(0).to(device)


def _encode_single_frame(input_path: str, output_path: str, bits100, encoder, device, t_val_256,
                         center_margin: float = 0.0):
    image = Image.open(input_path).convert("RGB")
    orig_w, orig_h = image.size

    if center_margin > 0.0:
        # Encode into the center (1 - 2*margin) portion so the watermark survives edge crops.
        mx = int(orig_w * center_margin)
        my = int(orig_h * center_margin)
        center_w = orig_w - 2 * mx
        center_h = orig_h - 2 * my
        region_pil = image.crop((mx, my, orig_w - mx, orig_h - my))
    else:
        mx, my = 0, 0
        center_w, center_h = orig_w, orig_h
        region_pil = image

    stretched = region_pil.resize((256, 256), Image.Resampling.BICUBIC)
    stretched_tensor = (2.0 * transforms.ToTensor()(stretched) - 1.0).unsqueeze(0).to(device)
    input_image = (2.0 * transforms.ToTensor()(image) - 1.0).unsqueeze(0).to(device)
    watermark = _bits100_tensor(bits100, device)

    with torch.no_grad():
        encoded_image_256 = encoder(stretched_tensor, watermark)
        residual_256 = encoded_image_256 - stretched_tensor
        residual_region = F.interpolate(residual_256, size=(center_h, center_w), mode="bicubic", align_corners=False)

        full_residual = torch.zeros(1, 3, orig_h, orig_w, device=device)
        full_residual[:, :, my:orig_h - my, mx:orig_w - mx] = residual_region

        encoded_image = input_image + full_residual
        encoded_image = torch.clamp(encoded_image * 0.5 + 0.5, 0.0, 1.0)

    transforms.ToPILImage()(encoded_image[0].cpu()).save(output_path)


def watermark_multi_chunked_finetuned(input_folder: str, output_folder: str, bits, encoder, device, t_val_256, center_margin: float = 0.0):
    os.makedirs(output_folder, exist_ok=True)
    frames = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(IMG_EXTS)])
    chunks = [bits[i : i + 100] for i in range(0, len(bits), 100)]
    n_chunks = len(chunks)
    if n_chunks == 0:
        raise ValueError("Empty watermark bits.")
    if len(frames) < n_chunks:
        raise ValueError(f"Not enough frames: need at least {n_chunks}, found {len(frames)}")

    frames_per_chunk = [0] * n_chunks
    encoded_frames = 0
    for fi, frame in enumerate(frames):
        chunk_idx = fi % n_chunks
        chunk_bits = chunks[chunk_idx]
        src = os.path.join(input_folder, frame)
        dst = os.path.join(output_folder, frame)
        _encode_single_frame(src, dst, chunk_bits, encoder, device, t_val_256, center_margin=center_margin)
        frames_per_chunk[chunk_idx] += 1
        encoded_frames += 1

    return {"chunks": n_chunks, "encoded_frames": encoded_frames, "frames_per_chunk": frames_per_chunk}


def decode_multi_chunked_finetuned(folder_path: str, total_bits: int, decoder, device, t_val_256):
    frames = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(IMG_EXTS)])
    n_chunks = int(math.ceil(total_bits / 100))
    if len(frames) < n_chunks:
        raise ValueError(f"Not enough frames to decode: need {n_chunks}, found {len(frames)}")

    chunk_preds = [[] for _ in range(n_chunks)]
    for fi, frame in enumerate(frames):
        image = Image.open(os.path.join(folder_path, frame)).convert("RGB")
        image = t_val_256(image).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = decoder(image)[0].detach().cpu().numpy()
        chunk_idx = fi % n_chunks
        chunk_preds[chunk_idx].append(pred)

    out_bits = []
    for chunk_idx in range(n_chunks):
        preds = np.stack(chunk_preds[chunk_idx], axis=0)
        pred = np.round(np.mean(preds, axis=0)).astype(int).tolist()
        take = min(100, total_bits - len(out_bits))
        out_bits.extend(pred[:take])
    return out_bits


def crop_frames_edges(input_folder: str, output_folder: str, pixels: int = 10):
    """Copy every frame from input_folder into output_folder with `pixels`
    removed from each of the four edges."""
    os.makedirs(output_folder, exist_ok=True)
    frames = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(IMG_EXTS)])
    for frame in frames:
        img = Image.open(os.path.join(input_folder, frame)).convert("RGB")
        w, h = img.size
        cropped = img.crop((pixels, pixels, w - pixels, h - pixels))
        cropped.save(os.path.join(output_folder, frame))
    return len(frames)


def run_pipeline(label: str, frames_wm: str, frames_orig: str, d_sub: dict,
                 bits: list, decoder, device, t_val_256,
                 args, step_offset: int = 0):
    """Run steps 4-10 (rebuild → compress → extract → decode → metrics)
    on a given set of watermarked frames. Returns a dict of results."""

    s = step_offset

    print(f"{s+4}) [{label}] Rebuild watermarked video (lossless)")
    wm_raw_video = os.path.join(d_sub["wm_raw"], "reconstructed_wm.mkv")
    frames_to_video(frames_wm, wm_raw_video, fps=args.target_fps)

    print(f"{s+5}) [{label}] Compress watermarked video (H.264)")
    wm_h264_video = to_h264(wm_raw_video, d_sub["wm_h264"], crf=args.crf, fps=args.target_fps)

    print(f"{s+6}) [{label}] Extract compressed watermarked frames")
    video_to_frames(wm_h264_video, d_sub["frames_wm_h264"], fps=args.target_fps)

    print(f"{s+7}) [{label}] Decode watermark")
    pred_bits = decode_multi_chunked_finetuned(
        d_sub["frames_wm_h264"], total_bits=len(bits),
        decoder=decoder, device=device, t_val_256=t_val_256,
    )
    acc = bit_accuracy(bits, pred_bits)
    decoded_text = bits_to_text(pred_bits)
    print(f"   bit_accuracy={acc:.4f}  decoded='{decoded_text}'")

    print(f"{s+8}) [{label}] Metrics: original vs compressed watermarked")
    psnr_w, ssim_w, lpips_w = compute_metrics_sampled(
        frames_orig, d_sub["frames_wm_h264"], every_n=args.sample_every_n
    )
    print(f"   PSNR={psnr_w:.2f}  SSIM={ssim_w:.4f}  LPIPS={lpips_w:.4f}")

    return {
        "bit_accuracy": acc,
        "decoded_text": decoded_text,
        "psnr": psnr_w,
        "ssim": ssim_w,
        "lpips": lpips_w,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_video", required=True)
    p.add_argument("--watermark_text", default="VINE2024")
    p.add_argument("--workdir", default="data/finetune_test")
    p.add_argument("--target_fps", type=float, default=25.0)
    p.add_argument("--crf", type=int, default=23)
    p.add_argument("--sample_every_n", type=int, default=30)
    p.add_argument("--checkpoint_dir", type=str, default="output/finetune_curriculum/checkpoint-2000")
    p.add_argument("--base_decoder", type=str, default="Shilin-LU/VINE-B-Dec")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--crop_analysis", action="store_true",
                   help="Also run the full pipeline on edge-cropped frames (+10px removed per edge) "
                        "to test robustness to reframing.")
    p.add_argument("--crop_pixels", type=int, default=10,
                   help="Pixels to remove from each edge in crop analysis (default: 10).")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This finetuned VINE flow requires CUDA.")

    device = torch.device(args.device)
    encoder = load_finetuned_encoder(args.checkpoint_dir, device)
    decoder = load_finetuned_decoder(args.checkpoint_dir, args.base_decoder, device)
    t_val_256 = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])

    wd = os.path.abspath(args.workdir)
    d = {
        "baseline_h264":    os.path.join(wd, "0_baseline_h264"),
        "frames_orig":      os.path.join(wd, "1_frames_orig"),
        "frames_wm":        os.path.join(wd, "2_frames_wm"),
        "wm_raw":           os.path.join(wd, "3_wm_raw"),
        "wm_h264":          os.path.join(wd, "4_wm_h264"),
        "frames_wm_h264":   os.path.join(wd, "5_frames_wm_h264"),
        "frames_base_h264": os.path.join(wd, "6_frames_base_h264"),
    }
    if args.crop_analysis:
        d.update({
            "frames_wm_cropped":      os.path.join(wd, "7_frames_wm_cropped"),
            "frames_orig_cropped":    os.path.join(wd, "8_frames_orig_cropped"),
            "wm_raw_cropped":         os.path.join(wd, "9_wm_raw_cropped"),
            "wm_h264_cropped":        os.path.join(wd, "10_wm_h264_cropped"),
            "frames_wm_h264_cropped": os.path.join(wd, "11_frames_wm_h264_cropped"),
        })

    os.makedirs(wd, exist_ok=True)
    for v in d.values():
        clean_dir(v)

    # ------------------------------------------------------------------ #
    #  Steps 1-3: common setup                                            #
    # ------------------------------------------------------------------ #
    print("1) Baseline H264")
    baseline_video = to_h264(args.input_video, d["baseline_h264"], crf=args.crf, fps=args.target_fps)

    print("2) Extract original frames")
    video_to_frames(args.input_video, d["frames_orig"], fps=args.target_fps)

    print("3) Watermark frames")
    bits = text_to_bits(args.watermark_text)
    info = watermark_multi_chunked_finetuned(d["frames_orig"], d["frames_wm"], bits, encoder, device, t_val_256)
    print(f"   bits={len(bits)} chunks={info['chunks']} encoded_frames={info['encoded_frames']}")

    # ------------------------------------------------------------------ #
    #  Steps 4-10: standard pipeline on full watermarked frames           #
    # ------------------------------------------------------------------ #
    d_full = {"wm_raw": d["wm_raw"], "wm_h264": d["wm_h264"], "frames_wm_h264": d["frames_wm_h264"]}
    res_full = run_pipeline(
        label="full frames",
        frames_wm=d["frames_wm"],
        frames_orig=d["frames_orig"],
        d_sub=d_full,
        bits=bits,
        decoder=decoder,
        device=device,
        t_val_256=t_val_256,
        args=args,
        step_offset=0,
    )

    print("   Metrics original vs baseline_compressed")
    video_to_frames(baseline_video, d["frames_base_h264"], fps=args.target_fps)
    psnr_b, ssim_b, lpips_b = compute_metrics_sampled(
        d["frames_orig"], d["frames_base_h264"], every_n=args.sample_every_n
    )
    print(f"   PSNR={psnr_b:.2f}  SSIM={ssim_b:.4f}  LPIPS={lpips_b:.4f}")

    # ------------------------------------------------------------------ #
    #  Steps A1-A7: crop analysis pipeline                                #
    # ------------------------------------------------------------------ #
    if args.crop_analysis:
        px = args.crop_pixels
        print(f"\n=== Crop analysis: removing {px}px from each edge ===")

        print(f"A1) Crop watermarked frames ({px}px per edge)")
        n = crop_frames_edges(d["frames_wm"], d["frames_wm_cropped"], pixels=px)
        print(f"   {n} frames cropped")

        print(f"A2) Crop original frames ({px}px per edge, for fair metric comparison)")
        crop_frames_edges(d["frames_orig"], d["frames_orig_cropped"], pixels=px)

        d_crop = {
            "wm_raw":         d["wm_raw_cropped"],
            "wm_h264":        d["wm_h264_cropped"],
            "frames_wm_h264": d["frames_wm_h264_cropped"],
        }
        res_crop = run_pipeline(
            label="cropped frames",
            frames_wm=d["frames_wm_cropped"],
            frames_orig=d["frames_orig_cropped"],
            d_sub=d_crop,
            bits=bits,
            decoder=decoder,
            device=device,
            t_val_256=t_val_256,
            args=args,
            step_offset=6,
        )

        print("\n=== Summary ===")
        print(f"  {'':30s}  {'Full':>8}  {'Cropped':>8}")
        print(f"  {'Bit accuracy':30s}  {res_full['bit_accuracy']:>8.4f}  {res_crop['bit_accuracy']:>8.4f}")
        print(f"  {'PSNR (dB)':30s}  {res_full['psnr']:>8.2f}  {res_crop['psnr']:>8.2f}")
        print(f"  {'SSIM':30s}  {res_full['ssim']:>8.4f}  {res_crop['ssim']:>8.4f}")
        print(f"  {'LPIPS':30s}  {res_full['lpips']:>8.4f}  {res_crop['lpips']:>8.4f}")
        print(f"  {'Decoded text (full)':30s}  {res_full['decoded_text']}")
        print(f"  {'Decoded text (cropped)':30s}  {res_crop['decoded_text']}")


if __name__ == "__main__":
    main()
