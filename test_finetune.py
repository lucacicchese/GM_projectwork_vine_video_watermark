import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from vine.src.stega_encoder_decoder import CustomConvNeXt
from vine.src.vine_turbo import VINE_Turbo


def text_to_bits(message: str) -> torch.Tensor:
    if len(message) > 12:
        raise ValueError("Message must be <= 12 characters (100 bits total).")
    data = bytearray(message + " " * (12 - len(message)), "utf-8")
    packet_binary = "".join(format(x, "08b") for x in data)
    watermark = [int(x) for x in packet_binary]
    watermark.extend([0, 0, 0, 0])
    return torch.tensor(watermark, dtype=torch.float).unsqueeze(0)


def bits_to_text(bits: np.ndarray) -> str:
    bits = bits[:96]  # first 12 chars
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i : i + 8]
        chars.append(chr(int("".join(str(int(b)) for b in byte), 2)))
    return "".join(chars).rstrip()


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


def load_pretrained_encoder(encoder_repo: str, device: torch.device) -> VINE_Turbo:
    encoder = VINE_Turbo.from_pretrained(encoder_repo, device=str(device))
    encoder.to(device)
    encoder.eval()
    return encoder


def load_pretrained_decoder(decoder_repo: str, device: torch.device) -> CustomConvNeXt:
    decoder = CustomConvNeXt.from_pretrained(decoder_repo)
    decoder.to(device)
    decoder.eval()
    return decoder


def save_visual_diff(reference_image_path: str, target_image_path: str, output_image_path: str, scale: int = 16):
    with Image.open(reference_image_path).convert("RGB") as ref_img, \
         Image.open(target_image_path).convert("RGB") as tgt_img:
        rw, rh = ref_img.size
        tw, th = tgt_img.size
        cw, ch = min(rw, tw), min(rh, th)
        if cw <= 0 or ch <= 0:
            raise ValueError(f"Invalid crop area from sizes ref={ref_img.size}, tgt={tgt_img.size}")
        ref_crop = ref_img.crop(((rw - cw) // 2, (rh - ch) // 2, (rw + cw) // 2, (rh + ch) // 2))
        tgt_crop = tgt_img.crop(((tw - cw) // 2, (th - ch) // 2, (tw + cw) // 2, (th + ch) // 2))
        diff = np.abs(np.array(ref_crop).astype(np.int16) - np.array(tgt_crop).astype(np.int16))
        diff_amp = np.clip(diff * scale, 0, 255).astype(np.uint8)
        Image.fromarray(diff_amp, mode="RGB").save(output_image_path)


def encode_image(input_image_pil: Image.Image, watermark: torch.Tensor, encoder, device: torch.device, stretch_to_256):
    orig_w, orig_h = input_image_pil.size
    stretched_256 = stretch_to_256(input_image_pil).unsqueeze(0).to(device)
    stretched_256 = 2.0 * stretched_256 - 1.0

    input_tensor = transforms.ToTensor()(input_image_pil).unsqueeze(0).to(device)
    input_tensor = 2.0 * input_tensor - 1.0

    with torch.no_grad():
        encoded_256 = encoder(stretched_256, watermark)
        residual_256 = encoded_256 - stretched_256
        residual_original = F.interpolate(
            residual_256, size=(orig_h, orig_w), mode="bicubic", align_corners=False
        )
        encoded_full = (residual_original + input_tensor) * 0.5 + 0.5
        encoded_full = torch.clamp(encoded_full, 0.0, 1.0)
    return encoded_full


def decode_bits(image_tensor: torch.Tensor, decoder, device: torch.device, stretch_to_256):
    with torch.no_grad():
        pil = transforms.ToPILImage()(image_tensor[0].cpu())
        pred = decoder(stretch_to_256(pil).unsqueeze(0).to(device))
        pred_bits = torch.round(pred[0]).detach().cpu().numpy().astype(int)
    return pred_bits


def crop_edges(image_tensor: torch.Tensor, pixels: int = 10) -> torch.Tensor:
    """Remove `pixels` from each edge of a [1, C, H, W] tensor."""
    _, _, H, W = image_tensor.shape
    return image_tensor[:, :, pixels:H - pixels, pixels:W - pixels]


def bit_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    return float((pred == gt).sum() / len(gt))


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This project's VINE_Turbo initialization uses CUDA-only layers; run on a CUDA machine."
        )

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(args.input_path))[0]

    encoder_finetuned = load_finetuned_encoder(args.checkpoint_dir, device)
    decoder_finetuned = load_finetuned_decoder(args.checkpoint_dir, args.base_decoder, device)

    if not args.no_vine_r:
        encoder_vine_r = load_pretrained_encoder(args.vine_r_encoder, device)
        decoder_vine_r = load_pretrained_decoder(args.vine_r_decoder, device)

    input_image_pil = Image.open(args.input_path).convert("RGB")

    stretch_to_256 = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])

    gt_watermark = text_to_bits(args.message).to(device)
    gt_bits = gt_watermark[0].detach().cpu().numpy().astype(int)

    # --- Finetuned model ---
    encoded_finetuned = encode_image(input_image_pil, gt_watermark, encoder_finetuned, device, stretch_to_256)

    wm_finetuned_path = os.path.join(args.output_dir, f"{stem}_wm_finetuned.png")
    transforms.ToPILImage()(encoded_finetuned[0].cpu()).save(wm_finetuned_path)

    # Diff: original vs finetuned watermarked
    diff_path = os.path.join(args.output_dir, f"{stem}_diff_original_vs_finetuned.png")
    save_visual_diff(args.input_path, wm_finetuned_path, diff_path, scale=args.diff_scale)

    # Decode from full image
    pred_full = decode_bits(encoded_finetuned, decoder_finetuned, device, stretch_to_256)
    acc_full = bit_accuracy(pred_full, gt_bits)

    # Decode from 10px-cropped image
    encoded_cropped = crop_edges(encoded_finetuned, pixels=10)
    pred_cropped = decode_bits(encoded_cropped, decoder_finetuned, device, stretch_to_256)
    acc_cropped = bit_accuracy(pred_cropped, gt_bits)

    print(f"\nCheckpoint  : {args.checkpoint_dir}")
    print(f"Input image : {args.input_path}")
    print(f"Message     : '{args.message}'")
    print("\n--- Finetuned ---")
    print(f"  Watermarked image : {wm_finetuned_path}")
    print(f"  Diff image        : {diff_path}")
    print(f"  Full image  — decoded: '{bits_to_text(pred_full)}'  bit acc: {acc_full:.4f}")
    print(f"  Cropped (-10px) — decoded: '{bits_to_text(pred_cropped)}'  bit acc: {acc_cropped:.4f}")

    # --- VINE-R baseline (optional) ---
    if not args.no_vine_r:
        encoded_vine_r = encode_image(input_image_pil, gt_watermark, encoder_vine_r, device, stretch_to_256)

        wm_vine_r_path = os.path.join(args.output_dir, f"{stem}_wm_vine_r.png")
        transforms.ToPILImage()(encoded_vine_r[0].cpu()).save(wm_vine_r_path)

        diff_vine_r_path = os.path.join(args.output_dir, f"{stem}_diff_original_vs_vine_r.png")
        save_visual_diff(args.input_path, wm_vine_r_path, diff_vine_r_path, scale=args.diff_scale)

        pred_vine_r_full = decode_bits(encoded_vine_r, decoder_vine_r, device, stretch_to_256)
        acc_vine_r_full = bit_accuracy(pred_vine_r_full, gt_bits)

        encoded_vine_r_cropped = crop_edges(encoded_vine_r, pixels=10)
        pred_vine_r_cropped = decode_bits(encoded_vine_r_cropped, decoder_vine_r, device, stretch_to_256)
        acc_vine_r_cropped = bit_accuracy(pred_vine_r_cropped, gt_bits)

        print("\n--- VINE-R baseline ---")
        print(f"  Watermarked image : {wm_vine_r_path}")
        print(f"  Diff image        : {diff_vine_r_path}")
        print(f"  Full image  — decoded: '{bits_to_text(pred_vine_r_full)}'  bit acc: {acc_vine_r_full:.4f}")
        print(f"  Cropped (-10px) — decoded: '{bits_to_text(pred_vine_r_cropped)}'  bit acc: {acc_vine_r_cropped:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a finetuned VINE checkpoint on one image.")
    parser.add_argument("--checkpoint_dir", type=str, default="output/finetune_curriculum/checkpoint-2000",
                        help="Path to finetune checkpoint directory.")
    parser.add_argument("--base_decoder", type=str, default="Shilin-LU/VINE-B-Dec",
                        help="Base decoder repo used to build architecture before loading checkpoint weights.")
    parser.add_argument("--vine_r_encoder", type=str, default="Shilin-LU/VINE-R-Enc")
    parser.add_argument("--vine_r_decoder", type=str, default="Shilin-LU/VINE-R-Dec")
    parser.add_argument("--input_path", type=str, default="frame_000150_original.png")
    parser.add_argument("--output_dir", type=str, default="results/test2000_new")
    parser.add_argument("--message", type=str, default="Hello World!", help="Watermark message (<=12 chars).")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--diff_scale", type=int, default=16, help="Amplification factor for diff image.")
    parser.add_argument("--no_vine_r", action="store_true",
                        help="Skip loading and running the VINE-R baseline model.")
    args = parser.parse_args()
    main(args)
