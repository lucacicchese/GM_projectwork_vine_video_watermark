import argparse
import os

import torch
from PIL import Image
from torchvision import transforms

from vine.src.stega_encoder_decoder import CustomConvNeXt


def bits_to_text(bits):
    bits = bits[: len(bits) - (len(bits) % 8)]
    if not bits:
        return ""
    out = bytearray()
    for i in range(0, len(bits), 8):
        out.append(int("".join(str(x) for x in bits[i : i + 8]), 2))
    return out.decode("utf-8", errors="replace")


def load_finetuned_decoder(checkpoint_dir: str, base_decoder: str, device: torch.device) -> CustomConvNeXt:
    dec_path = os.path.join(checkpoint_dir, "CustomConvNeXt.pth")
    decoder = CustomConvNeXt.from_pretrained(base_decoder)
    if os.path.exists(dec_path):
        state = torch.load(dec_path, map_location="cpu")
        decoder.load_state_dict(state, strict=True)
    else:
        print(f"[decode_image] No CustomConvNeXt.pth in {checkpoint_dir} — decoder was frozen during training, using pretrained base weights.")
    decoder.to(device)
    decoder.eval()
    return decoder


def decode_image(input_path: str, checkpoint_dir: str, base_decoder: str, device: torch.device) -> str:
    stretch_to_256 = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])

    decoder = load_finetuned_decoder(checkpoint_dir, base_decoder, device)

    image = Image.open(input_path).convert("RGB")
    image_tensor = stretch_to_256(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = decoder(image_tensor)[0].detach().cpu().numpy()

    bits = [int(round(b)) for b in pred]
    return bits_to_text(bits)


def main():
    p = argparse.ArgumentParser(description="Decode a watermark message from an image.")
    p.add_argument("--input", required=True, help="Path to the watermarked image")
    p.add_argument("--checkpoint_dir", type=str, default="output/finetune/checkpoint-2000")
    p.add_argument("--base_decoder", type=str, default="Shilin-LU/VINE-B-Dec")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use --device cpu or ensure CUDA is set up.")

    device = torch.device(args.device)
    message = decode_image(args.input, args.checkpoint_dir, args.base_decoder, device)
    print(f"Decoded message: {message}")


if __name__ == "__main__":
    main()
