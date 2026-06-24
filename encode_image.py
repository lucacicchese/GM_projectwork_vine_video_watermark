import argparse
import os

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from vine.src.vine_turbo import VINE_Turbo


IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def text_to_bits(text: str):
    data = text.encode("utf-8")
    return [int(b) for byte in data for b in format(byte, "08b")]


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


def encode_image(input_path: str, output_path: str, message: str, checkpoint_dir: str, device: torch.device):
    stretch_to_256 = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])

    encoder = load_finetuned_encoder(checkpoint_dir, device)
    bits = text_to_bits(message)
    if len(bits) > 100:
        raise ValueError(f"Message too long: {len(bits)} bits (max 100, ~12 ASCII chars)")

    image = Image.open(input_path).convert("RGB")
    orig_w, orig_h = image.size

    resized_img = (2.0 * stretch_to_256(image) - 1.0).unsqueeze(0).to(device)
    input_image = (2.0 * transforms.ToTensor()(image) - 1.0).unsqueeze(0).to(device)

    padded = bits + [0] * (100 - len(bits))
    watermark = torch.tensor(padded, dtype=torch.float).unsqueeze(0).to(device)

    with torch.no_grad():
        encoded_image_256 = encoder(resized_img, watermark)
        residual_256 = encoded_image_256 - resized_img
        residual_back = F.interpolate(residual_256, size=(orig_h, orig_w), mode="bicubic", align_corners=False)
        encoded_image = input_image + residual_back
        encoded_image = torch.clamp(encoded_image * 0.5 + 0.5, 0.0, 1.0)

    transforms.ToPILImage()(encoded_image[0].cpu()).save(output_path)
    print(f"Encoded image saved to: {output_path}")


def main():
    p = argparse.ArgumentParser(description="Encode a watermark message into an image.")
    p.add_argument("--input", required=True, help="Path to the input image")
    p.add_argument("--output", required=True, help="Path to save the watermarked image")
    p.add_argument("--message", required=True, help="Text message to embed (max ~12 ASCII chars)")
    p.add_argument("--checkpoint_dir", type=str, default="output/finetune/checkpoint-2000")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use --device cpu or ensure CUDA is set up.")

    device = torch.device(args.device)
    encode_image(args.input, args.output, args.message, args.checkpoint_dir, device)


if __name__ == "__main__":
    main()
