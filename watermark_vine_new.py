import math
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from vine.src.vine_turbo import VINE_Turbo
from vine.src.stega_encoder_decoder import CustomConvNeXt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
watermark_encoder = VINE_Turbo.from_pretrained("Shilin-LU/VINE-B-Enc", device=device).to(device).eval()
decoder = CustomConvNeXt.from_pretrained("Shilin-LU/VINE-B-Dec").to(device).eval()

t_val_256 = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
])


def text_to_bits(text: str):
    data = text.encode("utf-8")
    return [int(b) for byte in data for b in format(byte, "08b")]


def bits_to_text(bits):
    bits = bits[: len(bits) - (len(bits) % 8)]
    if not bits:
        return ""
    out = bytearray()
    for i in range(0, len(bits), 8):
        out.append(int("".join(str(x) for x in bits[i:i+8]), 2))
    return out.decode("utf-8", errors="replace")


def _bits100_tensor(bits100):
    if len(bits100) > 100:
        raise ValueError("chunk > 100 bits")
    padded = bits100 + [0] * (100 - len(bits100))
    return torch.tensor(padded, dtype=torch.float).unsqueeze(0).to(device)


def _crop_to_square(image: Image.Image):
    w, h = image.size
    m = min(w, h)
    left = (w - m) // 2
    top = (h - m) // 2
    return image.crop((left, top, left + m, top + m))


def _encode_single_frame(input_path: str, output_path: str, bits100):
    print(f"Encoding {input_path} with bits {bits100}")
    image = Image.open(input_path).convert("RGB")
    orig_w, orig_h = image.size

    resized_img = t_val_256(image)
    resized_img = (2.0 * resized_img - 1.0).unsqueeze(0).to(device)
    input_image = (2.0 * transforms.ToTensor()(image) - 1.0).unsqueeze(0).to(device)
    watermark = _bits100_tensor(bits100)

    with torch.no_grad():
        encoded_image_256 = watermark_encoder(resized_img, watermark)
        residual_256 = encoded_image_256 - resized_img
        residual_back = F.interpolate(residual_256, size=(orig_h, orig_w), mode="bicubic", align_corners=False)
        encoded_image = input_image + residual_back
        encoded_image = torch.clamp(encoded_image * 0.5 + 0.5, 0.0, 1.0)

    transforms.ToPILImage()(encoded_image[0].cpu()).save(output_path)


def watermark_multi_chunked(input_folder: str, output_folder: str, bits):
    os.makedirs(output_folder, exist_ok=True)
    frames = sorted([f for f in os.listdir(input_folder) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))])
    chunks = [bits[i:i + 100] for i in range(0, len(bits), 100)]
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
        _encode_single_frame(src, dst, chunk_bits)
        frames_per_chunk[chunk_idx] += 1
        encoded_frames += 1

    return {"chunks": n_chunks, "encoded_frames": encoded_frames, "frames_per_chunk": frames_per_chunk}


def decode_multi_chunked(folder_path: str, total_bits: int):
    frames = sorted([f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))])
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
