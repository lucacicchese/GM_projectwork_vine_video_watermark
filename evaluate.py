import os
import cv2
import numpy as np
import torch
import lpips
from scipy import linalg
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision.models.video import r3d_18


def load_video_frames(path, fps_sample=None):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    step = 1
    if fps_sample is not None and src_fps > 0:
        step = max(1, int(round(src_fps / fps_sample)))

    frames = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        i += 1

    cap.release()
    if len(frames) == 0:
        raise ValueError(f"No frames from {path}")
    return frames


def compute_psnr_ssim(video1, video2):
    psnr_vals, ssim_vals = [], []
    for f1, f2 in zip(video1, video2):
        if f1.shape != f2.shape:
            f2 = cv2.resize(f2, (f1.shape[1], f1.shape[0]), interpolation=cv2.INTER_AREA)
        psnr_vals.append(peak_signal_noise_ratio(f1, f2, data_range=255))
        ssim_vals.append(structural_similarity(f1, f2, channel_axis=2))
    return float(np.mean(psnr_vals)), float(np.mean(ssim_vals))


def compute_lpips(video1, video2):
    loss_fn = lpips.LPIPS(net='alex').eval()
    vals = []
    with torch.no_grad():
        for f1, f2 in zip(video1, video2):
            if f1.shape != f2.shape:
                f2 = cv2.resize(f2, (f1.shape[1], f1.shape[0]), interpolation=cv2.INTER_AREA)
            t1 = torch.tensor(f1).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
            t2 = torch.tensor(f2).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
            vals.append(loss_fn(t1, t2).item())
    return float(np.mean(vals))


def compute_metrics_sampled(folder_orig: str, folder_comp: str, every_n: int = 30):
    orig_paths = sorted([os.path.join(folder_orig, f) for f in os.listdir(folder_orig) if f.endswith('.png')])
    comp_paths = sorted([os.path.join(folder_comp, f) for f in os.listdir(folder_comp) if f.endswith('.png')])

    n = min(len(orig_paths), len(comp_paths))
    orig_paths = orig_paths[:n:every_n]
    comp_paths = comp_paths[:n:every_n]

    loss_fn = lpips.LPIPS(net='alex').eval()
    psnr_vals, ssim_vals, lpips_vals = [], [], []

    with torch.no_grad():
        for p1, p2 in zip(orig_paths, comp_paths):
            f1 = cv2.cvtColor(cv2.imread(p1), cv2.COLOR_BGR2RGB)
            f2 = cv2.cvtColor(cv2.imread(p2), cv2.COLOR_BGR2RGB)

            if f1.shape != f2.shape:
                f2 = cv2.resize(f2, (f1.shape[1], f1.shape[0]), interpolation=cv2.INTER_AREA)

            psnr_vals.append(peak_signal_noise_ratio(f1, f2, data_range=255))
            ssim_vals.append(structural_similarity(f1, f2, channel_axis=2))

            t1 = torch.tensor(f1).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
            t2 = torch.tensor(f2).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
            lpips_vals.append(loss_fn(t1, t2).item())

    return float(np.mean(psnr_vals)), float(np.mean(ssim_vals)), float(np.mean(lpips_vals))


def _video_to_r3d_tensor(frames, size=(112, 112)):
    out = []
    for f in frames:
        x = cv2.resize(f, size, interpolation=cv2.INTER_AREA)
        x = torch.tensor(x).permute(2, 0, 1).float() / 255.0
        out.append(x)
    x = torch.stack(out, dim=1).unsqueeze(0)
    mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1)
    return (x - mean) / std


def _extract_video_features_r3d(videos, device='cpu'):
    model = r3d_18(weights=None)
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()

    feats = []
    with torch.no_grad():
        for frames in videos:
            x = _video_to_r3d_tensor(frames).to(device)
            f = model(x).squeeze(0).cpu().numpy()
            feats.append(f)
    return np.stack(feats, axis=0)


def _frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def compute_fvd(real_videos, generated_videos, device='cpu'):
    real_feats = _extract_video_features_r3d(real_videos, device=device)
    gen_feats = _extract_video_features_r3d(generated_videos, device=device)

    mu_r = real_feats.mean(axis=0)
    mu_g = gen_feats.mean(axis=0)

    sigma_r = np.cov(real_feats, rowvar=False)
    sigma_g = np.cov(gen_feats, rowvar=False)

    return _frechet_distance(mu_r, sigma_r, mu_g, sigma_g)
