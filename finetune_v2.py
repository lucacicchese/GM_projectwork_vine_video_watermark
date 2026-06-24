import os
import gc
import math
import lpips
import torch
import torch.nn.functional as F
import wandb
from glob import glob
import numpy as np
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
import shutil
from diffusers.optimization import get_scheduler
from packaging import version
from transformers import AutoTokenizer, CLIPTextModel
import accelerate
from diffusers.utils.torch_utils import is_compiled_module
from vine.src.vine_turbo import VINE_Turbo, VAE_encode, VAE_decode
from vine.src.training_src.training_utils import parse_args
from vine.src.training_src.stega_utils import get_secret_acc, count_parameters
import vine.src.training_src.extra_utils as extra_utils
from kornia import color
from vine.src.stega_encoder_decoder import CustomConvNeXt
from vine.src.training_src.transformations import TransformNet
from vine.src.training_src.wm_modules import Discriminator
from datasets import load_dataset

IMAGE_SIZE = 256
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def sim_crop(x, scale=8, min_px=4, max_px=8):
    _, _, h, w = x.shape
    px = torch.randint(min_px, max_px + 1, (1,)).item()
    up = F.interpolate(x, size=(h * scale, w * scale), mode='bicubic', align_corners=False)
    cropped = up[:, :, px:-px, px:-px]
    return F.interpolate(cropped, size=(h, w), mode='bicubic', align_corners=False)


class HFDataset(torch.utils.data.Dataset):
    _to_256 = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    _to_512 = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
    ])

    def __init__(self, secret_size):
        self.secret_size = secret_size
        self.ds = load_dataset("fusing/instructpix2pix-1000-samples", split="train")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img = self.ds[idx]["input_image"].convert("RGB")
        img_256 = self._to_256(img) * 2.0 - 1.0
        img_512 = self._to_512(img) * 2.0 - 1.0
        secret = torch.randint(0, 2, (self.secret_size,)).float()
        return {"cover_img_256": img_256, "cover_img": img_512, "secret": secret,
                "prompt": self.ds[idx].get("edit_prompt", "")}


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.seed is not None:
        set_seed(args.seed)

    watermark_encoder = VINE_Turbo.from_pretrained("Shilin-LU/VINE-B-Enc")
    if hasattr(watermark_encoder, "vae_dec"):
        watermark_encoder.vae_dec.to(dtype=torch.float16)
    if hasattr(watermark_encoder, "vae_enc"):
        watermark_encoder.vae_enc.to(dtype=torch.float16)

    decoder = CustomConvNeXt.from_pretrained("Shilin-LU/VINE-B-Dec")
    decoder.to(accelerator.device)

    transform_net = TransformNet(device=accelerator.device)

    watermark_encoder.sec_encoder.requires_grad_(True)
    watermark_encoder.unet.requires_grad_(True)
    watermark_encoder.vae_a2b.requires_grad_(True)
    decoder.requires_grad_(True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    watermark_encoder.vae_a2b.to(accelerator.device)
    net_disc_a = Discriminator()

    gc.collect()
    torch.cuda.empty_cache()

    if args.enable_xformers_memory_efficient_attention:
        watermark_encoder.unet.enable_xformers_memory_efficient_attention()
    if args.gradient_checkpointing:
        watermark_encoder.unet.enable_gradient_checkpointing()
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    params_gen = (list(decoder.parameters()) +
                  list(watermark_encoder.sec_encoder.parameters()) +
                  list(watermark_encoder.unet.parameters()) +
                  list(watermark_encoder.vae_a2b.parameters()))
    params_sec = (list(decoder.parameters()) +
                  list(watermark_encoder.sec_encoder.parameters()))

    optimizer_gen = torch.optim.Adam(params_gen, lr=args.learning_rate)
    optimizer_sec = torch.optim.Adam(params_sec, lr=args.learning_rate)
    optimizer_disc = torch.optim.RMSprop(list(net_disc_a.parameters()), lr=0.00001)

    dataset_train = HFDataset(secret_size=args.secret_size)
    print(f"Training dataset size: {len(dataset_train)}")
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.train_batch_size, shuffle=True,
        num_workers=args.dataloader_num_workers, drop_last=True,
    )

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1
                while len(weights) > 0:
                    weights.pop()
                    model = models[i]
                    sub_dir = model.__class__.__name__
                    if isinstance(model, type(unwrap_model(watermark_encoder.sec_encoder))):
                        torch.save(model.state_dict(), os.path.join(output_dir, f'{sub_dir}.pth'))
                    elif isinstance(model, type(unwrap_model(decoder))):
                        torch.save(model.state_dict(), os.path.join(output_dir, f'{sub_dir}.pth'))
                    elif isinstance(model, type(unwrap_model(net_disc_a))):
                        torch.save(model.state_dict(), os.path.join(output_dir, f'{sub_dir}.pth'))
                    elif isinstance(model, type(unwrap_model(watermark_encoder.unet))):
                        torch.save(model.state_dict(), os.path.join(output_dir, f'{sub_dir}.pth'))
                    elif isinstance(model, type(unwrap_model(watermark_encoder.vae_enc))):
                        torch.save(model.vae.state_dict(), os.path.join(output_dir, 'vae.pth'))
                    i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                model = models.pop()
                if isinstance(model, type(unwrap_model(watermark_encoder.sec_encoder))):
                    path = os.path.join(input_dir, 'StegaStampEncoder.pth')
                    if os.path.exists(path):
                        model.load_state_dict(torch.load(path, map_location="cpu"))
                elif isinstance(model, type(unwrap_model(decoder))):
                    path = os.path.join(input_dir, 'CustomConvNeXt.pth')
                    if os.path.exists(path):
                        model.load_state_dict(torch.load(path, map_location="cpu"))
                elif isinstance(model, type(unwrap_model(net_disc_a))):
                    path = os.path.join(input_dir, 'Discriminator.pth')
                    if os.path.exists(path):
                        model.load_state_dict(torch.load(path, map_location="cpu"))
                elif isinstance(model, type(unwrap_model(watermark_encoder.unet))):
                    path = os.path.join(input_dir, 'UNet2DConditionModel.pth')
                    if os.path.exists(path):
                        model.load_state_dict(torch.load(path, map_location="cpu"))
                elif isinstance(model, VAE_encode):
                    path = os.path.join(input_dir, 'vae.pth')
                    if os.path.exists(path):
                        model.vae.load_state_dict(torch.load(path, map_location="cpu"))
                elif isinstance(model, VAE_decode):
                    path = os.path.join(input_dir, 'vae.pth')
                    if os.path.exists(path):
                        model.vae.load_state_dict(torch.load(path, map_location="cpu"))

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    lr_scheduler_gen = get_scheduler(args.lr_scheduler, optimizer=optimizer_gen,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power)
    lr_scheduler_sec = get_scheduler(args.lr_scheduler, optimizer=optimizer_sec,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power)
    lr_scheduler_disc = get_scheduler(args.lr_scheduler, optimizer=optimizer_disc,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power)

    net_lpips = lpips.LPIPS(net='vgg')
    net_lpips.to(accelerator.device)
    net_lpips.requires_grad_(False)
    cross_entropy = torch.nn.BCELoss().to(accelerator.device)

    tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer",
                                               revision=args.revision, use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder")
    text_encoder.requires_grad_(False)
    text_encoder.to(accelerator.device)
    fixed_a2b_tokens = tokenizer("", max_length=tokenizer.model_max_length, padding="max_length",
                                  truncation=True, return_tensors="pt").input_ids[0]
    watermark_encoder.fixed_a2b_emb_base = text_encoder(
        fixed_a2b_tokens.unsqueeze(0).to(accelerator.device))[0].detach()
    del text_encoder, tokenizer, fixed_a2b_tokens

    watermark_encoder.unet, watermark_encoder.vae_enc, watermark_encoder.vae_dec, \
        net_disc_a, decoder, watermark_encoder.sec_encoder, transform_net = accelerator.prepare(
            watermark_encoder.unet, watermark_encoder.vae_enc, watermark_encoder.vae_dec,
            net_disc_a, decoder, watermark_encoder.sec_encoder, transform_net,
        )
    net_lpips, optimizer_gen, optimizer_sec, optimizer_disc, train_dataloader, \
        lr_scheduler_sec, lr_scheduler_gen, lr_scheduler_disc = accelerator.prepare(
            net_lpips, optimizer_gen, optimizer_sec, optimizer_disc, train_dataloader,
            lr_scheduler_sec, lr_scheduler_gen, lr_scheduler_disc,
        )

    watermark_encoder.to(accelerator.device, dtype=weight_dtype)
    net_disc_a.to(accelerator.device)
    decoder.to(accelerator.device)
    net_lpips.to(accelerator.device, dtype=weight_dtype)

    if accelerator.is_main_process:
        accelerator.init_trackers(
            args.tracker_project_name, config=dict(vars(args)),
            init_kwargs={"wandb": {"name": args.key_change}},
        )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    first_epoch = 0
    global_step = 0

    if args.resume_from_checkpoint:
        path = os.path.basename(args.resume_from_checkpoint)
        if path is None:
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            gc.collect()
            torch.cuda.empty_cache()
    else:
        initial_global_step = 0

    progress_bar = tqdm(range(0, args.max_train_steps), initial=initial_global_step, desc="Steps",
                        disable=not accelerator.is_local_main_process)

    for name, module in net_disc_a.named_modules():
        if "attn" in name:
            module.fused_attn = False

    t_val = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])

    val_img = glob(os.path.join(args.val_folder, "*.jpg"))
    watermark_encoder.fixed_a2b_emb_base = (
        watermark_encoder.fixed_a2b_emb_base
        .repeat(args.train_batch_size, 1, 1)
        .to(dtype=weight_dtype)
    )
    timesteps = torch.tensor(
        [watermark_encoder.sched.config.num_train_timesteps - 1] * args.train_batch_size,
        device=accelerator.device,
    ).long()

    for epoch in range(first_epoch, args.max_train_epochs):
        for step, batch in enumerate(train_dataloader):
            l_acc = [net_disc_a, watermark_encoder, decoder]
            with accelerator.accumulate(*l_acc):
                img_a_256 = batch["cover_img_256"].to(dtype=weight_dtype)
                secret = batch["secret"].to(dtype=weight_dtype)

                no_im_loss = global_step < args.no_im_loss_steps
                l2_loss_scale = min(args.l2_loss_scale * global_step / max(args.l2_loss_ramp, 1), args.l2_loss_scale)
                lpips_loss_scale = min(args.lpips_loss_scale * global_step / max(args.lpips_loss_ramp, 1), args.lpips_loss_scale)
                G_loss_scale = min(args.G_loss_scale * global_step / max(args.G_loss_ramp, 1), args.G_loss_scale)

                encoded_image_256 = watermark_encoder(img_a_256, secret, timesteps)

                transformed_image = transform_net(encoded_image_256, img_a_256, global_step, args)
                avg_psnr = extra_utils.computePsnr(0.5 * (encoded_image_256 + 1), 0.5 * (img_a_256 + 1))
                transformed_image = transformed_image.to(device=accelerator.device, dtype=weight_dtype)
                transformed_image = (transformed_image / 2 + 0.5).clamp(0, 1)

                if torch.isnan(transformed_image).any() or torch.isinf(transformed_image).any():
                    optimizer_gen.zero_grad()
                    optimizer_sec.zero_grad()
                    continue

                secret_f = secret.float().clamp(0, 1)

                with torch.no_grad():
                    decoded_full = decoder((transformed_image * 2.0 - 1.0).float()).to(dtype=weight_dtype)
                bit_acc, str_acc = get_secret_acc(secret, decoded_full)

                lpips_loss = torch.mean(net_lpips(img_a_256, encoded_image_256))

                # decode from a differentiable crop so gradients reach the encoder
                encoded_01 = (encoded_image_256.float() * 0.5 + 0.5).clamp(0, 1)
                decoded_secret = decoder((sim_crop(encoded_01) * 2.0 - 1.0))
                decoded_secret = decoded_secret.to(dtype=weight_dtype)
                secret_loss = cross_entropy(decoded_secret.float().clamp(1e-6, 1 - 1e-6), secret_f)


                residual = encoded_image_256 - img_a_256
                h, w = residual.shape[2], residual.shape[3]

                border_width = max(1, int(0.06 * h))
                border_mask = torch.zeros(residual.shape[0], 1, h, w, device=residual.device, dtype=residual.dtype)
                border_mask[:, :, :border_width, :] = 1.0
                border_mask[:, :, -border_width:, :] = 1.0
                border_mask[:, :, :, :border_width] = 1.0
                border_mask[:, :, :, -border_width:] = 1.0
                edge_loss = (residual * border_mask).pow(2).mean()

                n_tiles = 4
                tile_energies = torch.stack([
                    residual[:, :, i * (h // n_tiles):(i + 1) * (h // n_tiles),
                                   j * (w // n_tiles):(j + 1) * (w // n_tiles)].pow(2).mean()
                    for i in range(n_tiles) for j in range(n_tiles)
                ])
                spread_loss = tile_energies.std() / (tile_energies.mean() + 1e-8)


                im_diff = encoded_image_256 - img_a_256
                image_loss = torch.mean(im_diff ** 2)

                D_output_fake_forG, _ = net_disc_a(encoded_image_256.detach())
                G_loss = D_output_fake_forG

                if no_im_loss:
                    loss = secret_loss
                    accelerator.backward(loss, retain_graph=False)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                    optimizer_sec.step()
                    lr_scheduler_sec.step()
                    optimizer_sec.zero_grad()
                else:
                    loss = (l2_loss_scale * image_loss
                            + lpips_loss_scale * lpips_loss
                            + args.secret_loss_scale * secret_loss)
                    if not args.no_gan:
                        loss += G_loss_scale * G_loss
                    loss += 50.0 * edge_loss
                    loss += 2.0 * spread_loss

                    accelerator.backward(loss, retain_graph=False)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                    optimizer_gen.step()
                    lr_scheduler_gen.step()
                    optimizer_gen.zero_grad()

                    if not args.no_gan:
                        D_output_real, _ = net_disc_a(0.5 * (img_a_256 + 1))
                        D_output_fake_forD, _ = net_disc_a(encoded_image_256.detach())
                        D_loss = D_output_real - D_output_fake_forD
                        accelerator.backward(D_loss, retain_graph=False)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(list(net_disc_a.parameters()), 0.25)
                        optimizer_disc.step()
                        lr_scheduler_disc.step()
                        optimizer_disc.zero_grad()
                        for p in net_disc_a.parameters():
                            p.data.clamp_(-0.01, 0.01)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                logs = {
                    "train_chart/loss": loss.detach().item(),
                    "train_chart/image_loss": image_loss.detach().item(),
                    "train_chart/lpips_loss": lpips_loss.detach().item(),
                    "train_chart/secret_loss": secret_loss.detach().item(),
                    "train_chart/edge_loss": edge_loss.detach().item(),
                    "train_chart/spread_loss": spread_loss.detach().item(),
                    "train_chart/bit_acc": bit_acc,
                    "train_chart/str_acc": str_acc,
                    "train_chart/psnr": avg_psnr,
                }
                if not args.no_gan:
                    logs["train_chart/loss_gan"] = G_loss.detach().item()
                    if not no_im_loss:
                        logs["train_chart/loss_D"] = D_loss.detach().item()

                if accelerator.is_main_process:
                    if global_step % args.viz_freq == 0:
                        for tracker in accelerator.trackers:
                            if tracker.name == "wandb":
                                log_dict = {
                                    "train/cover_img": [wandb.Image(batch["cover_img"][idx].float().detach().cpu()) for idx in range(1)],
                                    "train/watermarked": [wandb.Image(encoded_image_256[idx].float().detach().cpu()) for idx in range(1)],
                                    "train/transformed": [wandb.Image(transformed_image[idx].float().detach().cpu()) for idx in range(1)],
                                }
                                tracker.log(log_dict)
                                gc.collect()
                                torch.cuda.empty_cache()

                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        print(f"Saving checkpoint to {save_path}")
                        accelerator.save_state(save_path)
                        if args.checkpoints_total_limit is not None:
                            checkpoints = sorted(
                                [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")],
                                key=lambda x: int(x.split("-")[1]),
                            )
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                for removing in checkpoints[:len(checkpoints) - args.checkpoints_total_limit + 1]:
                                    shutil.rmtree(os.path.join(args.output_dir, removing))

                    if global_step % args.validation_steps == 0:
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.manual_seed(42)
                        secret_val = torch.randint(0, 2, (1, args.secret_size)).float()
                        secret_val = secret_val.to(accelerator.device, dtype=weight_dtype)

                        val_bit_acc_total = val_str_acc_total = val_avg_psnr = 0
                        val_image_loss = val_lpips_loss = val_secret_loss = 0
                        with torch.no_grad():
                            for i in range(len(val_img)):
                                input_image = Image.open(val_img[i]).convert('RGB')
                                input_image = t_val(input_image).unsqueeze(0).to(accelerator.device, dtype=weight_dtype)
                                input_image = input_image * 2 - 1
                                timesteps_val = torch.tensor(
                                    [watermark_encoder.sched.config.num_train_timesteps - 1],
                                    device=input_image.device,
                                ).long()
                                encoded_image = watermark_encoder(input_image, secret_val, timesteps_val)
                                transformed_image_val = transform_net(encoded_image, input_image, global_step, args)
                                transformed_image_val = 0.5 * (transformed_image_val + 1)

                                decoded_full_val = decoder(transformed_image_val.float())
                                encoded_01_val = (encoded_image.float() * 0.5 + 0.5).clamp(0, 1)
                                decoded_crop_val = decoder((sim_crop(encoded_01_val) * 2.0 - 1.0))
                                decoded_secret_val = (0.5 * decoded_full_val + 0.5 * decoded_crop_val).to(dtype=weight_dtype)

                                bit_acc_v, str_acc_v = get_secret_acc(secret_val, decoded_secret_val)
                                val_bit_acc_total += bit_acc_v
                                val_str_acc_total += str_acc_v
                                val_avg_psnr += extra_utils.computePsnr(0.5 * (encoded_image + 1), 0.5 * (input_image + 1))
                                val_lpips_loss += torch.mean(net_lpips(input_image, encoded_image))
                                val_secret_loss += cross_entropy(decoded_secret_val.float().clamp(1e-6, 1 - 1e-6), secret_val.float().clamp(0, 1))

                                im_diff_val = encoded_image - input_image
                                val_image_loss += torch.mean(im_diff_val ** 2)

                        n = len(val_img)
                        logs.update({
                            "val_chart/image_loss": val_image_loss.item() / n,
                            "val_chart/lpips_loss": val_lpips_loss.item() / n,
                            "val_chart/secret_loss": val_secret_loss.item() / n,
                            "val_chart/bit_acc": val_bit_acc_total / n,
                            "val_chart/str_acc": val_str_acc_total / n,
                            "val_chart/psnr": val_avg_psnr / n,
                        })

                    gc.collect()
                    torch.cuda.empty_cache()
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break


if __name__ == "__main__":
    args = parse_args()
    main(args)
