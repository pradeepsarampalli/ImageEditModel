import sys
from pathlib import Path
from types import ModuleType

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPTokenizer

import config
from dataset import get_data


def _disable_torchao_peft_checks() -> None:
    for name, module in tuple(sys.modules.items()):
        if (name == "peft" or name.startswith("peft.")) and isinstance(
            module, ModuleType
        ):
            setattr(module, "is_torchao_available", lambda: False)


def train() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA GPU, such as a Colab T4.")

    device = torch.device("cuda")
    dtype = torch.float16
    tokenizer = CLIPTokenizer.from_pretrained(config.MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        config.MODEL_ID, subfolder="text_encoder", torch_dtype=dtype
    ).to(device)
    vae = AutoencoderKL.from_pretrained(
        config.MODEL_ID, subfolder="vae", torch_dtype=dtype
    ).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        config.MODEL_ID, subfolder="unet", torch_dtype=dtype
    ).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(
        config.MODEL_ID, subfolder="scheduler"
    )

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    unet.enable_gradient_checkpointing()

    _disable_torchao_peft_checks()
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_config)
    for parameter in unet.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()

    train_dataloader, _, _ = get_data()
    if len(train_dataloader) == 0:
        raise ValueError("The training dataset is empty.")

    optimizer = torch.optim.AdamW(
        (parameter for parameter in unet.parameters() if parameter.requires_grad),
        lr=config.LR,
    )
    scaler = torch.amp.GradScaler("cuda")
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    unet.train()
    vae.eval()
    text_encoder.eval()
    for epoch in range(config.EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        batch_count = len(train_dataloader)

        for batch_index, batch in enumerate(train_dataloader):
            source_images = batch["source_image"].to(device, dtype=dtype)
            target_images = batch["target_image"].to(device, dtype=dtype)
            prompts = list(batch["prompt"])
            tokens = tokenizer(
                prompts,
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
            input_ids = tokens.input_ids.to(device)

            with torch.no_grad():
                target_latents = vae.encode(target_images).latent_dist.sample()
                target_latents = target_latents * vae.config.scaling_factor
                image_latents = vae.encode(source_images).latent_dist.sample()
                image_latents = image_latents * vae.config.scaling_factor
                encoder_hidden_states = text_encoder(input_ids)[0]

            noise = torch.randn_like(target_latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (target_latents.shape[0],),
                device=device,
                dtype=torch.long,
            )
            noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)
            model_input = torch.cat((noisy_latents, image_latents), dim=1)

            with torch.autocast(device_type="cuda", dtype=dtype):
                prediction = unet(
                    model_input,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

            prediction_type = noise_scheduler.config.prediction_type
            if prediction_type == "epsilon":
                target = noise
            elif prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(target_latents, noise, timesteps)
            else:
                raise ValueError(f"Unsupported prediction type: {prediction_type}")

            loss = F.mse_loss(prediction.float(), target.float(), reduction="mean")
            group_start = (
                batch_index // config.GRAD_ACCUM_STEPS
            ) * config.GRAD_ACCUM_STEPS
            accumulation_size = min(
                config.GRAD_ACCUM_STEPS, batch_count - group_start
            )
            scaler.scale(loss / accumulation_size).backward()
            total_loss += loss.detach().item()

            is_accumulation_boundary = (
                (batch_index + 1) % config.GRAD_ACCUM_STEPS == 0
                or batch_index + 1 == batch_count
            )
            if is_accumulation_boundary:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        print(
            f"Epoch {epoch + 1}/{config.EPOCHS} "
            f"- loss: {total_loss / batch_count:.6f}"
        )

    unet.save_pretrained(output_dir, safe_serialization=True)
    print(f"Saved LoRA weights to {output_dir}")


if __name__ == "__main__":
    train()
