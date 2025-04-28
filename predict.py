import torch
from diffusers import DiffusionPipeline
from jsonargparse import CLI


def main(
    lora_weights_fp: str = "lightning_logs/version_4/lora_weights.safetensors",
    prompt: str = "A photo of sks dog in a bucket",
    save_fp: str = "results/dog-bucket.png",
):
    pipeline = DiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        use_safetensors=True,
    ).to("cuda")

    pipeline.load_lora_weights(lora_weights_fp)

    image = pipeline(prompt, num_inference_steps=100, guidance_scale=7.5).images[0]
    image.save(save_fp)


if __name__ == "__main__":
    CLI(main, as_positional=False)
