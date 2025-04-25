import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler
from diffusers.training_utils import free_memory
from PIL import Image


def log_validation(
    logger,
    pipeline,
    args,
    accelerator,
    pipeline_args,
    epoch,
    torch_dtype,
    is_final_validation=False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    scheduler_args = {}

    if "variance_type" in pipeline.scheduler.config:
        variance_type = pipeline.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)

    pipeline = pipeline.to(accelerator.device, dtype=torch_dtype)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None

    if args.validation_images is None:
        images = []
        for _ in range(args.num_validation_images):
            with torch.amp.autocast(accelerator.device.type):
                image = pipeline(**pipeline_args, generator=generator).images[0]
                images.append(image)
    else:
        images = []
        for image in args.validation_images:
            image = Image.open(image)
            with torch.amp.autocast(accelerator.device.type):
                image = pipeline(**pipeline_args, image=image, generator=generator).images[0]
            images.append(image)

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        # if tracker.name == "wandb":
        #     tracker.log(
        #         {
        #             phase_name: [
        #                 wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
        #             ]
        #         }
        #     )

    del pipeline
    free_memory()

    return images
