
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel
import torch
import os

import argparse
# General Options
parser = argparse.ArgumentParser(description='ClasswiseNoise')
parser.add_argument('--model_path', type=str, default="./results/sd-monet-model/")
parser.add_argument('--checkpoint', type=int, default=0)
parser.add_argument('--ema', action='store_true', default=False)
parser.add_argument('--prompt_file', type=str, default="")
parser.add_argument('--basemodel', action='store_true', default=False)
parser.add_argument('--pretrained_model_name_or_path', type=str, default=None, required=True,)
parser.add_argument("--revision", type=str, default=None, required=False, help="Revision of pretrained model identifier from huggingface.co/models.",)
parser.add_argument("--non_ema_revision", type=str, default=None, required=False)
parser.add_argument("--prompt_postfix", type=str, default='', required=False)
parser.add_argument("--job_id", type=str, default='local', required=False)

args = parser.parse_args()

if not args.basemodel:
    if args.checkpoint > 0:
        os.system('rm -f {0}/unet/diffusion_pytorch_model.bin'.format(args.model_path))
        os.system('cp {0}/checkpoint-{1}/unet_ema/diffusion_pytorch_model.bin {0}/unet'.format(args.model_path, args.checkpoint, ))
        if args.ema:
            os.system('cp {0}/checkpoint-{1}/unet_ema/diffusion_pytorch_model.bin {0}/unet'.format(args.model_path, args.checkpoint, ))
        else:
            os.system('cp {0}/checkpoint-{1}/unet/diffusion_pytorch_model.bin {0}/unet'.format(args.model_path, args.checkpoint, ))

    pipe = StableDiffusionPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16)
    pipe.to("cuda")

    output_template = "{}/output/TEMPPROMPT_ckpt{}.png".format(args.model_path, args.checkpoint)

else:
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, cache_dir="/localscratch/renjie/huggingface", subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, cache_dir="/localscratch/renjie/huggingface", subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, cache_dir="/localscratch/renjie/huggingface", subfolder="unet", revision=args.non_ema_revision
    )

    pipe = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            cache_dir="/localscratch/renjie/huggingface",
            revision=args.revision,
        )
    pipe.to("cuda")

    output_template = "{}/output/TEMPPROMPT_basemodel.png".format(args.model_path)

if args.prompt_file == '':
    prompt="A tree{}".format(args.prompt_postfix.replace('_', ' '))
    image = pipe(prompt=prompt).images[0]
    image.save(output_template.replace('TEMPPROMPT', prompt.replace(' ', '_')))
else:
    f2 = open("./{}.txt".format(args.prompt_file),"r")
    lines = f2.readlines()
    for line3 in lines:
        prompt = line3.strip() + '{}'.format(args.prompt_postfix.replace('_', ' '))
        image = pipe(prompt=prompt).images[0]
        image.save(output_template.replace('TEMPPROMPT', prompt.replace(' ', '_')))
