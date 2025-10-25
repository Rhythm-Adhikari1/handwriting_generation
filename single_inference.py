import argparse
import os
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
import torch
from data_loader.loader import  ContentData
from models.unet import UNetModel
from diffusers import AutoencoderKL
from models.diffusion import Diffusion
import torchvision
from PIL import Image
import numpy as np
import cv2
import math
from utils.util import fix_seed
import torch.distributed as dist


# ---- Laplacian kernel ----
lap_kernel_8 = np.array([[1, 1, 1],
                         [1, -8, 1],
                         [1, 1, 1]], dtype=np.float32)

def laplacian_transform(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lap_8 = cv2.filter2D(image, cv2.CV_64F, lap_kernel_8)
    lap_abs = cv2.convertScaleAbs(lap_8)
    return lap_abs

def resize_to_height64(image, fill_color=(255, 255, 255)):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    orig_width, orig_height = image.size
    scale = 64 / orig_height
    new_width = int(orig_width * scale)
    resized_image = image.resize((new_width, 64), Image.Resampling.LANCZOS)
    padded_width = math.ceil(new_width / 64) * 64
    padded_image = Image.new("RGB", (padded_width, 64), fill_color)
    padded_image.paste(resized_image, (0, 0))
    return padded_image

def process_image(image_path, min_width=64):
    image = Image.open(image_path).convert("RGB")
    image_resized = resize_to_height64(image)
    style_image = np.array(image_resized).astype(np.float32) / 255.0
    lap_image = laplacian_transform(np.array(image_resized)).astype(np.float32) / 255.0
    return style_image, lap_image

def main(opt):
   
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()
    """fix the random seed"""
    fix_seed(cfg.TRAIN.SEED)

    """ set mulit-gpu """
    dist.init_process_group(backend='nccl')
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)

    load_content = ContentData()
    totol_process = dist.get_world_size()


    
    """setup data_loader instances"""

    style_image, laplace_image = process_image(opt.image_path)

    target_dir = opt.save_dir

    diffusion = Diffusion(device=opt.device)

    """build model architecture"""
    unet = UNetModel(in_channels=cfg.MODEL.IN_CHANNELS, model_channels=cfg.MODEL.EMB_DIM, 
                     out_channels=cfg.MODEL.OUT_CHANNELS, num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS, 
                     attention_resolutions=(1,1), channel_mult=(1, 1), num_heads=cfg.MODEL.NUM_HEADS, 
                     context_dim=cfg.MODEL.EMB_DIM).to(opt.device)
    
    

    if len(opt.one_dm) > 0:
        unet.load_state_dict(torch.load(opt.one_dm, map_location=opt.device))
        print(f"✅ Loaded model from {opt.one_dm}")
    else:
        raise IOError("❌ Please provide a valid checkpoint path or use --one_dm auto")


    unet.eval()

    vae = AutoencoderKL.from_pretrained(opt.stable_dif_path, subfolder="vae")
    vae = vae.to(opt.device)
    # Freeze vae and text_encoder
    vae.requires_grad_(False)


   
    data_val, laplace= style_image, laplace_image
    x_text = opt.text
    data_loader = []
    # split the data into two parts when the length of data is two large
    if len(data_val) > 224:
        data_loader.append((data_val[:224], laplace[:224]))
        data_loader.append((data_val[224:], laplace[224:]))
    else:
        data_loader.append((data_val, laplace))
    for (data_val, laplace) in data_loader:
        style_input = torch.from_numpy(data_val).permute(2, 0, 1).unsqueeze(0).to(opt.device)
        laplace = torch.from_numpy(laplace).unsqueeze(0).unsqueeze(0).to(opt.device)

        text_ref = load_content.get_content(x_text)
        text_ref = text_ref.to(opt.device).repeat(style_input.shape[0], 1, 1, 1)
        x = torch.randn((text_ref.shape[0], 4, style_input.shape[2]//8, (text_ref.shape[1]*32)//8)).to(opt.device)
        
        if opt.sample_method == 'ddim':
            ema_sampled_images = diffusion.ddim_sample(unet, vae, style_input.shape[0], 
                                                    x, style_input, laplace, text_ref,
                                                    opt.sampling_timesteps, opt.eta)
        elif opt.sample_method == 'ddpm':
            ema_sampled_images = diffusion.ddpm_sample(unet, vae, style_input.shape[0], 
                                                    x, style_input, laplace, text_ref)
        else:
            raise ValueError('sample method is not supported')
        
        for index in range(len(ema_sampled_images)):
            im = torchvision.transforms.ToPILImage()(ema_sampled_images[index])
            image = im.convert("L")
            out_path = os.path.join(target_dir)
            os.makedirs(out_path, exist_ok=True)
            image.save(os.path.join(out_path, x_text + ".png"))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", type=str, required=True, help="YAML config path")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--save_dir", type=str, default="Generated", help="Directory to save outputs")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference")
    parser.add_argument("--one_dm", type=str, default="auto", help="Pretrained UNet path or auto")
    parser.add_argument("--stable_dif_path", type=str, default="runwayml/stable-diffusion-v1-5", help="VAE path")
    parser.add_argument("--sampling_timesteps", type=int, default=50)
    parser.add_argument("--sample_method", type=str, default="ddim")
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--drive_backup_dir", type=str, default='/content/drive/MyDrive/model_checkpoints')
    parser.add_argument("--local_model_dir", type=str, default='checkpoints')
    parser.add_argument("--text",type=str, required=True, help="Text prompt to feed into the model"
)

    opt = parser.parse_args()
    main(opt)
