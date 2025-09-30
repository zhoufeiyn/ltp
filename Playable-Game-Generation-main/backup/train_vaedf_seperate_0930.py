# 0920 update: try to overfit level1-1 in one directory
import math
import os
import time
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import re
from network.df.models.vae.autoencoder import AutoencoderKL
from network.df.models.diffusion.diffusion_forcing import DiffusionForcingBase
from network.df.config.Config_DF import ConfigDF

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import configTrain as cfg
import matplotlib.pyplot as plt
from PIL import Image
import imageio
from model import get_model, get_data, get_web_img

device: str = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Custom Dataset for Mario Data
# -----------------------------

class MarioDataset(Dataset):
    """load mario dataset __init__ action and img paths,
     __getitem__  will return image and corresponding action"""
    """up to date: 2025-09-20 only load all frames in one directory,
     return array ofimages and actions"""

    def __init__(self, data_path: str, image_size):
        self.data_path = data_path
        self.image_size = image_size
        self.image_files = []  # image files path (xxx.png)
        self.actions = []  # action (0-255)
        self._load_data()
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),  # [-1, 1]
        ])

    def _load_data(self):
        """load all png files and corresponding actions"""
        print(f"🔍 data path is scanning: {self.data_path}")
        if not os.path.exists(self.data_path):
            print(f"❌ data path not found: {self.data_path}")
            return

        total_files = 0
        valid_files = 0

        for root, dirs, files in os.walk(self.data_path):
            if root == self.data_path:
                continue
            for file in files:
                if file.lower().endswith('.png'):
                    total_files += 1
                    file_path = os.path.join(root, file)

                    # 尝试从文件名提取动作
                    action = self._extract_action_from_filename(file)
                    if action is not None:
                        self.image_files.append(file_path)
                        self.actions.append(action)
                        valid_files += 1
                    else:
                        print(f"⚠️ can't extract action from filename: {file}")

    def _extract_action_from_filename(self, filename: str) -> Optional[int]:
        """extract action from filename"""
        # 文件名格式: Rafael_dp2a9j4i_e6_1-1_f1000_a20_2019-04-13_20-13-16.win.png
        pattern = r'_a(\d+)_'
        match = re.search(pattern, filename)
        if match:
            return [int(match.group(1))]
        return None

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """get the data sample of the specified index"""
        if idx >= len(self.image_files):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.image_files)}")

        # 加载图像
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # 获取动作（如果有的话）
        action = self.actions[idx] if idx < len(self.actions) else 0

        return image, action


def map_Key_to_Action(key) -> int:
    """map key(s) to action based on SMB dataset encoding:
    A=128(jump), up=64(climb), left=32, B=16(run/fire),
    start=8, right=4, down=2, select=1

    Args:
        key: str or list of str - single key or list of pressed keys

    Examples:
        map_Key_to_Action("r") -> 4 (right)
        map_Key_to_Action(["r", "f"]) -> 20 (right + B = running right)
        map_Key_to_Action(["r", "f", "j"]) -> 148 (right + B + A = running jump right)
    """
    # 如果输入是单个键，转换为列表
    if isinstance(key, str):
        keys = [key]
    else:
        keys = key

    action = 0

    # 遍历所有按下的键，累加动作值
    for k in keys:
        if k == "r" or k == "right" or k == "→":
            action += 4  # right
        elif k == "l" or k == "left" or k == "←":
            action += 32  # left
        elif k == "j" or k == "a":
            action += 128  # A (jump)
        elif k == "up" or k == "↑":
            action += 64  # up (climb)
        elif k == "f" or k == "b":
            action += 16  # B (run/fire)
        elif k == "s":
            action += 8  # start
        elif k == "d" or k == "down" or k == "↓":
            action += 2  # down
        elif k == "enter":
            action += 1  # select

    return action


def build_video_sequence(dataset, num_frames):
    """build video sequence from dataset"""
    total_samples = len(dataset)

    # 检查是否有足够的数据
    if total_samples < num_frames:
        print(f"❌ dataset not enough: need at least {num_frames} samples, but only {total_samples} samples")
        return

    # 计算可以创建多少个完整的视频序列
    num_videos = total_samples // num_frames
    print(
        f"dataset loaded: {total_samples} samples, construct {num_videos} complete video sequences, each video has {num_frames} frames")

    # 存储所有视频序列的数据
    all_video_images = []  # 存储所有视频的图像
    all_video_actions = []  # 存储所有视频的动作
    all_video_nonterminals = []  # 存储所有视频的nonterminals

    # 创建多个视频序列
    for video_idx in range(num_videos):
        video_images = []  # 存储当前视频的8帧图像
        video_actions = []  # 存储当前视频的8个动作
        video_nonterminals = []  # 存储当前视频的8个nonterminals

        # 构建当前视频序列
        start_idx = video_idx * num_frames
        for frame_idx in range(num_frames):
            idx = start_idx + frame_idx
            image, action = dataset[idx]
            video_images.append(image)  # image shape: [3, 128, 128]
            video_actions.append(action[0])  # action是列表，取第一个元素
            video_nonterminals.append(True)  # 先都默认True

        # 转换为tensor并组织成目标格式
        # [num_frames, channels, h, w] = [num_frames, 3, 128, 128]
        images_tensor = torch.stack(video_images, dim=0)  # [num_frames, 3, 128, 128]
        images_tensor = images_tensor.unsqueeze(0)  # [1, num_frames, 3, 128, 128]

        # [batch_size, num_frames, action_dim] = [1, num_frames, 1]
        actions_tensor = torch.tensor(video_actions, dtype=torch.long)  # [num_frames]
        actions_tensor = actions_tensor.unsqueeze(0).unsqueeze(-1)  # [1, num_frames, 1]

        # [batch_size, num_frames] = [1, num_frames]
        nonterminals_tensor = torch.tensor(video_nonterminals, dtype=torch.bool)  # [num_frames]
        nonterminals_tensor = nonterminals_tensor.unsqueeze(0)  # [1, num_frames]

        # 添加到总列表中
        all_video_images.append(images_tensor)
        all_video_actions.append(actions_tensor)
        all_video_nonterminals.append(nonterminals_tensor)

    # 合并所有视频序列数据
    # 最终格式: [num_videos, num_frames, channels, h, w]
    batch_data = [
        torch.cat(all_video_images, dim=0),  # [num_videos, num_frames, 3, 128, 128]
        torch.cat(all_video_actions, dim=0),  # [num_videos, num_frames, 1]
        torch.cat(all_video_nonterminals, dim=0)  # [num_videos, num_frames]
    ]
    print(f"1.build video sequence completed")
    print(f"   batch_data[0] (images): {batch_data[0].shape}")  # [num_videos, 8, 3, 128, 128]
    print(f"   batch_data[1] (actions): {batch_data[1].shape}")  # [num_videos, 8, 1]
    print(f"   batch_data[2] (nonterminals): {batch_data[2].shape}")  # [num_videos, 8]
    return batch_data


def train():
    device_obj = torch.device(device)
    print(f"🚀 device: {device_obj}")
    dataset = MarioDataset(cfg.data_path, cfg.image_size)

    # video sequence parameters
    num_frames = cfg.num_frames
    print("1.build video sequence")
    batch_data = build_video_sequence(dataset, num_frames)

    model_name = cfg.model_name
    model_config = ConfigDF(model_name=model_name)

    # 使用Algorithm类加载完整的预训练模型（包含VAE和Diffusion）
    from network.df.algorithm import Algorithm
    model = Algorithm(model_name, device_obj)

    # 加载预训练checkpoint
    checkpoint_path = cfg.checkpoint_path
    if os.path.exists(checkpoint_path):
        print(f"📥 load pretrained checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
        model.load_state_dict(state_dict['network_state_dict'], strict=False)
        print("✅ Checkpoint loaded successfully！")
    else:
        print(f"⚠️ Checkpoint not found: {checkpoint_path},use random initialized model")
    model = model.to(device_obj)
    model.eval()  # 设置为评估模式，但允许训练

    # 获取VAE和Diffusion模型
    vae = model.vae if hasattr(model, 'vae') else None
    diffusion_model = model.df_model

    if vae is not None:
        vae.eval()
        print("✅ VAE already loaded")
    else:
        print("⚠️ Cannot find VAE model")
    epochs, lr, batch_size = cfg.epochs, cfg.lr, cfg.batch_size

    # 只优化diffusion模型，冻结VAE
    if vae is not None:
        # 冻结VAE参数
        for param in vae.parameters():
            param.requires_grad = False
        print("🔒 VAE parameters has been frozen")

    # 只优化diffusion模型参数
    diffusion_params = list(diffusion_model.parameters())
    opt = torch.optim.AdamW(diffusion_params, lr)

    print(f"   model device: {next(model.parameters()).device}")
    print(f"   Diffusion parameters number: {sum(p.numel() for p in diffusion_params if p.requires_grad)}")
    if vae is not None:
        print(f"   VAE parameters number: {sum(p.numel() for p in vae.parameters())} (frozen)")

    print("2.VAE encoding images to latent space")
    # 将图像编码到潜在空间: [batch_size, num_frames, 3, 128, 128] -> [batch_size, num_frames, 4, 32, 32]
    with torch.no_grad():
        batch_size_videos, num_frames, channels, h, w = batch_data[0].shape
        # 重塑为 [batch_size * num_frames, 3, 128, 128] 进行批量编码
        images_flat = batch_data[0].reshape(-1, channels, h, w).to(device)

        # VAE编码
        if vae is not None:
            print(f"   Input image shape: {images_flat.shape}")
            latent_dist = vae.encode(images_flat)
            latent_images = latent_dist.sample()  # 采样潜在表示
            print(f"   VAE encoded shape: {latent_images.shape}")
            # 使用正确的缩放因子
            from network.df.config.Config import Config
            latent_images = latent_images * Config.scale_factor  # 0.64
            print(f"   Using scale factor: {Config.scale_factor}")

            # 重塑回 [batch_size, num_frames, 4, 32, 32]
            latent_images = latent_images.reshape(batch_size_videos, num_frames, 4, 32, 32)
            print(f"   Reshaped shape: {latent_images.shape}")
        else:
            print("⚠️ Cannot find VAE model, use original image")
            # 如果没有VAE，直接使用原始图像，但需要调整形状
            latent_images = images_flat.reshape(batch_size_videos, num_frames, channels, h, w)
            print(f"   Using original image shape: {latent_images.shape}")

        # 更新batch_data[0]为编码后的潜在表示，保持在GPU上
        batch_data[0] = latent_images

    print("3.start training")
    num_videos = batch_data[0].shape[0]  # 总视频数量

    for epoch in range(epochs):
        total_loss = 0

        # 遍历所有视频序列
        for i in range(0, num_videos, batch_size):
            # 获取当前批次的数据
            end_idx = min(i + batch_size, num_videos)
            current_batch = [
                batch_data[0][i:end_idx].to(device),  # images: [batch_size, num_frames, c, h, w]
                batch_data[1][i:end_idx].to(device),  # actions: [batch_size, num_frames, action_dim]
                batch_data[2][i:end_idx].to(device)  # nonterminals: [batch_size, num_frames]
            ]

            # 确保所有数据都在同一设备上

            # 训练步骤
            try:
                out_dict = diffusion_model.training_step(current_batch)
                loss = out_dict["loss"]

                # 反向传播
                opt.zero_grad()
                loss.backward()
                opt.step()

                total_loss += loss.item()
                print(f"   Batch {i // batch_size + 1}, Loss: {loss.item():.6f}")

            except Exception as e:
                print(f"   ❌ 训练步骤出错: {e}")
                print(f"   current_batch shapes:")
                print(f"     images: {current_batch[0].shape}")
                print(f"     actions: {current_batch[1].shape}")
                print(f"     nonterminals: {current_batch[2].shape}")
                raise e
        avg_loss = total_loss / (num_videos // batch_size + (1 if num_videos % batch_size else 0))
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
    print("Training completed!")


if __name__ == "__main__":
    train()