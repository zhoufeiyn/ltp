# 0920 update: try to overfit level1-1 in one directory
import math
import os
import time
from turtle import st
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import re
from network.df.models.vae.autoencoder import AutoencoderKL
from network.df.models.diffusion.diffusion_forcing import DiffusionForcingBase
from network.df.config.Config_DF import ConfigDF
from network.df.algorithm import Algorithm
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
        self.image_files = [] # image files path (xxx.png)
        self.actions = [] # action (0-255)
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

def build_video_sequence(dataset, start_idx, end_idx, num_frames):
    """build video sequence in one batch from dataset, return [1, num_frames, ch, h, w]"""

    # 存储一个视频序列的数据
    video_images = []  # 存储当前视频的num_frames帧图像
    video_actions = []  # 存储当前视频的num_frames个动作
    video_nonterminals = []  # 存储当前视频的num_frames个nonterminals

    # 构建当前视频序列
    for frame_idx in range(start_idx, end_idx):
        image, action = dataset[frame_idx]
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

    # 返回tensor而不是列表
    return images_tensor, actions_tensor, nonterminals_tensor

def vae_encode(batch_data_images, vae_model, device):
    """vae encode the images"""
    # 将图像编码到潜在空间: [batch_size, num_frames, 3, 128, 128] -> [batch_size, num_frames, 4, 32, 32]
    with torch.no_grad():
        batch_size_videos, num_frames, channels, h, w = batch_data_images.shape
        # 重塑为 [batch_size * num_frames, 3, 128, 128] 进行批量编码
        images_flat = batch_data_images.reshape(-1, channels, h, w).to(device)
        
        # VAE编码
        if vae_model is not None:
            print(f"   Input image shape: {images_flat.shape}")
            latent_dist = vae_model.encode(images_flat)
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
        return latent_images

def train():
    device_obj = torch.device(device)
    print(f"🚀 device: {device_obj}")
    dataset = MarioDataset(cfg.data_path, cfg.image_size)

    # video sequence parameters
    num_frames = cfg.num_frames
    batch_size = cfg.batch_size


    model_name = cfg.model_name
    model_config = ConfigDF(model_name=model_name)
    
    # 使用Algorithm类加载完整的预训练模型（包含VAE和Diffusion）
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


    print("---start training----")
    print("---load dataset---")
    total_samples = len(dataset)
    # 检查是否有足够的数据
    if total_samples < num_frames:
        print(f"❌ dataset not enough: need at least {num_frames} samples, but only {total_samples} samples")
        return
    # 计算可以创建多少个完整的视频序列
    num_videos = total_samples // num_frames
    print(f"dataset loaded: {total_samples} samples, construct {num_videos} complete video sequences, each video has {num_frames} frames")
    
    
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        
        # 遍历所有视频序列
        for i in range(0, total_samples, batch_size*num_frames):
            print(f"---build video sequence in batch: {i}---")
            batch_images = []
            batch_actions = []
            batch_nonterminals = []
            
            # 检查是否有足够的数据构建完整批次
            if i + batch_size*num_frames > total_samples:
                print(f"⚠️ 跳过不完整的批次: 需要 {batch_size*num_frames} 个样本，但只有 {total_samples - i} 个样本")
                break
            
            for batch_idx in range(batch_size):
                start_idx = i + batch_idx*num_frames
                end_idx = start_idx + num_frames
                
                # 确保不超出数据集边界
                if end_idx > total_samples:
                    print(f"⚠️ 跳过不完整的视频序列: start_idx={start_idx}, end_idx={end_idx}, total_samples={total_samples}")
                    break
                    
                video_images, video_actions, video_nonterminals = build_video_sequence(dataset, start_idx, end_idx, num_frames)
                
                # 添加到批次列表中
                batch_images.append(video_images)
                batch_actions.append(video_actions)
                batch_nonterminals.append(video_nonterminals)
            
            # 拼接成批次tensor: [batch_size, num_frames, c, h, w]
            batch_data = [
                torch.cat(batch_images, dim=0).to(device_obj),
                torch.cat(batch_actions, dim=0).to(device_obj),
                torch.cat(batch_nonterminals, dim=0).to(device_obj)
            ]
           

            # 确保所有数据都在同一设备上
            print("VAE encoding images to latent space")
            batch_data[0] = vae_encode(batch_data[0], vae, device_obj)
            # 训练步骤
            try:
                out_dict = diffusion_model.training_step(batch_data)
                loss = out_dict["loss"]
                
                # 反向传播
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                total_loss += loss.item()
                batch_count += 1
                print(f"   Batch {batch_count}, Loss: {loss.item():.6f}")
                
            except Exception as e:
                print(f"   ❌ 训练步骤出错: {e}")
                print(f"   batch_data shapes:")
                print(f"     images: {batch_data[0].shape}")
                print(f"     actions: {batch_data[1].shape}")
                print(f"     nonterminals: {batch_data[2].shape}")
                raise e        
        
        # 计算平均损失
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")
        else:
            print(f"Epoch {epoch+1}/{epochs}, No batches processed")
    print("Training completed!")


if __name__ == "__main__":
    train()