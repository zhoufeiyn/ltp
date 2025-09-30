"""Train Model Global Config"""

image_size: int = 128      # 256x240 -> 128*128
in_ch: int = 3             # RGB
base_ch: int = 64          # 减少基础通道数以适应GPU内存
num_actions: int = 46
num_frames: int = 4
model_name: str = "df_z32_c1_dit_n11_mario_km_tanh_ldm"


out_dir: str = "./output"
data_path: str = "./datatrain"

batch_size: int = 1        # 单张图像过拟合
epochs: int = 2          # 测试2个epoch
lr: float = 1e-3           # 提高学习率加速过拟合
num_samples: int = 1  # 只生成一张样本

grad_clip: float = 1.0
ode_steps: int = 10        # 采样时 ODE 欧拉步数