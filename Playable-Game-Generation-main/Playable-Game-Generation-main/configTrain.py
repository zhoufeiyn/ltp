"""Train Model Global Config"""

image_size: int = 128      # 256x240 -> 128*128
in_ch: int = 3             # RGB
base_ch: int = 64          # 减少基础通道数以适应GPU内存
num_actions: int = 46
num_frames: int = 16
model_name: str = "df_z32_c1_dit_n11_mario_km_tanh_ldm"

checkpoint_path: str= "ckpt/model.pth"
out_dir: str = "./output"
data_path: str = "./datatrain"
ckpt_path: str = "./ckpt"


best_save_interval: int = 50  # 最佳模型保存间隔（大于num个epoch,且超过最小改善幅度，保存一次最佳模型）
min_improvement: float = 0.15  # 最小改善幅度（15%）

batch_size: int = 1        # 单张图像过拟合
epochs: int = 200          # 测试epoch数量
lr: float = 1e-3          # 提高学习率加速过拟合
num_samples: int = 1  # 只生成一张样本

sample_step: int = 4
test_img_path: str = "./eval_data/demo1.png"
actions = ['r','r','r','r','r','r','r','r']


grad_clip: float = 1.0
ode_steps: int = 10        # 采样时 ODE 欧拉步数