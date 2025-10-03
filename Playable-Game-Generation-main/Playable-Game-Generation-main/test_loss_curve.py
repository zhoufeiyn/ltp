#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试损失曲线功能的简单脚本
"""

import os
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

def save_loss_curve(loss_history, epochs, save_path="output"):
    """保存损失曲线图（只包含每20个epoch的损失值）"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 创建损失曲线图
    plt.figure(figsize=(10, 6))
    # x轴是每20个epoch的索引，y轴是对应的损失值
    x_values = [(i + 1) * 20 for i in range(len(loss_history))]
    plt.plot(x_values, loss_history, 'b-', linewidth=2, label='Training Loss (every 20 epochs)')
    plt.title('Training Loss Curve', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H")
    loss_curve_path = os.path.join(save_path, f"loss_curve_{timestamp}.png")
    plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Loss curve saved to: {loss_curve_path}")
    return loss_curve_path

def test_loss_curve():
    """测试损失曲线功能（模拟每20个epoch记录一次损失）"""
    # 模拟训练过程中的损失数据（每20个epoch记录一次）
    epochs = 100
    loss_history = []
    
    # 生成模拟损失数据（每20个epoch记录一次）
    for epoch in range(0, epochs, 20):
        if epoch == 0:
            epoch = 20  # 第一个记录点是20
        # 模拟损失递减，加入一些噪声
        base_loss = 2.0 * np.exp(-epoch / 30)  # 指数衰减
        noise = np.random.normal(0, 0.05)  # 添加噪声
        loss = max(0.1, base_loss + noise)  # 确保损失不为负
        loss_history.append(loss)
        print(f"Epoch {epoch}/{epochs}, Average Loss: {loss:.6f}")
    
    print(f"\n记录了 {len(loss_history)} 个损失值（每20个epoch）")
    print(f"初始损失: {loss_history[0]:.4f}")
    print(f"最终损失: {loss_history[-1]:.4f}")
    
    # 保存损失曲线
    curve_path = save_loss_curve(loss_history, epochs)
    
    return curve_path

if __name__ == "__main__":
    curve_file = test_loss_curve()
    print(f"\n✅ 损失曲线测试完成！请检查图片文件: {curve_file}")
