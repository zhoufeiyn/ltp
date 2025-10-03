#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试日志功能的简单脚本
"""

import os
import logging
from datetime import datetime

def setup_logging():
    """设置日志记录"""
    # 创建logs目录
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 生成日志文件名（包含时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"test_training_log_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"训练日志已初始化，日志文件: {log_path}")
    return logger, log_path

def test_logging():
    """测试日志记录功能"""
    logger, log_path = setup_logging()
    
    # 模拟训练过程中的损失记录
    for epoch in range(1, 6):
        # 模拟损失值
        avg_loss = 1.0 - epoch * 0.15  # 模拟损失递减
        
        # 记录损失到日志文件和控制台
        loss_message = f"Epoch {epoch}/5, Average Loss: {avg_loss:.6f}"
        print(loss_message)
        logger.info(loss_message)
        
        # 模拟最佳损失更新
        if epoch == 3:
            best_message = f"🎉 new best loss: {avg_loss:.6f} (improvement: 25.00%)"
            print(best_message)
            logger.info(best_message)
            
            save_message = f"💾 save best model (improvement: 25.00%)"
            print(save_message)
            logger.info(save_message)
    
    # 训练完成
    completion_message = "Training completed!"
    print(completion_message)
    logger.info(completion_message)
    
    # 记录统计信息
    stats_message = f"📊 training statistics: total epochs: 5, best loss: 0.550000, final loss: 0.250000, total batches: 50"
    print(f"📊 training statistics:")
    print(f"    total epochs: 5")
    print(f"    best loss: 0.550000")
    print(f"    final loss: 0.250000")
    print(f"    total batches: 50")
    logger.info(stats_message)
    
    # 记录日志文件路径
    final_log_message = f"训练日志已保存至: {log_path}"
    print(final_log_message)
    logger.info(final_log_message)
    
    return log_path

if __name__ == "__main__":
    log_file = test_logging()
    print(f"\n✅ 日志测试完成！请检查日志文件: {log_file}")
