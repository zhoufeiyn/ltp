#!/usr/bin/env python3
"""
Colab环境设置脚本
在Colab中运行此脚本来设置环境并启动Flask应用
"""

import subprocess
import sys
import os

def install_requirements():
    """安装必要的依赖包"""
    print("正在安装依赖包...")
    
    packages = [
        "pyngrok",
        "flask",
        "flask-socketio", 
        "torch",
        "torchvision",
        "pillow",
        "numpy"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} 安装成功")
        except subprocess.CalledProcessError:
            print(f"✗ {package} 安装失败")
    
    print("依赖包安装完成！")

def check_colab_environment():
    """检查是否在Colab环境中"""
    try:
        import google.colab
        print("✓ 检测到Colab环境")
        return True
    except ImportError:
        print("✗ 未检测到Colab环境")
        return False

def setup_ngrok():
    """设置ngrok"""
    try:
        from pyngrok import ngrok
        print("✓ pyngrok已安装")
        
        # 可选：设置ngrok认证token
        # token = input("请输入ngrok认证token（可选，按回车跳过）: ").strip()
        # if token:
        #     ngrok.set_auth_token(token)
        #     print("✓ ngrok认证token已设置")
        
        return True
    except ImportError:
        print("✗ pyngrok未安装，请先运行安装命令")
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("Colab环境设置脚本")
    print("=" * 50)
    
    # 检查环境
    if not check_colab_environment():
        print("此脚本专为Colab环境设计")
        return
    
    # 安装依赖
    install_requirements()
    
    # 设置ngrok
    if setup_ngrok():
        print("\n环境设置完成！")
        print("现在可以运行: python app.py")
        print("=" * 50)
    else:
        print("\n环境设置部分完成")
        print("请手动安装pyngrok: !pip install pyngrok")
        print("然后运行: python app.py")
        print("=" * 50)

if __name__ == "__main__":
    main()

