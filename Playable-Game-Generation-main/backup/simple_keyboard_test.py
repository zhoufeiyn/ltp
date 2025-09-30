#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版键盘测试程序 - 手动输入测试
"""

from train import map_Key_to_Action

def test_single_keys():
    """测试单个按键"""
    print("🔍 测试单个按键:")
    test_keys = ["r", "l", "j", "a", "f", "b", "s", "d", "enter"]
    
    for key in test_keys:
        action = map_Key_to_Action(key)
        print(f"  {key:>6} → {action:>3} ({action:08b})")
    print()

def test_combinations():
    """测试组合按键"""
    print("🔍 测试组合按键:")
    combinations = [
        ["r", "f"],           # 向右跑步
        ["r", "f", "j"],      # 向右跳跃跑步  
        ["l", "j"],           # 向左跳跃
        ["l", "f"],           # 向左跑步
        ["r", "f", "a"],       # 向右跳跃跑步 (a=jump)
        ["up", "right"],       # 向上+右
    ]
    
    for combo in combinations:
        action = map_Key_to_Action(combo)
        combo_str = " + ".join(combo)
        print(f"  {combo_str:>15} → {action:>3} ({action:08b})")
    print()

def interactive_test():
    """交互式测试"""
    print("🎮 交互式测试模式")
    print("💡 输入按键组合，用空格分隔，如: r f j")
    print("🛑 输入 'quit' 退出")
    print("-" * 40)
    
    while True:
        try:
            user_input = input("请输入按键: ").strip().lower()
            
            if user_input == "quit":
                print("👋 退出测试!")
                break
                
            if not user_input:
                continue
                
            # 解析输入
            keys = user_input.split()
            action = map_Key_to_Action(keys)
            
            print(f"🎯 按键: {' + '.join(keys)}")
            print(f"📊 Action值: {action}")
            print(f"🔢 二进制: {action:08b}")
            
            # 解释动作
            explain_action(action)
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n👋 退出测试!")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")

def explain_action(action):
    """解释动作值"""
    if action == 0:
        print("📝 解释: 无动作")
        return
        
    parts = []
    if action & 128: parts.append("A(跳跃)")
    if action & 64: parts.append("up(攀爬)")
    if action & 32: parts.append("left(左移)")
    if action & 16: parts.append("B(跑步/开火)")
    if action & 8: parts.append("start(开始)")
    if action & 4: parts.append("right(右移)")
    if action & 2: parts.append("down(下移)")
    if action & 1: parts.append("select(选择)")
    
    print(f"📝 解释: {' + '.join(parts)}")

def main():
    """主函数"""
    print("🎯 简化版键盘测试程序")
    print("=" * 50)
    
    # 运行所有测试
    test_single_keys()
    test_combinations()
    
    # 交互式测试
    interactive_test()

if __name__ == "__main__":
    main()
