#!/usr/bin/env python3
"""
测试聊天程序的功能
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from week01_basic.chat_app import AIChatApp

def test_chat_app():
    """测试聊天应用程序"""
    print("开始测试 AI 聊天程序...")

    # 创建聊天应用实例
    app = AIChatApp()

    # 测试基本对话
    print("\n=== 测试基本对话 ===")
    response1 = app.chat("你好")
    print(f"用户: 你好")
    print(f"AI: {response1}")

    # 测试多轮对话记忆
    print("\n=== 测试多轮对话记忆 ===")
    response2 = app.chat("我刚才说了什么？")
    print(f"用户: 我刚才说了什么？")
    print(f"AI: {response2}")

    # 测试清除历史
    print("\n=== 测试清除历史 ===")
    app.clear_history()
    response3 = app.chat("我刚才说了什么？")
    print(f"用户: 我刚才说了什么？")
    print(f"AI: {response3}")

    print("\n测试完成！")

if __name__ == "__main__":
    test_chat_app()