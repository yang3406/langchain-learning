#!/usr/bin/env python3
"""
简单的 AI 聊天程序
使用 LangChain 和通用大模型，支持多轮对话

功能特性：
- 使用 LangChain 框架
- 集成通用大模型（支持 DeepSeek、OpenAI、Anthropic 等）
- 支持多轮对话记忆
- 使用 python-dotenv 管理 API Key
- 简单的命令行交互界面
- 详细的代码注释
- 易于切换不同的模型提供商

作者：AI Assistant
日期：2024年
"""

import os
import sys
from typing import Optional

# LangChain 相关导入
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.chat_models import init_chat_model

# 导入自定义的环境变量工具和配置
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env_util import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
from config import get_current_model_config


class AIChatApp:
    """
    AI 聊天应用程序类

    封装了聊天功能的核心逻辑，包括：
    - 通用 LLM 初始化（支持多种模型提供商）
    - 对话历史管理
    - 用户交互处理
    """

    def __init__(self):
        """
        初始化聊天应用程序

        使用通用的模型初始化方式，支持多种模型提供商
        设置对话历史管理
        """
        print("正在初始化 AI 聊天程序...")

        try:
            # 获取当前模型配置
            model_config = get_current_model_config()

            # 使用通用的模型初始化方式
            # 支持多种模型提供商（如 DeepSeek、OpenAI、Anthropic 等）
            self.llm = init_chat_model(
                model=model_config["model_name"],           # 模型名称
                model_provider=model_config["model_provider"], # 模型提供商
                api_key=model_config["api_key"],            # API 密钥
                base_url=model_config["base_url"],          # API 基础URL（可为None）
                temperature=0.7,                            # 创造性参数，0.7 适合聊天
                max_tokens=1000,                            # 限制响应长度
            )

            print(f"使用模型: {model_config['model_provider']} - {model_config['model_name']}")

            # 初始化对话历史列表
            # 使用列表存储 HumanMessage 和 AIMessage 对象
            self.conversation_history = [
                SystemMessage(content="你是一个友好的AI助手。请用中文回复用户的问题，并保持友好的对话风格。")
            ]

            print("AI 聊天程序初始化完成！")
            print("=" * 50)

        except Exception as e:
            print(f"初始化失败：{e}")
            sys.exit(1)

    def chat(self, user_input: str) -> str:
        """
        处理用户输入并返回 AI 响应

        Args:
            user_input (str): 用户的输入文本

        Returns:
            str: AI 的响应内容
        """
        try:
            # 将用户输入添加到对话历史
            self.conversation_history.append(HumanMessage(content=user_input))

            # 调用 LLM 生成响应
            response = self.llm.invoke(self.conversation_history)

            # 提取响应内容
            if hasattr(response, 'content'):
                ai_response = response.content
            else:
                ai_response = str(response)

            # 将 AI 响应添加到对话历史
            self.conversation_history.append(AIMessage(content=ai_response))

            return ai_response

        except Exception as e:
            error_msg = f"抱歉，处理您的消息时出现错误：{e}"
            print(f"错误详情：{e}")
            return error_msg

    def clear_history(self):
        """
        清除对话历史

        重置对话历史，只保留系统消息
        """
        self.conversation_history = [
            SystemMessage(content="你是一个友好的AI助手。请用中文回复用户的问题，并保持友好的对话风格。")
        ]
        print("对话历史已清除。")

    def run(self):
        """
        运行聊天应用程序的主循环

        提供命令行交互界面，循环处理用户输入
        """
        print("欢迎使用 AI 聊天程序！")
        print("输入 'quit'、'exit' 或 'bye' 退出程序")
        print("输入 'clear' 清除对话历史")
        print("-" * 50)

        while True:
            try:
                # 获取用户输入
                user_input = input("\n您：").strip()

                # 检查退出命令
                if user_input.lower() in ['quit', 'exit', 'bye', '退出']:
                    print("\n再见！感谢使用 AI 聊天程序。")
                    break

                # 检查清空历史命令
                elif user_input.lower() == 'clear':
                    self.clear_history()
                    continue

                # 检查空输入
                elif not user_input:
                    print("请输入您的问题...")
                    continue

                # 处理用户输入并获取响应
                print("AI：", end="", flush=True)
                response = self.chat(user_input)

                # 显示 AI 响应
                print(response)

            except KeyboardInterrupt:
                # 处理 Ctrl+C 中断
                print("\n\n程序被用户中断。")
                break

            except EOFError:
                # 处理 Ctrl+D 或输入流结束
                print("\n\n输入流结束。")
                break

            except Exception as e:
                print(f"\n发生未知错误：{e}")
                print("请重试或联系技术支持。")


def main():
    """
    主函数

    创建并运行聊天应用程序
    """
    try:
        # 创建聊天应用实例
        app = AIChatApp()

        # 运行聊天程序
        app.run()

    except KeyboardInterrupt:
        print("\n\n程序被终止。")
    except Exception as e:
        print(f"程序运行出错：{e}")
        sys.exit(1)


if __name__ == "__main__":
    # 当脚本直接运行时，执行主函数
    main()