"""
Function Calling 示例
使用OpenAI兼容API实现工具调用功能

环境变量配置：
- QWEN_API_KEY: API密钥（从阿里云控制台获取）
- QWEN_BASE_URL: API服务地址（默认 https://dashscope.aliyuncs.com/compatible-mode/v1）
"""

from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import logging

# 配置logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 加载环境变量
load_dotenv()

# API配置
api_key = os.getenv("QWEN_API_KEY")
if not api_key:
    raise ValueError("请设置 QWEN_API_KEY 环境变量（从阿里云控制台获取）")

base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# 初始化OpenAI客户端
client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)


# ===== 工具函数定义 =====

def get_weather(location: str, date: str = None) -> dict:
    """获取指定城市、日期的天气信息"""
    # 模拟天气数据
    mock_weather = {
        "北京": {"condition": "晴", "temperature": "15℃", "humidity": "40%"},
        "上海": {"condition": "多云", "temperature": "18℃", "humidity": "65%"},
        "深圳": {"condition": "阴", "temperature": "22℃", "humidity": "75%"}
    }

    weather_info = mock_weather.get(location, {
        "condition": "未知",
        "temperature": "未知",
        "humidity": "未知"
    })

    return {
        "location": location,
        "date": date or "今天",
        "condition": weather_info["condition"],
        "temperature": weather_info["temperature"],
        "humidity": weather_info["humidity"]
    }


def calculate(expression: str) -> dict:
    """执行数学计算"""
    try:
        # 安全检查
        allowed_chars = set('0123456789+-*/(). ')
        if not all(c in allowed_chars for c in expression):
            raise ValueError("表达式包含不安全的字符")

        result = eval(expression)
        return {
            "expression": expression,
            "result": result,
            "success": True
        }
    except Exception as e:
        return {
            "expression": expression,
            "error": str(e),
            "success": False
        }


# ===== 工具定义 =====

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市名称"
                    },
                    "date": {
                        "type": "string",
                        "description": "日期（可选）"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "执行数学表达式计算",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]


# ===== 核心功能 =====

def call_llm_with_tools(messages: list) -> dict:
    """调用大模型API，支持工具调用"""
    try:
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",  # 让模型决定是否使用工具
            temperature=0.1
        )

        return response.choices[0].message

    except Exception as e:
        logging.error(f"调用模型API失败: {e}")
        return None


def execute_tool(tool_name: str, tool_args: dict) -> dict:
    """执行工具函数"""
    tool_functions = {
        "get_weather": get_weather,
        "calculate": calculate
    }

    if tool_name not in tool_functions:
        return {"error": f"未知工具: {tool_name}"}

    try:
        func = tool_functions[tool_name]
        return func(**tool_args)
    except Exception as e:
        logging.error(f"执行工具 {tool_name} 失败: {e}")
        return {"error": str(e)}


def run_function_calling_example(user_query: str) -> str:
    """运行Function Calling示例"""
    print(f"\n=== 用户查询: {user_query} ===")

    # 初始化消息历史
    messages = [
        {
            "role": "system",
            "content": "你是一个智能助手，可以使用工具来帮助用户解决问题。当需要使用工具时，请调用相应的函数。"
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    # 第一次调用模型
    response_message = call_llm_with_tools(messages)

    if not response_message:
        return "抱歉，模型调用失败。"

    # 检查是否有工具调用
    if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
        print("🤖 模型决定使用工具...")

        # 处理所有工具调用
        for tool_call in response_message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            print(f"🔧 执行工具: {tool_name}")
            print(f"   参数: {tool_args}")

            # 执行工具
            tool_result = execute_tool(tool_name, tool_args)
            print(f"   结果: {tool_result}")

            # 添加工具调用结果到消息历史
            messages.append({
                "role": "assistant",
                "content": response_message.content or "",
                "tool_calls": [tool_call]
            })

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(tool_result, ensure_ascii=False)
            })

        # 基于工具结果获取最终回答
        final_response = call_llm_with_tools(messages)
        if final_response:
            return final_response.content
        else:
            return "生成最终回答失败。"

    else:
        # 模型直接回答
        print("💬 模型直接回答...")
        return response_message.content


# ===== 主函数 =====

def main():
    """主函数：运行Function Calling示例"""
    print("🚀 Function Calling 示例")
    print("=" * 40)

    # 测试用例
    test_queries = [
        "北京今天的天气怎么样？",
        "计算一下 25 + 17 * 3 的结果",
        "帮我介绍一下什么是Function Calling",
        "计算 (10 + 5) * 2"
    ]

    for query in test_queries:
        try:
            response = run_function_calling_example(query)
            print(f"🤖 助手回复: {response}")
            print("-" * 40)
        except Exception as e:
            print(f"❌ 处理查询失败: {e}")
            print("-" * 40)


if __name__ == "__main__":
    main()