from dotenv import load_dotenv
from openai import OpenAI
import json
import os
from datetime import datetime
import logging

# 配置logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 加载环境变量
load_dotenv()

"""
工具增强型智能体Demo
使用OpenAI兼容API（这里示例使用阿里云千问）实现工具调用功能

环境变量配置：
- QWEN_API_KEY: API密钥（从阿里云控制台获取）
- QWEN_BASE_URL: API服务地址（默认 https://dashscope.aliyuncs.com/compatible-mode/v1）
"""

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


# ===== 工具定义 =====

def get_weather(location: str, date: str = None) -> dict:
    """获取指定城市、日期的天气信息

    Args:
        location: 城市名称
        date: 日期（格式：YYYY-MM-DD），默认为今天

    Returns:
        天气信息字典
    """
    # 模拟天气数据（实际应用中应调用真实天气API）
    mock_weather_data = {
        "北京": {
            "2024-01-15": {"condition": "晴", "temperature": "5℃", "humidity": "30%"},
            "2024-01-16": {"condition": "多云", "temperature": "3℃", "humidity": "35%"},
        },
        "上海": {
            "2024-01-15": {"condition": "小雨", "temperature": "8℃", "humidity": "70%"},
            "2024-01-16": {"condition": "阴", "temperature": "7℃", "humidity": "75%"},
        },
        "深圳": {
            "2024-01-15": {"condition": "晴", "temperature": "22℃", "humidity": "60%"},
            "2024-01-16": {"condition": "多云", "temperature": "20℃", "humidity": "65%"},
        }
    }

    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    city_data = mock_weather_data.get(location, {})
    weather_info = city_data.get(date, {
        "condition": "未知",
        "temperature": "未知",
        "humidity": "未知"
    })

    return {
        "location": location,
        "date": date,
        "condition": weather_info["condition"],
        "temperature": weather_info["temperature"],
        "humidity": weather_info["humidity"]
    }


def calculate(expression: str) -> dict:
    """执行数学计算

    Args:
        expression: 数学表达式字符串

    Returns:
        计算结果字典
    """
    try:
        # 安全检查：只允许基本数学运算符
        allowed_chars = set('0123456789+-*/(). ')
        if not all(c in allowed_chars for c in expression):
            raise ValueError("表达式包含不安全的字符")

        # 计算结果
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


def search_web(query: str) -> dict:
    """模拟网页搜索功能

    Args:
        query: 搜索查询

    Returns:
        搜索结果字典
    """
    # 模拟搜索结果（实际应用中应调用真实搜索API）
    mock_results = {
        "Python教程": [
            {"title": "Python官方教程", "url": "https://docs.python.org/3/tutorial/"},
            {"title": "Python入门指南", "url": "https://www.python.org/about/gettingstarted/"}
        ],
        "机器学习": [
            {"title": "Scikit-learn文档", "url": "https://scikit-learn.org/stable/"},
            {"title": "TensorFlow教程", "url": "https://www.tensorflow.org/tutorials/"}
        ]
    }

    results = mock_results.get(query, [
        {"title": f"关于'{query}'的搜索结果", "url": f"https://example.com/search?q={query}"}
    ])

    return {
        "query": query,
        "results": results[:3],  # 限制返回前3个结果
        "total_results": len(results)
    }


# ===== 工具注册 =====

# 工具元数据定义
TOOL_METADATA = {
    "get_weather": {
        "function": get_weather,
        "description": "获取指定城市和日期的天气信息",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "城市名称"
                },
                "date": {
                    "type": "string",
                    "description": "日期（YYYY-MM-DD格式）"
                }
            },
            "required": ["location"]
        },
        "usage_description": "查询指定城市和日期的天气状况",
        "param_descriptions": {
            "location": "城市名称（必需）",
            "date": "日期（格式：YYYY-MM-DD，可选，默认为今天）"
        }
    },
    "calculate": {
        "function": calculate,
        "description": "执行数学计算",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式"
                }
            },
            "required": ["expression"]
        },
        "usage_description": "执行数学表达式计算",
        "param_descriptions": {
            "expression": "数学表达式（必需，支持+-*/()）"
        }
    },
    "search_web": {
        "function": search_web,
        "description": "搜索网页信息",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词"
                }
            },
            "required": ["query"]
        },
        "usage_description": "搜索相关网页信息",
        "param_descriptions": {
            "query": "搜索关键词（必需）"
        }
    }
}

# 可用工具字典（从元数据生成）
TOOLS = {name: meta["function"] for name, meta in TOOL_METADATA.items()}

# 动态生成工具描述（用于构建系统提示词）
def generate_tool_descriptions() -> str:
    descriptions = ["你是一个智能助手，可以使用以下工具来帮助用户解决问题："]
    for i, (name, meta) in enumerate(TOOL_METADATA.items(), 1):
        desc = f"\n{i}. **{meta['usage_description']}** ({name})\n   - 功能：{meta['description']}\n   - 参数："
        for param_name, param_desc in meta["param_descriptions"].items():
            required = "（必需）" if param_name in meta["parameters"]["required"] else "（可选）"
            desc += f"\n     - {param_name}: {param_desc} {required}"
        descriptions.append(desc)

    descriptions.append("""
使用说明：
- 如果需要使用工具，请以JSON格式回复，格式如下：
  {
    "name": "工具名",
    "parameters": {
      "参数名1": "值1",
      "参数名2": "值2"
    }
  }
- 如果不需要使用工具，直接回答用户问题即可。""")

    return "".join(descriptions)

TOOL_DESCRIPTIONS = generate_tool_descriptions()

# 动态生成OpenAI工具定义
def generate_openai_tools() -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": meta["description"],
                "parameters": meta["parameters"]
            }
        }
        for name, meta in TOOL_METADATA.items()
    ]


# ===== 核心功能 =====

def parse_tool_call(response_text: str) -> dict:
    """解析模型回复中的工具调用JSON

    Args:
        response_text: 模型回复文本

    Returns:
        工具调用字典或None
    """
    try:
        # 查找JSON格式的工具调用
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1

        if start_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            tool_call = json.loads(json_str)

            # 验证必需字段
            if "name" in tool_call and "parameters" in tool_call:
                return tool_call

        return None
    except (json.JSONDecodeError, KeyError) as e:
        logging.warning(f"解析工具调用失败: {e}")
        return None


def call_llm(messages: list, use_tools: bool = True) -> str:
    """调用大模型API

    Args:
        messages: 消息列表
        use_tools: 是否启用工具调用

    Returns:
        模型回复文本
    """
    try:
        # 如果启用工具调用，使用function calling格式
        if use_tools:
            # 使用动态生成的工具定义
            tools = generate_openai_tools()

            response = client.chat.completions.create(
                model="qwen-plus",
                messages=messages,
                tools=tools,
                tool_choice="auto",  # 让模型决定是否使用工具
                temperature=0.1
            )
        else:
            # 不使用工具的普通调用
            response = client.chat.completions.create(
                model="qwen-plus",
                messages=messages,
                temperature=0.1
            )

        return response.choices[0].message

    except Exception as e:
        logging.error(f"调用模型API失败: {e}")
        return None


def execute_tool(tool_name: str, parameters: dict) -> dict:
    """执行工具函数

    Args:
        tool_name: 工具名称
        parameters: 工具参数

    Returns:
        工具执行结果
    """
    if tool_name not in TOOLS:
        return {"error": f"未知工具: {tool_name}"}

    try:
        tool_func = TOOLS[tool_name]
        result = tool_func(**parameters)
        return result
    except Exception as e:
        logging.error(f"执行工具 {tool_name} 失败: {e}")
        return {"error": str(e)}


def run_agent(user_query: str) -> str:
    """运行工具增强型智能体

    Args:
        user_query: 用户查询

    Returns:
        智能体回复
    """
    print(f"\n=== 用户查询: {user_query} ===")

    # 构建系统提示词
    system_prompt = f"""你是一个智能助手，可以使用工具来帮助用户解决问题。

{TOOL_DESCRIPTIONS}

请根据用户的问题判断是否需要使用工具。如果需要，请使用适当的工具；如果不需要，直接回答即可。"""

    # 初始化消息历史
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

    # 第一次调用模型
    response_message = call_llm(messages, use_tools=True)

    if not response_message:
        return "抱歉，模型调用失败，请稍后重试。"

    # 检查是否有工具调用
    tool_call = None
    if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
        print("🤖 模型决定使用工具...")
        tool_call = response_message.tool_calls[0]  # 假设只有一个工具调用
    elif response_message.content:
        # 尝试解析content中的JSON工具调用
        tool_call = parse_tool_call(response_message.content)
        if tool_call:
            print("🤖 模型决定使用工具...")

    if tool_call:
        # 处理工具调用
        if hasattr(tool_call, 'function'):
            # OpenAI格式
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
        else:
            # JSON格式
            tool_name = tool_call["name"]
            tool_args = tool_call["parameters"]

        print(f"🔧 执行工具: {tool_name}，参数: {tool_args}")

        # 执行工具
        tool_result = execute_tool(tool_name, tool_args)

        # 添加工具调用和结果到消息历史
        if hasattr(response_message, 'tool_calls'):
            messages.append({
                "role": "assistant",
                "content": response_message.content or "",
                "tool_calls": response_message.tool_calls
            })
        else:
            messages.append({
                "role": "assistant",
                "content": response_message.content
            })

        messages.append({
            "role": "tool",
            "tool_call_id": getattr(tool_call, 'id', 'manual_call'),
            "content": json.dumps(tool_result, ensure_ascii=False)
        })

        # 基于工具结果再次调用模型获取最终回答
        final_response = call_llm(messages, use_tools=False)
        if final_response:
            return final_response.content
        else:
            return "基于工具结果生成回答失败。"

    else:
        # 模型直接回答
        print("💬 模型直接回答...")
        return response_message.content


# ===== 主函数和测试 =====

def main():
    """主函数：运行demo示例"""
    print("🚀 工具增强型智能体Demo")
    print("=" * 50)

    # 测试用例
    test_queries = [
        "北京今天的天气怎么样？",
        "计算一下 (25 + 17) * 3 的结果",
        "帮我搜索一下Python机器学习相关的教程",
        "深圳明天天气如何？",
        "计算 2**10 + 100"
    ]

    for query in test_queries:
        try:
            response = run_agent(query)
            print(f"🤖 助手回复: {response}")
            print("-" * 50)
        except Exception as e:
            print(f"❌ 处理查询失败: {e}")
            print("-" * 50)


if __name__ == "__main__":
    main()