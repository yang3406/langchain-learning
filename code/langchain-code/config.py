"""
模型配置管理
支持多种大模型提供商的配置

使用方法：
1. 在 .env 文件中设置相应的环境变量
2. 在此文件中选择要使用的模型配置
3. 程序会自动使用对应的配置初始化模型
"""

import os
from env_util import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

# 模型配置字典
# 每个配置包含：model_name, model_provider, api_key, base_url
MODEL_CONFIGS = {
    "deepseek": {
        "model_name": "deepseek-chat",
        "model_provider": "deepseek",
        "api_key": DEEPSEEK_API_KEY,
        "base_url": DEEPSEEK_BASE_URL,
    },

    # 示例：OpenAI 配置（需要设置相应环境变量）
    "openai": {
        "model_name": "gpt-4",  # 或 "gpt-3.5-turbo"
        "model_provider": "openai",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": None,  # OpenAI 使用默认 base_url
    },

    # 示例：Anthropic 配置（需要设置相应环境变量）
    "anthropic": {
        "model_name": "claude-3-sonnet-20240229",
        "model_provider": "anthropic",
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
        "base_url": None,  # Anthropic 使用默认 base_url
    },
}

# 当前使用的模型配置
# 可以修改此变量来切换不同的模型
CURRENT_MODEL = "deepseek"

def get_current_model_config():
    """
    获取当前使用的模型配置

    Returns:
        dict: 当前模型的配置信息
    """
    if CURRENT_MODEL not in MODEL_CONFIGS:
        raise ValueError(f"未知的模型配置: {CURRENT_MODEL}")

    config = MODEL_CONFIGS[CURRENT_MODEL]
    if not config["api_key"]:
        raise ValueError(f"API key 未设置，请在 .env 文件中配置 {CURRENT_MODEL.upper()}_API_KEY")

    return config

def list_available_models():
    """
    列出所有可用的模型配置

    Returns:
        list: 可用模型名称列表
    """
    return list(MODEL_CONFIGS.keys())