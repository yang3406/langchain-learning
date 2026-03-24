# 示例：如何配置不同的模型提供商

# 1. 安装必要的依赖包
# 在 requirements.txt 中添加：
# langchain-openai==0.2.14  # OpenAI 支持
# langchain-anthropic==0.2.14  # Anthropic 支持

# 2. 在 .env 文件中添加相应的环境变量
# OPENAI_API_KEY=你的OpenAI_API密钥
# ANTHROPIC_API_KEY=你的Anthropic_API密钥

# 3. 在 config.py 中取消注释相应的配置
# 例如，启用 OpenAI 配置：
"""
MODEL_CONFIGS = {
    "openai": {
        "model_name": "gpt-4",  # 或 "gpt-3.5-turbo"
        "model_provider": "openai",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": None,  # OpenAI 使用默认 base_url
    },
}
"""

# 4. 修改 CURRENT_MODEL 变量
# CURRENT_MODEL = "openai"

# 5. 运行程序
# python week01_basic/chat_app.py

# 支持的模型提供商和模型：
# - DeepSeek: deepseek-chat, deepseek-coder
# - OpenAI: gpt-4, gpt-4-turbo, gpt-3.5-turbo
# - Anthropic: claude-3-opus, claude-3-sonnet, claude-3-haiku, claude-2.1

# 注意：确保你的 API 密钥有相应的权限，并且网络连接正常。