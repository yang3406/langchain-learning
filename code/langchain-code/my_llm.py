# 两种方式 1. 直接使用对应的模型类  
from langchain_deepseek import ChatDeepSeek

from env_util import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

deepSeek_llm = ChatDeepSeek(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    model="deepseek-chat"
)

# resp = deepSeek_llm.invoke("你好");

# print(type(resp))
# print(resp)
