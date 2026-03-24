import json

import streamlit as st
import time

from dotenv import load_dotenv
from openai import OpenAI

import os
import sys

# 添加模块搜索路径，由于导入的llm及common模块位于当前文件main.py的上上级目录。否则会报找不到module异常
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
# 添加模块路径到sys.path中
if module_path not in sys.path:
    sys.path.append(module_path)

from common import logger

from agent_architecture.memory.memory_manager import TrimMemoryManager, TrimSummarizeMemoryManager, \
    SummarizeMemoryManager, \
    PeriodSummarizeMemoryManager, HierarchicalMemoryManager

# 设置页面配置
st.set_page_config(
    page_title="智能聊天助手",
    page_icon="💬",
    layout="wide"
)

# 加载环境变量,读取.env文件配置信息
load_dotenv()
# 首先尝试从Streamlit secrets获取API密钥
if not os.environ.get("QWEN_API_KEY"):
    raise ValueError("请配置环境变量QWEN_API_KEY（千问模型密钥）以使用AI功能")

# 初始化OpenAI客户端
client = OpenAI(
    # 使用千问模型
    api_key=os.environ.get("QWEN_API_KEY"),
    base_url=os.environ.get("QWEN_BASE_URL"),
)

log = logger.configure_logging(logger_name="chat_memory_app")


# 生成回复
def generate_response(user_message: str) -> str:
    try:
        # 获取当前记忆上下文
        context = st.session_state.memory_manager.get_context()

        # 将上下文转换为所需的格式
        messages = []
        for msg in context:
            role = msg["role"]
            if role == "user":
                openai_role = "user"
            elif role == "assistant":
                openai_role = "assistant"
            else:  # system或其他角色
                openai_role = "system"
            messages.append({"role": openai_role, "content": msg["content"]})
        # 添加当前用户消息
        messages.append({"role": "user", "content": user_message})
        # 调用模型
        response = client.chat.completions.create(
            model="openai/gpt-5.2",
            messages=messages,
            temperature=0.7,
            max_tokens=200
        )

        # 提取回复内容
        return response.choices[0].message.content.strip()

    except Exception as e:
        st.error(f"调用模型 API时出错: {str(e)}")
        # 如果API调用失败，返回默认回复
        return "抱歉，我暂时无法回答这个问题。请稍后再试。"


# 应用样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        height: 600px;
        overflow-y: auto;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e1f5fe;
        border-radius: 10px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        max-width: 80%;
        align-self: flex-end;
    }
    .bot-message {
        background-color: #f1f8e9;
        border-radius: 10px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        max-width: 80%;
        align-self: flex-start;
    }
    .message-meta {
        font-size: 0.8rem;
        color: #666;
        margin-bottom: 0.25rem;
    }
    .parameter-panel {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0D47A1;
    }
</style>
""", unsafe_allow_html=True)


# 重置聊天消息
def reset_chat():
    st.session_state.messages = []
    # 添加欢迎消息
    welcome_msg = {
        "role": "assistant",
        "content": "👋 欢迎使用智能聊天助手！我可以回答您的问题并与您交流。\n\n请在下方输入框中输入您的问题，我会尽力为您解答。您也可以在左侧面板调整记忆模式和参数。"
    }
    st.session_state.messages.append(welcome_msg)
    st.session_state.memory_manager.add_message("assistant", welcome_msg["content"])


# 初始化聊天
def initialize_chat():
    if "memory_manager" not in st.session_state:
        st.session_state.memory_manager = TrimMemoryManager(max_length=4)

    if "messages" not in st.session_state:
        reset_chat()


# 主应用
def main():
    initialize_chat()

    # 页面标题
    st.markdown('<h1 class="main-header">智能聊天助手</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">支持多种记忆管理模式的AI聊天应用</p>', unsafe_allow_html=True)

    # 侧边栏参数设置
    with st.sidebar:
        st.markdown("### 🛠️ 记忆设置")

        # 记忆模式选择
        memory_mode = st.selectbox(
            "选择记忆模式",
            ["trim", "summarize", "trim_summarize", "period_summarize", "hierarchical"],
            format_func=lambda x: {
                "trim": "裁剪模式 (Trim)",
                "summarize": "总结模式 (Summarize)",
                "trim_summarize": "裁剪+总结模式 (Trim+Summarize)",
                "period_summarize": "定期总结模式 (Period Summarize)",
                "hierarchical": "分层记忆模式 (Hierarchical)"
            }.get(x, x)
        )

        # 记忆长度设置
        if memory_mode != "hierarchical":
            memory_length = st.slider("记忆长度 (消息数量)", 3, 20, 4)

        # 分层记忆参数（仅对Hierarchical模式有效）
        short_term_length = 5
        long_term_length = 5
        if memory_mode == "hierarchical":
            short_term_length = st.slider("短期记忆长度", 2, 10, 2)
            long_term_length = st.slider("长期记忆长度", 2, 10, 2)
            summary_threshold = st.slider("摘要阈值 (总结的消息数量)", 2, 10, 4)

        # 定期总结间隔（仅对Regularly Summarize模式有效）
        # summary_interval = 5
        # if memory_mode == "regularly_summarize":
        #    summary_interval = st.slider("总结间隔 (消息数量)", 2, 10, 5)

        # 应用设置按钮
        if st.button("应用设置"):
            # 根据选择的模式创建相应的记忆管理器
            if memory_mode == "trim":
                st.session_state.memory_manager = TrimMemoryManager(max_length=memory_length)
            elif memory_mode == "summarize":
                st.session_state.memory_manager = SummarizeMemoryManager(max_length=memory_length)
            elif memory_mode == "trim_summarize":
                st.session_state.memory_manager = TrimSummarizeMemoryManager(max_length=memory_length)
            elif memory_mode == "period_summarize":
                st.session_state.memory_manager = PeriodSummarizeMemoryManager(
                    max_length=memory_length
                )
            elif memory_mode == "hierarchical":
                st.session_state.memory_manager = HierarchicalMemoryManager(
                    short_term_length=short_term_length,
                    long_term_length=long_term_length,
                    summary_threshold=summary_threshold
                )
            # 重置聊天历史
            reset_chat()
            # st.session_state.messages = []
            if memory_mode != "hierarchical":
                welcome_msg = {
                    "role": "assistant",
                    "content": f"👋 已应用新的记忆设置！当前模式: {memory_mode}, 记忆长度: {memory_length}"
                }
            else:
                welcome_msg = {
                    "role": "assistant",
                    "content": f"👋 已应用新的记忆设置！当前模式: {memory_mode}, 短期记忆长度: {short_term_length}, 长期记忆长度:"
                               f" {long_term_length},摘要阈值: {summary_threshold}"
                }
            st.session_state.messages.append(welcome_msg)
            st.session_state.memory_manager.add_message("assistant", welcome_msg["content"])

            st.success("设置已更新！")

        # 显示当前使用的token数量
        token_count = st.session_state.memory_manager.get_context_token_count()
        st.markdown(f"### 📊 当前Token使用量: {token_count}")

        st.markdown("---")
        st.markdown("### 📖 记忆模式说明")
        st.markdown("""
        - **裁剪模式 (Trim)**: 只保留最近的对话历史
        - **总结模式 (Summarize)**: 自动总结对话历史
        - **裁剪+总结模式 (Trim+Summarize)**: 结合裁剪和总结策略
        - **定期总结模式 (Period Summarize)**: 按固定间隔总结对话
        - **分层记忆模式 (Hierarchical)**: 区分短期和长期记忆
        """)
    # 聊天界面
    # st.markdown('<div class="chat-container" id="chat-container">', unsafe_allow_html=True)
    # 显示聊天历史
    history_messages = st.session_state.messages
    log.info(f" 当前聊天历史:{json.dumps(history_messages, ensure_ascii=False)}")
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'''
                <div class="user-message" style="margin-left: auto;">
                    <div class="message-meta">用户 @ {time.strftime("%H:%M:%S")}</div>
                    <p>{message["content"]}</p>
                </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
                <div class="bot-message">
                    <div class="message-meta">助手 @ {time.strftime("%H:%M:%S")}</div>
                    <p>{message["content"]}</p>
                </div>
            ''', unsafe_allow_html=True)

    # st.markdown('</div>', unsafe_allow_html=True)

    # 发送消息
    if user_input := st.chat_input("请问有什么可以帮助您？"):
        # 添加用户消息到会话和记忆
        user_message = {"role": "user", "content": user_input}
        st.markdown(f'''
            <div class="user-message" style="margin-left: auto;">
                <div class="message-meta">用户 @ {time.strftime("%H:%M:%S")}</div>
                <p>{user_input}</p>
            </div>
        ''', unsafe_allow_html=True)
        st.session_state.messages.append(user_message)
        st.session_state.memory_manager.add_message("user", user_input)

        # 生成回复
        with st.spinner("思考中..."):
            response = generate_response(user_input)
            bot_message = {"role": "assistant", "content": response}
            st.markdown(f'''
                 <div class="bot-message">
                     <div class="message-meta">助手 @ {time.strftime("%H:%M:%S")}</div>
                     <p>{response}</p>
                 </div>
             ''', unsafe_allow_html=True)
            st.session_state.messages.append(bot_message)
            st.session_state.memory_manager.add_message("assistant", response)
            log.info(f"AI生成回复后内容:{json.dumps(st.session_state.messages, ensure_ascii=False)}")

        # 滚动到底部
        st.markdown("""
        <script>
            window.scrollTo({
                top: document.body.scrollTop,
                behavior: "smooth"
            });
        </script>
        """, unsafe_allow_html=True)

    # 显示当前记忆上下文
    st.markdown(" 🧠 当前模型上下文")
    context = st.session_state.memory_manager.get_context()
    log.info(f"当前上下文内容:{json.dumps(context, ensure_ascii=False)}")
    if context:
        with st.expander("查看当前上下文内容"):
            context_text = "\n\n".join([f"**{msg['role']}**: {msg['content']}" for msg in context])
            st.info(context_text)
    else:
        st.info("上下文为空")


if __name__ == "__main__":
    main()
