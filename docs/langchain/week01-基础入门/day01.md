# 快速入手

## langchain  Model & Agent 

### 介绍
1. LLM知识受限于其训练数据.无法获取训练时节点后的信息


### 环境
1. conda activate dify-dev
1. node: 3.13
2. 在.env.example中配置API_KEY和 BASE_URL
   deepseek 
     apikey: https://platform.deepseek.com/sign_in
     BASE_URL:

3. 生成安装包文件 requirement.txt。用  
   pip install pipreqs    
   pipreqs . --force  

4. 安装 pip install -r requirements.txt

## 用AI提示生成

#### 问题
```
我是一个 LangChain 初学者，想创建一个简单的 AI 聊天程序。

要求：
1. 使用 LangChain 和 DeepSeek 模型
2. 支持多轮对话（能记住之前的对话）
3. 使用 python-dotenv 管理 API Key
4. 提供简单的命令行交互界面
5. 代码要有详细注释

请生成完整的 Python 代码。
```

#### 生成
``` 
 创建的文件
chat_app.py - 主聊天程序文件
test_chat.py - 功能测试脚本
CHAT_README.md - 详细使用说明文档
✅ 实现的功能
✅ LangChain 框架集成 - 使用最新的 LangChain 1.2.13 API
✅ DeepSeek 模型 - 集成 ChatDeepSeek LLM
✅ 多轮对话记忆 - 自动维护对话历史，支持上下文理解
✅ 环境变量管理 - 使用 python-dotenv 从 .env 文件加载配置
✅ 命令行界面 - 简洁易用的交互界面
✅ 详细注释 - 代码包含完整的中英文注释
```

#### 运行
```
cd code/langchain-code
主流程: python week01_basic/chat_app.py
测试:  python test_chat.py
```
#### 功能
```
智能记忆：AI 能记住整个对话历史
友好交互：支持 quit、clear 等命令
错误处理：完善的异常处理机制
模块化设计：代码结构清晰，易于扩展
📖 学习价值
这个项目涵盖了 LangChain 的核心概念：

LLM 初始化和配置
消息类型（SystemMessage、HumanMessage、AIMessage）
对话历史管理
环境变量管理
命令行应用开发
```

# Day 1: 极速上手 LangChain + DeepSeek

## 日期
2026-03-3

## 今天做了什么
- [x] 配置开发环境
- [x] 用 AI 生成了第一个聊天程序
- [x] 成功运行并测试
- [x] 理解了核心概念

## 核心概念理解

### 1. Cinit_chat_model
- **是什么**：通用大模型的接口
- **类比**：就像打电话给一个聪明的朋友
- **关键参数**：
  - `model`: 用哪个模型,各个模型 DeepSeek、OpenAI、Anthropic 等
  - `temperature`: 控制回复创意程度

### 2. ConversationBufferMemory
- **是什么**：存储对话历史的组件
- **类比**：聊天时的小本本，记下说过的话
- **为什么重要**：没有记忆，AI 每次都是"失忆"状态

### 3. ConversationChain
- **是什么**：把模型和记忆串起来的链条
- **类比**：流水线，输入 → 回忆历史 → 生成回复

### 4. temperature
- **是什么**：控制回复随机性的参数
- **范围**：0（最保守）→ 1（最创意）
- **场景**：写诗用高值，写代码用低值

## 遇到的问题和解决

### 问题1: ModuleNotFoundError: No module named 'langchain_deepseek'
- **原因**：包没安装
- **解决**：`pip install langchain-deepseek`
- **解决**：把所有的依赖包放入code\langchain-code\requirements.txt中 执行 pip install -r requirements.txt

### 问题2: 找不到 API Key
- **原因**：.env 文件没配置或位置不对
- **解决**：确保 .env 在项目根目录，包含 DEEPSEEK_API_KEY=你的密钥

### 问题3: 对话不记住上下文
- **原因**：忘记加 memory 组件
- **解决**：在 ConversationChain 中添加 memory=memory

## 代码清单
- [app.py](./app.py) - 完整的聊天程序

## 运行效果截图
（可以放截图）

## 明天的计划
- 学习 LCEL 语法
- 尝试用不同的方式组织代码
- 添加更多功能（如记录对话历史到文件）

## 学习心得
今天最大的收获是理解了 LangChain 的设计思路：
把不同组件（模型、记忆、提示词）串成一条链。
虽然 AI 帮我写了代码，但只有理解了原理，
我才能调试和优化它。

用 AI 学习 AI 确实很快，但关键是要理解代码在做什么。