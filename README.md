# RAG + Agent 知识库问答系统

基于 LangChain + ChromaDB + DeepSeek 构建的智能问答系统，支持文档检索、网页搜索和多轮对话。

## 功能特点

- **RAG知识库检索**：上传文档（txt/pdf/docx），系统自动分块、向量化、存储，并基于文档内容回答问题
- **Agent工具调用**：知识库找不到答案时，Agent自主决定调用网页搜索工具获取实时信息
- **多轮对话**：支持上下文连续追问，具备对话记忆能力
- **来源溯源**：回答附带引用的源文档信息
- **Web界面**：基于Streamlit构建，支持文件上传和实时问答

## 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| 框架 | LangChain | Agent编排和工具调用 |
| 向量数据库 | ChromaDB | 文档Embedding存储和相似度检索 |
| Embedding模型 | HuggingFace all-MiniLM-L6-v2 | 本地运行，零成本 |
| 大语言模型 | DeepSeek API | 生成回答 |
| Web界面 | Streamlit | 交互式Web应用 |
| 网页搜索 | DuckDuckGo Search | Agent的外部搜索工具 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置API Key

编辑 `rag.py`，将 `DEEPSEEK_API_KEY` 替换为你的DeepSeek API Key：
```python
DEEPSEEK_API_KEY = "sk-your-api-key-here"
```

### 3. 准备知识库

创建 `knowledge.txt`，写入你的知识库内容。

### 4. 启动命令行版本

```bash
python rag.py
```

### 5. 启动Web界面

```bash
streamlit run app.py
```

打开浏览器访问 `http://localhost:8501`

## 使用方式

1. 点击左侧「选择文件」上传文档（支持txt/pdf/docx）
2. 点击「加载到知识库」处理文档
3. 点击「启动Agent」初始化
4. 在输入框提问，Agent会自主选择从知识库检索还是网页搜索

## Agent工作原理

```
用户提问 → Agent分析问题 → 决策使用哪个工具
  ├── 文档相关问题 → 调用「知识库检索」工具 → 返回答案+来源
  └── 实时/外部信息 → 调用「网页搜索」工具 → 返回搜索结果
```

## 项目结构

```
rag-qa-system/
├── rag.py              # 核心逻辑：RAG + Agent + 多轮对话
├── app.py              # Streamlit Web界面
├── requirements.txt   # 依赖列表
├── knowledge.txt       # 知识库示例文档
├── chroma_db/          # 向量数据库（自动生成）
└── README.md
```
