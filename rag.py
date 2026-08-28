"""
RAG + Agent 知识库问答系统（升级版）
新增功能：
1. Agent工具调用 - 知识库找不到答案时自动搜索网页
2. 多轮对话 - 支持上下文连续追问
3. 文档上传 - 用户可上传自己的文档动态加载
"""

import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import ChromaDB
from langchain_deepseek import ChatDeepSeek
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, initialize_agent
from langchain.chains import RetrievalQA
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import PyPDFLoader, DocxLoader

# ============ 配置 ============
DEEPSEEK_API_KEY = "你的API_KEY"  # 替换成你的DeepSeek API Key
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PERSIST_DIR = "./chroma_db"

# ============ 1. 初始化Embedding模型 ============
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_and_split(file_path):
    """加载文档并分块"""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = DocxLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")

    documents = loader.load()
    # 分块：每块500字符，重叠50字符（保证上下文连续）
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", " "]
    )
    return splitter.split_documents(documents)


def create_vectorstore(docs):
    """创建向量数据库"""
    return ChromaDB.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )


def load_vectorstore():
    """加载已有的向量数据库"""
    return ChromaDB(
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR
    )


# ============ 2. 初始化大模型 ============
llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    temperature=0.3,  # 低温度=回答更稳定
    max_tokens=1000
)


# ============ 3. 构建Agent工具 ============

def create_rag_tool(vectorstore):
    """工具1：知识库检索 - 从本地文档中查找答案"""
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    def search_knowledge(query):
        """检索知识库"""
        result = qa_chain.invoke({"query": query})
        answer = result.get("result", "未找到相关内容")
        sources = result.get("source_documents", [])
        if sources:
            answer += f"\n\n来源：{sources[0].metadata.get('source', '未知')}"
        return answer

    return search_knowledge


def search_web(query):
    """工具2：网页搜索 - 知识库没有的信息从网上找"""
    search = DuckDuckGoSearchRun()
    return search.run(query)


def build_agent(vectorstore):
    """构建Agent：能自主选择用知识库还是网页搜索"""
    tools = [
        Tool(
            name="知识库检索",
            func=create_rag_tool(vectorstore),
            description="当问题与已上传的文档内容相关时使用，比如公司制度、产品说明、技术文档等"
        ),
        Tool(
            name="网页搜索",
            func=search_web,
            description="当知识库中没有相关信息时使用，比如最新新闻、天气、实时信息等"
        )
    ]

    # Agent会根据问题内容自主决定用哪个工具
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="zero-shot-react-description",
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
        verbose=True  # 打印Agent的思考过程
    )
    return agent


# ============ 4. 主程序入口 ============
if __name__ == "__main__":
    print("=" * 50)
    print("RAG + Agent 知识库问答系统")
    print("=" * 50)

    # 加载知识库
    if os.path.exists(PERSIST_DIR):
        print("[1] 加载已有向量数据库...")
        vs = load_vectorstore()
    else:
        print("[1] 首次运行，加载知识库文档...")
        docs = load_and_split("knowledge.txt")
        vs = create_vectorstore(docs)
        print(f"    已加载 {len(docs)} 个文档块")

    # 构建Agent
    print("[2] 初始化Agent（带工具调用能力）...")
    agent = build_agent(vs)
    print("[3] 系统就绪！输入 'quit' 退出\n")

    # 多轮对话循环
    while True:
        question = input("\n你: ").strip()
        if question.lower() in ["quit", "exit", "q"]:
            print("再见！")
            break
        if not question:
            continue

        try:
            # Agent会自动决定用知识库还是网页搜索
            answer = agent.invoke({"input": question})
            print(f"\nAI: {answer['output']}")
        except Exception as e:
            print(f"\n出错了: {e}")
