"""
RAG + Agent 知识库问答系统 - Web界面（升级版）
新增功能：
1. 文档上传 - 支持txt/pdf/docx格式
2. 多轮对话 - 显示聊天历史
3. Agent思考过程 - 可查看Agent选择了哪个工具
4. 来源溯源 - 显示答案来自哪个文档
"""

import streamlit as st
import os
import tempfile
from rag import (
    load_and_split, create_vectorstore, load_vectorstore,
    build_agent, embeddings, llm, PERSIST_DIR
)

# ============ 页面配置 ============
st.set_page_config(
    page_title="RAG + Agent 知识库问答系统",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG + Agent 知识库问答系统")
st.markdown("---")

# ============ 初始化Session State ============
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []


# ============ 侧边栏：文档管理 ============
with st.sidebar:
    st.header("📁 文档管理")

    # 显示当前知识库状态
    if os.path.exists(PERSIST_DIR):
        st.success("✅ 知识库已加载")
    else:
        st.warning("⚠️ 知识库为空，请上传文档")

    st.markdown("---")

    # 文档上传
    st.subheader("上传文档")
    uploaded_file = st.file_uploader(
        "选择文件上传",
        type=["txt", "pdf", "docx"],
        help="支持 txt、pdf、docx 格式"
    )

    if uploaded_file is not None:
        if st.button("📥 加载到知识库"):
            with st.spinner("正在处理文档..."):
                # 保存临时文件
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                try:
                    # 分块 + 向量化 + 存入数据库
                    docs = load_and_split(tmp_path)
                    if st.session_state.vectorstore is None:
                        st.session_state.vectorstore = create_vectorstore(docs)
                    else:
                        # 已有数据库，追加新文档
                        st.session_state.vectorstore.add_documents(docs)

                    st.success(f"✅ 已加载 {len(docs)} 个文档块")
                    st.session_state.uploaded_files.append(uploaded_file.name)
                except Exception as e:
                    st.error(f"❌ 加载失败: {e}")
                finally:
                    os.unlink(tmp_path)  # 删除临时文件

    # 显示已上传文件列表
    if st.session_state.uploaded_files:
        st.markdown("---")
        st.subheader("已加载文档")
        for f in st.session_state.uploaded_files:
            st.text(f"📄 {f}")

    # 初始化Agent按钮
    st.markdown("---")
    if st.button("🚀 启动Agent"):
        with st.spinner("正在初始化Agent..."):
            if st.session_state.vectorstore is None:
                try:
                    st.session_state.vectorstore = load_vectorstore()
                except Exception:
                    st.error("请先上传文档或确保知识库已有数据")
                    st.stop()

            st.session_state.agent = build_agent(st.session_state.vectorstore)
            st.success("✅ Agent已就绪！可以开始提问了")

    # 显示Agent能力说明
    st.markdown("---")
    st.subheader("Agent能力")
    st.info(
        "🔧 **知识库检索**：从已上传的文档中查找答案\n\n"
        "🌐 **网页搜索**：知识库没有的信息自动从网上搜索\n\n"
        "💬 **多轮对话**：支持上下文连续追问"
    )


# ============ 主界面：聊天区域 ============

# 显示对话历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📎 来源文档"):
                for src in message["sources"]:
                    st.text(src)

# 输入框
if prompt := st.chat_input("输入你的问题..."):
    # 检查Agent是否已启动
    if st.session_state.agent is None:
        st.warning("请先点击左侧「启动Agent」按钮")
        st.stop()

    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Agent回答
    with st.chat_message("assistant"):
        with st.spinner("Agent正在思考..."):
            try:
                result = st.session_state.agent.invoke({"input": prompt})
                answer = result.get("output", "未获取到回答")

                # 显示回答
                st.markdown(answer)

                # 检查是否有来源信息
                sources = []
                if "source_documents" in result:
                    for doc in result["source_documents"]:
                        src = doc.metadata.get("source", "未知来源")
                        if src not in sources:
                            sources.append(src)

                if sources:
                    with st.expander("📎 来源文档"):
                        for src in sources:
                            st.text(src)

                # 保存到对话历史
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                error_msg = f"出错了: {e}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": []
                })
