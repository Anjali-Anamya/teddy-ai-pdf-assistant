
import streamlit as st
from dotenv import load_dotenv

from document_processing.pdf_loader import process_pdf
from vector_store.db import create_vectorstore
from rag.retriever import create_retriever
from rag.chain import create_rag_chain


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Teddy AI | PDF Assistant",
    page_icon="🧸",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()


# --------------------------------------------------
# CUSTOM TEDDY BEAR THEME
# --------------------------------------------------

st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(
        135deg,
        #fff5f7 0%,
        #f4edff 50%,
        #eaf6ff 100%
    );
    color: #49394f;
}

/* Main content */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1150px;
}

/* Main headings */
h1 {
    color: #8e5b91 !important;
    font-weight: 800 !important;
}

h2, h3 {
    color: #9b6b9e !important;
}

/* General text */
.stApp p,
.stApp label,
.stApp li {
    color: #49394f !important;
    font-size: 16px;
    line-height: 1.7;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #fce4ec,
        #eee4ff,
        #e3f2fd
    );
    border-right: 2px solid #e8cce9;
}

/* Sidebar headings */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #87558d !important;
}

/* Sidebar text */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: #59445f !important;
}

/* Teddy header */
.teddy-header {
    background: linear-gradient(
        120deg,
        #f8dce8,
        #e9ddff,
        #dff2ff
    );
    padding: 28px;
    border-radius: 25px;
    text-align: center;
    border: 2px solid #edcce7;
    box-shadow: 0 5px 18px rgba(150, 100, 160, 0.12);
    margin-bottom: 25px;
}

.teddy-header h1 {
    color: #89558d !important;
    font-size: 38px !important;
    margin-bottom: 5px;
}

.teddy-header p {
    color: #725779 !important;
    font-size: 17px !important;
}

/* Information cards */
.teddy-card {
    background: rgba(255, 255, 255, 0.85);
    padding: 18px;
    border-radius: 18px;
    border: 1px solid #ead6ef;
    box-shadow: 0 3px 12px rgba(130, 90, 140, 0.08);
    margin-bottom: 15px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(
        100deg,
        #d99acb,
        #b49ae8
    );
    color: white !important;
    border: none;
    border-radius: 14px;
    font-weight: 700;
    padding: 10px 18px;
    transition: 0.2s;
}

.stButton > button:hover {
    background: linear-gradient(
        100deg,
        #c77fba,
        #987bd5
    );
    color: white !important;
    border: 1px solid #ffffff;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.88);
    border: 1px solid #ead6ef;
    border-radius: 18px;
    padding: 16px;
    box-shadow: 0 3px 10px rgba(120, 80, 130, 0.07);
    margin-bottom: 12px;
}

/* Chat message text */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {
    color: #49394f !important;
    font-size: 16px !important;
    line-height: 1.7 !important;
}

/* Chat input */
[data-testid="stChatInput"] {
    background: #ffffff !important;
    border: 2px solid #dfb9df !important;
    border-radius: 18px !important;
}

[data-testid="stChatInput"] textarea {
    color: #49394f !important;
    background: #ffffff !important;
    caret-color: #89558d !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #a17da4 !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background: #fffaff;
    border: 1px solid #e8cde9;
    border-radius: 14px;
}

[data-testid="stExpander"] p,
[data-testid="stExpander"] span {
    color: #59445f !important;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 15px;
}

/* Horizontal line */
hr {
    border-color: #e8cde9 !important;
}

/* Footer */
.teddy-footer {
    text-align: center;
    color: #987ba0;
    font-size: 14px;
    padding: 25px 0 5px 0;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-thumb {
    background: #d3a9d7;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None

if "rag_initialized" not in st.session_state:
    st.session_state.rag_initialized = False


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="teddy-header">
    <h1>🧸 Teddy AI</h1>
    <p>Your cute little AI companion for understanding PDFs!</p>
    <p>📚 Upload knowledge, ask questions and learn smarter.</p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("# 🧸 Teddy's Corner")

    st.markdown("---")

    st.markdown("### 📖 About this app")

    st.write(
        "Teddy AI is an intelligent PDF question-answering "
        "assistant powered by Gemini, LangChain and ChromaDB."
    )

    st.markdown("---")

    st.markdown("### ⚙️ RAG System")

    if st.button(
        "✨ Initialize RAG System",
        use_container_width=True
    ):

        try:
            with st.spinner(
                "🧸 Teddy is preparing your knowledge base..."
            ):

                # Load and split PDF
                chunks = process_pdf()

                if not chunks:
                    raise ValueError(
                        "No text could be extracted from the PDF. "
                        "Please check your PDF file."
                    )

                # Create or load ChromaDB
                vectorstore = create_vectorstore(chunks)

                # Create retriever
                retriever = create_retriever(vectorstore)

                # Create RAG pipeline
                rag_pipeline = create_rag_chain(retriever)

                # Store pipeline in session state
                st.session_state.rag_pipeline = rag_pipeline
                st.session_state.rag_initialized = True

            st.success("🎉 Teddy is ready to answer your questions!")

        except Exception as e:
            st.session_state.rag_initialized = False
            st.session_state.rag_pipeline = None

            st.error("Unable to initialize the RAG system.")
            st.exception(e)

    st.markdown("---")

    # Initialization status
    st.markdown("### 📡 System Status")

    if st.session_state.rag_initialized:
        st.success("🟢 RAG System Ready")
    else:
        st.warning("🟡 RAG System Not Initialized")

    st.markdown("---")

    # Clear chat
    st.markdown("### 🧹 Chat Controls")

    if st.button(
        "Clear Chat History",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    st.markdown("""
    <div style="text-align:center;">
        <div style="font-size:55px;">🧸</div>
        <p style="color:#89558d;">
            Made with love and AI
        </p>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# MAIN CHAT INTERFACE
# --------------------------------------------------

st.markdown("## 💬 Chat with your PDF")

st.markdown(
    "Ask Teddy anything about the information in your document!"
)

# Show initial welcome message
if not st.session_state.messages:

    with st.chat_message("assistant", avatar="🧸"):
        st.markdown(
            "Hello! I'm Teddy AI. "
            "Initialize the RAG system from the sidebar "
            "and I'll help you understand your PDF. "
            "Ask me anything!"
        )


# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------

for message in st.session_state.messages:

    avatar = "🧸" if message["role"] == "assistant" else "🌸"

    with st.chat_message(message["role"], avatar=avatar):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            sources = message.get("sources", [])

            if sources:

                with st.expander("📚 View Sources"):

                    for source in sources:

                        st.markdown(
                            f"- **File:** {source['source']}  \n"
                            f"  **Page:** {source['page']}"
                        )


# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

question = st.chat_input(
    "🌷 Ask Teddy a question about your PDF..."
)


if question:

    # Check if RAG is initialized
    if not st.session_state.rag_initialized:

        st.warning(
            "Please initialize the RAG system from the sidebar first!"
        )

    else:

        # Display user question
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user", avatar="🌸"):
            st.markdown(question)

        # Generate answer
        with st.chat_message("assistant", avatar="🧸"):

            with st.spinner("🧸 Teddy is thinking..."):

                try:

                    result = st.session_state.rag_pipeline(question)

                    answer = result.get(
                        "answer",
                        "Sorry, I couldn't generate an answer."
                    )

                    sources = result.get("sources", [])

                    # Ensure answer is a string
                    if isinstance(answer, list):

                        answer = "\n".join(
                            block.get("text", "")
                            for block in answer
                            if isinstance(block, dict)
                            and block.get("type") == "text"
                        )

                    if not isinstance(answer, str):
                        answer = str(answer)

                    st.markdown(answer)

                    # Display sources
                    if sources:

                        with st.expander("📚 View Sources"):

                            for source in sources:

                                st.markdown(
                                    f"- **File:** {source['source']}  \n"
                                    f"  **Page:** {source['page']}"
                                )

                    # Save assistant response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:

                    error_message = (
                        "Sorry! Teddy encountered an error "
                        "while answering your question."
                    )

                    st.error(error_message)
                    st.exception(e)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="teddy-footer">
    🧸 Teddy AI | Intelligent PDF Assistant
    <br>
    Powered by LangChain • Gemini • ChromaDB • Streamlit
    <br><br>
    🌷 Keep learning, keep growing! 🌷
</div>
""", unsafe_allow_html=True)
