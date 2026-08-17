import streamlit as st
from assistant import create_assistant


@st.cache_resource(show_spinner="Loading retrieval models...")
def get_assistant(use_reranker: bool):
    """Create one assistant per reranking configuration and reuse it on reruns."""
    return create_assistant(rerank=use_reranker)


st.sidebar.header("Retrieval settings")
use_reranker = st.sidebar.checkbox(
    "Use cross-encoder reranker",
    value=False,
    help="Improves result ordering but loads an additional model.",
)
assistant = get_assistant(use_reranker)

st.title("TW3K AI Assistant")

user_input = st.text_input("Enter your question:")

if st.button("Ask"):
    with st.spinner("Processing..."):
        answer = assistant.rag(user_input)
        st.success("Completed!")
        st.write(answer)

        record = assistant.last_call
        st.write(f"Response time: {record.response_time:.2f}s")
        st.write(f"Prompt tokens: {record.prompt_tokens}")
        st.write(f"Completion tokens: {record.completion_tokens}")
        st.write(f"Cost: ${record.cost:.4f}")
