import streamlit as st
from assistant import create_assistant
from db_save import save_conversation
from db_feedback import save_feedback
from judge import evaluate_relevance
from db_feedback import save_feedback

@st.cache_resource(show_spinner="Loading retrieval models...")
def get_assistant(use_reranker: bool):
    """Create one assistant per reranking configuration and reuse it on reruns."""
    return create_assistant(rerank=use_reranker)


st.sidebar.header("Retrieval settings")
use_reranker = st.sidebar.checkbox(
    "Use cross-encoder reranker",
    value=True,
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

        conversation_id = save_conversation(record, user_input)
        st.session_state.conversation_id = conversation_id

        relevance, explanation = evaluate_relevance(user_input, answer)
        save_feedback(conversation_id, "judge",
                        relevance=relevance, explanation=explanation)
        st.write(f"Relevance: {relevance}")
        st.write(f"Explanation: {explanation}")

        if conversation_id is not None:
            col1, col2 = st.columns(2)

            with col1:
                if st.button("+1", key=f"feedback_up_{conversation_id}"):
                    save_feedback(conversation_id, "user", score=1)
                    st.success("Thanks!")

            with col2:
                if st.button("-1", key=f"feedback_down_{conversation_id}"):
                    save_feedback(conversation_id, "user", score=-1)
                    st.success("Thanks for the feedback!")
