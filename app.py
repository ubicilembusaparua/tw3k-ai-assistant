import streamlit as st

from assistant import create_assistant
from db_feedback import save_feedback
from db_save import save_conversation
from judge import evaluate_relevance


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
    if not user_input.strip():
        st.warning("Enter a question first.")
    else:
        with st.spinner("Processing..."):
            answer = assistant.rag(user_input)
            record = assistant.last_call

            conversation_id = save_conversation(record, user_input)
            relevance, explanation = evaluate_relevance(user_input, answer)
            save_feedback(
                conversation_id,
                "judge",
                relevance=relevance,
                explanation=explanation,
            )

        # Keep the response visible when a feedback button triggers the next
        # Streamlit rerun.  The user vote is reset for each new conversation.
        st.session_state.last_response = {
            "answer": answer,
            "record": record,
            "relevance": relevance,
            "explanation": explanation,
        }
        st.session_state.conversation_id = conversation_id
        st.session_state.user_feedback = None


last_response = st.session_state.get("last_response")
conversation_id = st.session_state.get("conversation_id")

if last_response is not None:
    record = last_response["record"]

    st.success("Completed!")
    st.write(last_response["answer"])
    # st.write(f"Response time: {record.response_time:.2f}s")
    # st.write(f"Prompt tokens: {record.prompt_tokens}")
    # st.write(f"Completion tokens: {record.completion_tokens}")
    # st.write(f"Cost: ${record.cost:.4f}")
    # st.write(f"Relevance: {last_response['relevance']}")
    # st.write(f"Explanation: {last_response['explanation']}")

if conversation_id is not None:
    st.subheader("Was this answer helpful?")
    user_feedback = st.session_state.get("user_feedback")

    if user_feedback is None:
        col1, col2 = st.columns(2)

        with col1:
            if st.button("+1", key=f"feedback_up_{conversation_id}"):
                save_feedback(conversation_id, "user", score=1)
                st.session_state.user_feedback = 1
                st.success("Thanks for the positive feedback!")

        with col2:
            if st.button("-1", key=f"feedback_down_{conversation_id}"):
                save_feedback(conversation_id, "user", score=-1)
                st.session_state.user_feedback = -1
                st.success("Thanks for the feedback!")
    elif user_feedback == 1:
        st.success("Thanks for the positive feedback!")
    else:
        st.success("Thanks for the feedback!")
