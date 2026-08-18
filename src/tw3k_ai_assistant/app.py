from __future__ import annotations

import streamlit as st

from tw3k_ai_assistant.assistant import create_assistant
from tw3k_ai_assistant.database.feedback import save_feedback
from tw3k_ai_assistant.database.saves import save_conversation
from tw3k_ai_assistant.ui.dashboard import render_dashboard


st.set_page_config(
    page_title="TW3K Assistant",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading retrieval models...")
def get_assistant():
    """Create and reuse the assistant with reranking always enabled."""

    return create_assistant(rerank=True)


def initialize_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "feedback_by_conversation" not in st.session_state:
        st.session_state.feedback_by_conversation = {}


def render_response_details(message: dict) -> None:
    record = message["record"]
    with st.expander("Response details"):
        metrics = st.columns(4)
        metrics[0].metric("Response time", f"{record.response_time:.2f}s")
        metrics[1].metric("Prompt tokens", f"{record.prompt_tokens:,}")
        metrics[2].metric("Completion tokens", f"{record.completion_tokens:,}")
        metrics[3].metric("Cost", f"${record.cost:.6f}")


def render_feedback(conversation_id: int) -> None:
    feedback = st.session_state.feedback_by_conversation.get(conversation_id)
    if feedback is None:
        st.caption("Was this answer helpful?")
        helpful, not_helpful = st.columns(2)

        with helpful:
            if st.button("Helpful", key=f"feedback_up_{conversation_id}", width="stretch"):
                save_feedback(conversation_id, 1)
                st.session_state.feedback_by_conversation[conversation_id] = 1
                st.rerun()

        with not_helpful:
            if st.button("Not helpful", key=f"feedback_down_{conversation_id}", width="stretch"):
                save_feedback(conversation_id, -1)
                st.session_state.feedback_by_conversation[conversation_id] = -1
                st.rerun()
    else:
        st.caption("Thanks for your feedback.")


def render_message(message: dict, *, show_feedback: bool = False) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and not message.get("error"):
            render_response_details(message)
            if show_feedback:
                render_feedback(message["conversation_id"])


def render_chat() -> None:
    st.title("TW3K Assistant")
    st.caption("Ask about Total War: Three Kingdoms strategy, mechanics, and campaigns.")

    for index, message in enumerate(st.session_state.messages):
        render_message(
            message,
            show_feedback=(
                message["role"] == "assistant"
                and not message.get("error")
                and index == len(st.session_state.messages) - 1
            ),
        )

    question = st.chat_input("Ask a question...")
    if not question:
        return

    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    try:
        assistant = get_assistant()
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                streamed_answer = st.write_stream(assistant.rag_stream(question))

            record = assistant.last_call
            answer = (
                streamed_answer
                if isinstance(streamed_answer, str)
                else "".join(str(part) for part in streamed_answer)
            )
            answer = answer or record.answer
            conversation_id = save_conversation(record, question)
            assistant_message = {
                "role": "assistant",
                "content": answer,
                "record": record,
                "conversation_id": conversation_id,
            }
            render_response_details(assistant_message)
            render_feedback(conversation_id)

        st.session_state.messages.append(assistant_message)
    except Exception:
        error_message = {
            "role": "assistant",
            "content": "I couldn't process that request. Please try again.",
            "error": True,
        }
        st.session_state.messages.append(error_message)
        render_message(error_message)
        st.error("The request could not be completed.")


initialize_state()

# Warm the cached retrieval pipeline when the Streamlit application starts so
# the first submitted question does not pay the model-loading cost.
get_assistant()

with st.sidebar:
    st.header("TW3K Assistant")
    interface = st.selectbox(
        "Interface",
        options=("Chat", "Metrics Dashboard"),
        index=0,
        key="interface_view",
    )

    if interface == "Chat":
        st.divider()
        if st.button("Clear conversation", width="stretch"):
            st.session_state.messages = []
            st.session_state.feedback_by_conversation = {}
            st.rerun()

if interface == "Chat":
    render_chat()
else:
    render_dashboard()
