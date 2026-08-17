from __future__ import annotations

import streamlit as st

from assistant import create_assistant
from dashboard import render_dashboard
from db_feedback import save_feedback
from db_save import save_conversation
from judge import evaluate_relevance


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

        st.caption(f"Judge relevance: {message['relevance']}")
        if message["explanation"]:
            st.write(message["explanation"])


def render_feedback(conversation_id: int) -> None:
    feedback = st.session_state.feedback_by_conversation.get(conversation_id)
    if feedback is None:
        st.caption("Was this answer helpful?")
        helpful, not_helpful = st.columns(2)

        with helpful:
            if st.button("Helpful", key=f"feedback_up_{conversation_id}", width="stretch"):
                save_feedback(conversation_id, "user", score=1)
                st.session_state.feedback_by_conversation[conversation_id] = 1
                st.rerun()

        with not_helpful:
            if st.button("Not helpful", key=f"feedback_down_{conversation_id}", width="stretch"):
                save_feedback(conversation_id, "user", score=-1)
                st.session_state.feedback_by_conversation[conversation_id] = -1
                st.rerun()
    else:
        st.caption("Thanks for your feedback.")


def render_chat() -> None:
    st.title("TW3K Assistant")
    st.caption("Ask about Total War: Three Kingdoms strategy, mechanics, and campaigns.")

    question = st.chat_input("Ask a question...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        try:
            with st.spinner("Thinking..."):
                assistant = get_assistant()
                answer = assistant.rag(question)
                record = assistant.last_call
                conversation_id = save_conversation(record, question)
                relevance, explanation = evaluate_relevance(question, answer)
                save_feedback(
                    conversation_id,
                    "judge",
                    relevance=relevance,
                    explanation=explanation,
                )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "record": record,
                    "conversation_id": conversation_id,
                    "relevance": relevance,
                    "explanation": explanation,
                }
            )
        except Exception:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "I couldn't process that request. Please try again.",
                    "error": True,
                }
            )
            st.error("The request could not be completed.")

    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and not message.get("error"):
                render_response_details(message)

                is_latest_reply = index == len(st.session_state.messages) - 1
                if is_latest_reply:
                    render_feedback(message["conversation_id"])


initialize_state()

with st.sidebar:
    st.header("TW3K Assistant")
    interface = st.radio(
        "Interface",
        options=("Chat", "LLM metrics"),
        index=0,
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
