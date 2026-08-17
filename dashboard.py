"""Reusable Streamlit view for the captured TW3K assistant LLM metrics."""

from __future__ import annotations

from datetime import date

import streamlit as st

from dashboard_utils import (
    daily_metrics,
    filter_records,
    model_metrics,
    recent_rows,
    summarize_records,
)
from db_query import ConversationMetric, get_conversation_metrics


@st.cache_data(ttl=30, show_spinner=False)
def load_metrics(limit: int = 5000) -> list[ConversationMetric]:
    """Load recent metrics while avoiding a database query on every rerun."""

    return get_conversation_metrics(limit=limit)


def render_dashboard() -> None:
    """Render the metrics dashboard inside the current Streamlit page."""

    st.title("TW3K Assistant Dashboard")
    st.caption("Operational and user-feedback metrics captured from the conversation log.")

    with st.sidebar:
        st.header("Dashboard filters")
        if st.button("Refresh data", key="dashboard_refresh", width="stretch"):
            load_metrics.clear()
            st.rerun()

    try:
        records = load_metrics()
    except Exception as exc:
        st.error("Could not load metrics from PostgreSQL.")
        st.info(
            "Check that PostgreSQL is running, the conversations and feedback tables "
            "are initialized, and POSTGRES_* environment variables are configured."
        )
        with st.expander("Technical details"):
            st.exception(exc)
        return

    if not records:
        st.info("No captured conversations yet. Ask a question in the assistant to populate this dashboard.")
        return

    available_models = sorted({record.model for record in records})
    selected_models = st.sidebar.multiselect(
        "Models",
        options=available_models,
        default=available_models,
    )

    available_dates = [record.timestamp.date() for record in records]
    min_date = min(available_dates)
    max_date = max(available_dates)
    selected_dates = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date = end_date = selected_dates

    if not isinstance(start_date, date) or not isinstance(end_date, date):
        st.warning("Select a complete date range to view metrics.")
        return

    filtered = filter_records(
        records,
        models=selected_models,
        date_range=(start_date, end_date),
    )

    if not filtered:
        st.warning("No conversations match the selected filters.")
        return

    summary = summarize_records(filtered)

    headline = st.columns(4)
    headline[0].metric("Requests", f"{summary.total_requests:,}")
    headline[1].metric("Avg response time", f"{summary.avg_response_time:.2f}s")
    headline[2].metric("Total cost", f"${summary.total_cost:.4f}")
    headline[3].metric("Avg total tokens", f"{summary.avg_total_tokens:,.0f}")

    details = st.columns(4)
    details[0].metric("Avg prompt tokens", f"{summary.avg_prompt_tokens:,.0f}")
    details[1].metric("Avg completion tokens", f"{summary.avg_completion_tokens:,.0f}")
    details[2].metric("Cost per request", f"${summary.total_cost / summary.total_requests:.6f}")
    feedback_rate = "-" if summary.positive_feedback_rate is None else f"{summary.positive_feedback_rate:.1f}%"
    details[3].metric(
        "Positive user feedback",
        feedback_rate,
        help=f"Based on {summary.rated_feedback} rated requests.",
    )

    st.divider()

    daily = daily_metrics(filtered)
    st.subheader("Usage over time")
    usage_left, usage_right = st.columns(2)
    with usage_left:
        st.caption("Requests per day")
        st.line_chart(daily, x="date", y="requests")
    with usage_right:
        st.caption("Cost per day")
        st.line_chart(daily, x="date", y="cost")

    latency_left, latency_right = st.columns(2)
    with latency_left:
        st.caption("Average response time per day")
        st.line_chart(daily, x="date", y="avg_response_time")
    with latency_right:
        st.caption("Average tokens per day")
        st.line_chart(daily, x="date", y="avg_tokens")

    st.subheader("Model usage")
    model_rows = model_metrics(filtered)
    model_left, model_right = st.columns(2)
    with model_left:
        st.bar_chart(model_rows, x="model", y="requests")
    with model_right:
        st.bar_chart(model_rows, x="model", y="cost")
    st.dataframe(model_rows, hide_index=True, width="stretch")

    st.subheader("Recent requests")
    st.dataframe(
        recent_rows(filtered[:100]),
        hide_index=True,
        width="stretch",
        column_config={
            "response_time_s": st.column_config.NumberColumn("Response time (s)", format="%.3f"),
            "cost_usd": st.column_config.NumberColumn("Cost (USD)", format="$%.6f"),
            "user_score": st.column_config.NumberColumn("User score", format="%d"),
        },
    )

    st.caption(f"Showing {len(filtered):,} matching requests from the {len(records):,} most recent captured records.")

    st.subheader("Request details")
    for record in filtered[:20]:
        timestamp = record.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        question = record.question.replace("\n", " ").strip()
        label = f"{timestamp} - {record.model} - {question[:100]}"
        with st.expander(label):
            st.markdown("**Question**")
            st.write(record.question)
            st.markdown("**Answer**")
            st.write(record.answer)

            metadata = st.columns(5)
            metadata[0].metric("Latency", f"{record.response_time:.2f}s")
            metadata[1].metric("Prompt tokens", f"{record.prompt_tokens:,}")
            metadata[2].metric("Completion tokens", f"{record.completion_tokens:,}")
            metadata[3].metric("Cost", f"${record.cost:.6f}")
            metadata[4].metric("User score", "-" if record.user_score is None else str(record.user_score))


def main() -> None:
    """Run the dashboard as a standalone Streamlit page."""

    st.set_page_config(
        page_title="TW3K LLM Metrics",
        layout="wide",
    )
    render_dashboard()


if __name__ == "__main__":
    main()
