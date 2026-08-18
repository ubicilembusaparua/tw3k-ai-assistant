import sys

from openai import OpenAI

from tw3k_ai_assistant.config import load_environment
from tw3k_ai_assistant.database.saves import save_conversation
from tw3k_ai_assistant.rag.metrics import RAGWithMetrics

def create_assistant(*, rerank: bool = True, use_query_rewriter: bool = True):
    load_environment()

    client = OpenAI()
    return RAGWithMetrics(
        rerank=rerank,
        llm_client=client,
        use_query_rewriter=use_query_rewriter,
    )

if __name__ == "__main__":
    assistant = create_assistant()

    query = "How to manage food?"
    if len(sys.argv) > 1:
        query = sys.argv[1]

    answer = assistant.rag(query)
    save_conversation(assistant.last_call, query)
    # print(answer)
