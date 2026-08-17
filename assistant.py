import sys

from dotenv import load_dotenv
from rag_app import RAGBase
from openai import OpenAI

from metrics import RAGWithMetrics

def create_assistant(*, rerank: bool = True, use_query_rewriter: bool = True):
    load_dotenv()

    client = OpenAI()
    return RAGWithMetrics.from_src(
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
    print(answer)
