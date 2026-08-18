from tw3k_ai_assistant.config import load_environment


load_environment()

from openai import OpenAI

REWRITE_INSTRUCTIONS = """
Rewrite the user's query into one clear, standalone search query for a
Total War: Three Kingdoms knowledge base.

Rules:
- Preserve character names, faction names, mechanics, numbers, and terminology.
- Resolve obvious shorthand.
- Do not answer the question.
- Return only the rewritten query.
- If the query is already clear, return it unchanged.
"""

class QueryRewriter:
    def __init__(self, client: OpenAI, model: str = "gpt-5.4-mini"):
        self.client = client
        self.model = model

    def rewrite(self, query: str) -> str:
        if not query.strip():
            return query

        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "developer", "content": REWRITE_INSTRUCTIONS},
                {
                    "role": "user",
                    "content": f"<user_query>\n{query}\n</user_query>",
                },
            ],
        )

        return response.output_text.strip() or query
