import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HUGGINGFACE_API_KEY")
)
MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def generate_answer(question, context):
    prompt = f"""
You are a helpful document question-answering assistant.

Answer the user's question using ONLY the information provided
in the context below.

If the context contains information that helps answer the question,
use that information to give the best answer possible.

Only say:
"I couldn't find the answer in the provided document."
if the context contains no relevant information at all.


Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=300
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    answer = generate_answer(
        "What is RAG?",
        """
        Retrieval Augmented Generation combines information retrieval
        with a language model. The system first retrieves relevant
        information and then uses that information to generate a
        grounded answer.
        """
    )

    print("\nAnswer:")
    print(answer)