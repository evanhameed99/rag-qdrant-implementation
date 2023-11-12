from qdrantClient import QdrantClientWrapper
from embeddingModel import EmbeddingModelWrapper
import json
from qdrant_client.http.models import Distance, VectorParams
import os
import openai
from dotenv import load_dotenv


load_dotenv()


OPEN_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPEN_API_KEY

qdrant_client = QdrantClientWrapper()
embedding_model = EmbeddingModelWrapper()

# with open("data.json", "r") as json_file:
#     # Load the JSON data from the file
#     data = json.load(json_file)

# qdrant_client.recreate_collection(
#     "test_collection", VectorParams(size=384, distance=Distance.COSINE)
# )


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run():
    prompt = input("How can I help today? ")

    input_vector = embedding_model.encode(prompt)
    search_result = qdrant_client.client.search(
        collection_name="test_collection",
        query_vector=input_vector,
        limit=10,
        score_threshold=0.4,
    )
    for result in search_result:
        print("\n ####")
        print("SCORE => ", result.score)
        print("RESULT => ", result.payload)
        print("\n ####")
    context = ""
    if search_result:
        context = "\n".join(r.payload["abstract"] for r in search_result)
    print("CONTEXT =>", context)

    metaprompt = "Question: {}\n\n{}Answer:".format(
        prompt.strip(),
        "Context:\n{}".format(context.strip()) if len(context.strip()) > 0 else "",
    )
    print("metaprompt =>", metaprompt)

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": metaprompt},
        ],
    )

    print("LLM response =>", response.choices[0].message.content)


# qdrant_client.upsert_data("test_collection", data)
run()
