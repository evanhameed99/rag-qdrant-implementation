from sentence_transformers import SentenceTransformer


class EmbeddingModelWrapper:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, input_text):
        return self.model.encode(input_text)

    # Define other embedding model methods here
