from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, CollectionStatus, UpdateStatus
from embeddingModel import EmbeddingModelWrapper
from dotenv import load_dotenv
import uuid
import os

load_dotenv()


class QdrantClientWrapper:
    def __init__(
        self,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    ):
        self.client = QdrantClient(url, api_key=api_key)

    def recreate_collection(self, collection_name, vectors_config):
        self.client.recreate_collection(
            collection_name=collection_name, vectors_config=vectors_config
        )

    def upsert_data(self, collection_name, data):
        embedding_model = EmbeddingModelWrapper()
        points = []
        for item in data:
            abstract = item.get("abstract")
            abstract_vector = embedding_model.encode(abstract)
            text_id = str(uuid.uuid4())
            payload = {"abstract": abstract}
            point = PointStruct(id=text_id, vector=abstract_vector, payload=payload)
            points.append(point)

        operation_info = self.client.upsert(
            collection_name=collection_name, wait=True, points=points
        )

        if operation_info.status == UpdateStatus.COMPLETED:
            print("Data inserted successfully!")
        else:
            print("Failed to insert data")

    # Define other QdrantClient methods here
