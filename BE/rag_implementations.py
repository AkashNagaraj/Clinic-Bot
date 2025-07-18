from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from sentence_transformers import SentenceTransformer
import json

# Connect to Milvus
connections.connect(alias="default", host="localhost", port="19530")

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Global schema so it can be reused
fields = [
    FieldSchema(name="_id", dtype=DataType.VARCHAR, max_length=100, is_primary=True, auto_id=False),
    FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=1024),
    FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=2048),
    FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="updated_at", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=384)
]
schema = CollectionSchema(fields, description="FAQ Vector Collection")

def create_booking_rag():
    with open("../data/FAQ.json") as file:
        data = json.load(file)

    collection_name = "faq_collection"

    collection = Collection(name=collection_name, schema=schema)

    # Extract questions and embed
    questions = [item["question"] for item in data]
    embeddings = embedding_model.encode(questions).tolist()

    # Prepare data for insertion
    data_to_insert = [
        [item["_id"] for item in data],
        [item["question"] for item in data],
        [item["answer"] for item in data],
        [item["created_at"] for item in data],
        [item["updated_at"] for item in data],
        embeddings
    ]

    # Insert and index
    collection.insert(data_to_insert)
    collection.create_index(field_name="embeddings", index_params={
        "index_type": "IVF_FLAT",
        "metric_type": "IP",
        "params": {"nlist": 128}
    })
    collection.load()

    print("FAQ vector DB created and loaded successfully.")

def query_booking_rag(query):
    collection = Collection("faq_collection")
    query_vector = embedding_model.encode([query]).tolist()

    results = collection.search(
        data=query_vector,
        anns_field="embeddings",
        param={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=3,
        output_fields=["_id", "question", "answer"]
    )

    top_matches = []
    for hit in results[0]:
        #print(f"Score: {hit.score:.4f} | Q: {hit.entity.question} | A: {hit.entity.answer}")
        top_matches.append(f"A : {hit.entity.answer}")
    return top_matches


def main_rag_im():
    #create_booking_rag()
    query = "How do I book a dentist appointment?"
    query_booking_rag(query)

if __name__ == "__main__":
    main()
