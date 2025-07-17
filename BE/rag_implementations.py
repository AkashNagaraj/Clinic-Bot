from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from sentence_transformers import SentenceTransformer
import datetime, json

connections.connect(alias="default", host="localhost", port="19530")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def create_booking_rag():
    with open("../data/Booking_Data.json") as file:
        data = json.load(file)

    fields = [
    FieldSchema(name="_id", dtype=DataType.VARCHAR, max_length=100, is_primary=True, auto_id=False),
    FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=1024),
    FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=2048),
    FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="updated_at", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=384)
    ]
    
    schema = CollectionSchema(fields, description = "FAQ Vector Collection")
    collection_name = "faq_collection"
    collection = Collection(name=collection_name, schema=schema)
    questions = [val["question"] for val in data]
    embeddings = embedding_model.encode(questions).tolist()

    data_to_insert = [
    [item["_id"] for item in faq_data],               # _id
    [item["question"] for item in faq_data],          # question
    [item["answer"] for item in faq_data],            # answer
    [item["created_at"] for item in faq_data],        # created_at
    [item["updated_at"] for item in faq_data],        # updated_at
    embeddings                                        # embeddings
    ]
    collection.insert(data_to_insert)
    collection.create_index(field_name="embeddings", index_params={
    "index_type": "IVF_FLAT",
    "metric_type": "COSINE",
    "params": {"nlist": 128}
    })
    collection.load()

    print("Vector DB created and loaded successfully.")

def query_booking_rag(query):
    collection_name = "faq_collection"
    collection = Collection(name=collection_name, schema=schema)
    query_vector = embedding_model.encode([query]).tolist()
    results = collection.search(
        data = query_vector, anns_field = "embeddings", 
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=3,
        output_fields=["_id", "question", "answer"]
        )

    for hit in results[0]:
        print(f"Score: {hit.score:.4f}, Question: {hit.entity.question}, Answer: {hit.entity.answer}")

def main():
    query = "8pm slot"
    create_booking_rag()
    query_booking_rag(query)

if __name__=="__main__":
    main()