from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer
import json
import os
import uvicorn

app = FastAPI()

# Connect to Milvus
connections.connect(alias="default", host="localhost", port="19530")

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Pydantic model for query input
class QueryInput(BaseModel):
    query: str

# Define schema
fields = [
    FieldSchema(name="_id", dtype=DataType.VARCHAR, max_length=100, is_primary=True, auto_id=False),
    FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=1024),
    FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=2048),
    FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="updated_at", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=384)
]
schema = CollectionSchema(fields, description="FAQ Vector Collection")

collection_name = "faq_collection"

@app.get("/bookings")
def show_bookings():
    try:
        with open("../data/Booking_Data.json") as file:
            data = json.load(file)
        return {"status":"success", "message": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/create_faq_db")
def create_booking_rag():
    try:
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)

        with open("../data/FAQ.json") as file:
            data = json.load(file)

        collection = Collection(name=collection_name, schema=schema)

        questions = [item["question"] for item in data]
        embeddings = embedding_model.encode(questions).tolist()

        data_to_insert = [
            [item["_id"] for item in data],
            [item["question"] for item in data],
            [item["answer"] for item in data],
            [item["created_at"] for item in data],
            [item["updated_at"] for item in data],
            embeddings
        ]

        collection.insert(data_to_insert)
        collection.create_index(field_name="embeddings", index_params={
            "index_type": "IVF_FLAT",
            "metric_type": "IP",
            "params": {"nlist": 128}
        })
        collection.load()

        return {"status": "success", "message": "FAQ vector DB created and loaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query_faq")
def query_booking_rag(data: QueryInput):
    try:
        if not utility.has_collection(collection_name):
            raise HTTPException(status_code=400, detail="Collection not found. Please initialize it first.")

        collection = Collection(collection_name)
        collection.load()

        query_vector = embedding_model.encode([data.query]).tolist()

        results = collection.search(
            data=query_vector,
            anns_field="embeddings",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=3,
            output_fields=["_id", "question", "answer"]
        )

        top_matches = []
        for hit in results[0]:
            # top_matches.append({"score": hit.score,"question": hit.entity.question,"answer": hit.entity.answer})
            top_matches.append(hit.entity.answer)

        return {"matches": top_matches}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__=="__main__":
    uvicorn.run("rag_implementations:app", host="127.0.0.1", port=8040, reload=True)

