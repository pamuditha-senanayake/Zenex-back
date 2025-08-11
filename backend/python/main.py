from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.models import InputData
from app.db import get_db_connection
from app.sentiment import improved_sentiment_analysis

from app.crypto import router as crypto_router

from app.search import router as search_router



app = FastAPI()

app.include_router(crypto_router)
app.include_router(search_router)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # adjust your frontend URL here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze(input_data: InputData):
    sentiment = improved_sentiment_analysis(input_data.text)

    conn = await get_db_connection()
    await conn.execute(
        "INSERT INTO user_inputs(text, sentiment) VALUES($1, $2)",
        input_data.text,
        sentiment
    )
    await conn.close()

    return {"text": input_data.text, "sentiment": sentiment}
