from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.models import InputData
from app.db import get_db_connection

from app.crypto import router as crypto_router
from app.search import router as search_router

from app.sentiment import improved_sentiment_analysis
from app.problem_logging import router as problem_logging_router

import app.ml_service as ml_service

app = FastAPI()

# --- FastAPI Lifespan Events ---
# This context manager will run code on startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # On Startup:
    print("Application startup: Loading ML model...")
    ml_service.load_ml_model() # Load the model synchronously

    # You could also trigger an async training here if desired, but better as a cron
    # asyncio.create_task(ml_service.train_model_async()) # Example: train in background

    yield # Application runs

    # On Shutdown:
    print("Application shutdown.")


app = FastAPI(lifespan=lifespan)

app.include_router(crypto_router)
app.include_router(search_router)
app.include_router(problem_logging_router)

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
