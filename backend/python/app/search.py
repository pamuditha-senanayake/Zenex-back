from fastapi import APIRouter, HTTPException, Query, Body
import httpx
import os
from typing import List
from pydantic import BaseModel
from app.db import get_db_connection

router = APIRouter()

# --- Pydantic Models ---
class MomentSolution(BaseModel):
    id: int
    moment: str
    solution: str
    has_solution: bool # NEW FIELD: Indicates if a solution is known for this moment

    class Config:
        from_attributes = True # Required for Pydantic v2 to map ORM objects

class MomentSolutionCreate(BaseModel):
    moment: str
    solution: str

class MomentSolutionUpdate(BaseModel):
    moment: str
    solution: str

# --- Gemini API Configuration ---
# The same API key can be used for both generative models and embedding models
GEMINI_API_KEY = os.getenv("API_KEY") # Ensure this environment variable is set

# Define the embedding model and its corresponding API URL
# 'models/text-embedding-004' is the current recommended Gemini embedding model
GEMINI_EMBEDDING_MODEL_NAME = "text-embedding-004"
EMBEDDING_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_EMBEDDING_MODEL_NAME}:embedContent"

# --- Helper function to generate embeddings ---
async def generate_embedding(text: str) -> List[float]:
    """Generates an embedding vector for the given text using the Gemini Embedding API."""
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Gemini API key is not configured. Cannot generate embeddings."
        )

    # Correct Gemini embedding request payload for a single text string
    payload = {
        "model": GEMINI_EMBEDDING_MODEL_NAME,
        "content": {
            "parts": [
                {"text": text}
            ]
        }
    }
    headers = {"Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                EMBEDDING_API_URL,
                params={"key": GEMINI_API_KEY},
                json=payload,
                timeout=10.0 # Shorter timeout for embeddings compared to generative models
            )
        response.raise_for_status() # Raise an exception for bad HTTP status codes (4xx or 5xx)
        embedding_data = response.json()

        # Parse the embedding values from the API response
        # The structure is response["embedding"]["values"]
        embedding = embedding_data.get("embedding", {}).get("values")

        if not embedding:
            # If embedding values are missing, check for an error message from the API
            error_detail = embedding_data.get("error", {}).get("message", "No embedding values found in Gemini API response.")
            raise ValueError(f"Gemini API returned no embedding: {error_detail}")

        return embedding
    except httpx.HTTPStatusError as e:
        print(f"HTTP error from Gemini Embedding API: Status {e.response.status_code} - Detail: {e.response.text}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Error from Gemini Embedding API: {e.response.text}"
        )
    except Exception as e:
        print(f"Failed to generate embedding: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate embedding: {str(e)}. Please check your API key and network."
        )

# --- Moment & Solution Endpoints ---

@router.get("/api/moments", response_model=List[MomentSolution])
async def get_all_moments():
    """Retrieves all stored moments and their solutions, including their solution status."""
    conn = await get_db_connection()
    try:
        # Select the new 'has_solution' column
        rows = await conn.fetch("SELECT id, moment, solution, has_solution FROM moments ORDER BY id DESC LIMIT 100")
    finally:
        await conn.close()
    # Ensure has_solution is passed to the Pydantic model
    return [MomentSolution(id=row['id'], moment=row['moment'], solution=row['solution'], has_solution=row['has_solution']) for row in rows]

@router.delete("/api/moments/{moment_id}", status_code=204)
async def delete_moment(moment_id: int):
    """Deletes a moment and its solution by ID."""
    conn = await get_db_connection()
    try:
        result = await conn.execute("DELETE FROM moments WHERE id=$1", moment_id)
    finally:
        await conn.close()
    if result == 'DELETE 0':
        raise HTTPException(status_code=404, detail="Moment not found")
    return

@router.post("/api/moments", response_model=MomentSolution, status_code=201)
async def create_moment(create_data: MomentSolutionCreate):
    """
    Creates a new moment and solution, generates an embedding, and stores it.
    New moments default to having a solution (has_solution = TRUE).
    """
    conn = await get_db_connection()

    # Generate embedding for the new moment's content
    combined_text = f"{create_data.moment} {create_data.solution}"
    embedding = await generate_embedding(combined_text)

    try:
        # Insert with default has_solution = TRUE and return the new field
        row = await conn.fetchrow(
            "INSERT INTO moments(moment, solution, embedding, has_solution) VALUES($1, $2, $3, $4) RETURNING id, moment, solution, has_solution",
            create_data.moment, create_data.solution, embedding, True # Default has_solution to True
        )
    finally:
        await conn.close()

    # Ensure has_solution is passed to the Pydantic model
    return MomentSolution(id=row['id'], moment=row['moment'], solution=row['solution'], has_solution=row['has_solution'])

@router.put("/api/moments/{moment_id}", response_model=MomentSolution)
async def update_moment(moment_id: int, update_data: MomentSolutionUpdate):
    """
    Updates an existing moment's text and solution, and re-generates its embedding.
    The 'has_solution' status is updated via a separate endpoint (in problem_logging.py).
    """
    conn = await get_db_connection()

    # Generate new embedding for the updated moment's content
    combined_text = f"{update_data.moment} {update_data.solution}"
    embedding = await generate_embedding(combined_text)

    try:
        # Update moment and solution, but has_solution is NOT updated here.
        # However, we still RETURN has_solution to keep the response model consistent.
        row = await conn.fetchrow(
            "UPDATE moments SET moment=$1, solution=$2, embedding=$3 WHERE id=$4 RETURNING id, moment, solution, has_solution",
            update_data.moment, update_data.solution, embedding, moment_id
        )
    finally:
        await conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Moment not found")
    # Ensure has_solution is passed to the Pydantic model
    return MomentSolution(id=row['id'], moment=row['moment'], solution=row['solution'], has_solution=row['has_solution'])

# --- Vector Similarity Search Endpoint ---
@router.get("/api/search-moments", response_model=List[MomentSolution])
async def search_moments(
    query: str = Query(..., min_length=1, description="The user's search query."),
    top_k: int = Query(5, ge=1, le=20, description="Number of top similar moments to return.")
):
    """
    Searches for moments using vector similarity search based on the user query.
    Generates an embedding for the query and finds the closest moments in the database.
    """
    # 1. Generate an embedding vector for the user query
    query_embedding = await generate_embedding(query)

    conn = await get_db_connection()
    try:
        # 2. Query PostgreSQL using pgvector <-> operator for closest matches
        # The <-> operator calculates Euclidean distance. A smaller distance means a closer match.
        # ORDER BY embedding <-> $1 ensures results are ordered by similarity (closest first).
        # Also select the new 'has_solution' column
        rows = await conn.fetch(
            "SELECT id, moment, solution, has_solution FROM moments ORDER BY embedding <-> $1 LIMIT $2",
            query_embedding, top_k
        )
    finally:
        await conn.close()

    # 3. Return the found moments, ensuring has_solution is included
    return [MomentSolution(id=row['id'], moment=row['moment'], solution=row['solution'], has_solution=row['has_solution']) for row in rows]

# --- Optional: Endpoint to migrate existing moments to have embeddings ---
@router.post("/api/migrate-existing-moments-embeddings", status_code=200)
async def migrate_existing_moments_embeddings():
    """
    Generates and stores embeddings for any existing moments that currently have a NULL embedding.
    This is a one-time migration endpoint.
    """
    conn = await get_db_connection()
    processed_count = 0
    failed_count = 0
    try:
        # Fetch moments that do not have an embedding yet (where embedding IS NULL)
        # Note: This migration does not handle 'has_solution' as it's a separate concern.
        rows_to_process = await conn.fetch("SELECT id, moment, solution FROM moments WHERE embedding IS NULL")
        
        if not rows_to_process:
            return {"message": "All moments already have embeddings or no moments exist that require migration."}

        for moment_data in rows_to_process:
            combined_text = f"{moment_data['moment']} {moment_data['solution']}"
            try:
                embedding = await generate_embedding(combined_text)
                await conn.execute(
                    "UPDATE moments SET embedding=$1 WHERE id=$2",
                    embedding, moment_data['id']
                )
                processed_count += 1
            except HTTPException as e: # Catch HTTPExceptions specifically for API errors
                print(f"WARNING: API Error for moment ID {moment_data['id']}: {e.detail}")
                failed_count += 1
            except Exception as e: # Catch other general exceptions
                print(f"WARNING: General error for moment ID {moment_data['id']}: {e}")
                failed_count += 1
                # Continue processing other moments even if one fails
        
        return {
            "message": f"Migration complete. Processed {processed_count} moments successfully, {failed_count} failed.",
            "total_moments_to_process": len(rows_to_process),
            "note": "You can re-run this if some failed, or if new old data was added. It only processes NULL embeddings."
        }
    finally:
        await conn.close()