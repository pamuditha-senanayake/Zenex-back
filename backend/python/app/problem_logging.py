from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from app.db import get_db_connection
from app.search import generate_embedding # Import the embedding function

router = APIRouter()

# --- Pydantic Models for Problem Logs ---

# Model for data received from the client when creating a log
class ProblemLogCreate(BaseModel):
    moment_id: int
    mood: Optional[str] = None
    context: Optional[str] = None
    triggers: Optional[str] = None
    extra_details: Optional[str] = None
    # Client does not provide timestamp or embeddings for creation


# Model for data returned from the database / API response for a ProblemLog entry
class ProblemLog(BaseModel):
    id: int
    moment_id: int
    timestamp: datetime
    mood: Optional[str] = None
    context: Optional[str] = None
    triggers: Optional[str] = None
    extra_details: Optional[str] = None
    mood_embedding: Optional[List[float]] = None
    context_embedding: Optional[List[float]] = None
    triggers_embedding: Optional[List[float]] = None
    extra_details_embedding: Optional[List[float]] = None
    created_at: datetime

    class Config:
        from_attributes = True # Allows Pydantic to map from ORM objects (like asyncpg.Record)


# Pydantic Model for Solution Status Update (for the 'moments' table)
class SolutionStatusUpdate(BaseModel):
    has_solution: bool

# Pydantic Model for No-Solution Summary
class NoSolutionSummary(BaseModel):
    id: int
    moment: str
    solution: str
    log_count: int
    last_logged: Optional[datetime]

    class Config:
        from_attributes = True

# Pydantic Model for ML Predicted Risk (from ml_service.py)
class PredictedRisk(BaseModel):
    moment_id: int
    moment_text: str
    predicted_risk: float # Probability from 0 to 1
    prediction_time: str # e.g., "tomorrow morning", "within the next 3 days"
    # Add other context if needed, e.g., suggested solution based on historical success

    class Config:
        from_attributes = True


# --- Problem Logging Endpoints ---

@router.post("/api/problem-logs", response_model=ProblemLog, status_code=201)
async def create_problem_log(log_data: ProblemLogCreate):
    conn = await get_db_connection()

    # Check if the moment_id exists before proceeding
    moment_exists = await conn.fetchval("SELECT 1 FROM moments WHERE id = $1", log_data.moment_id)
    if not moment_exists:
        await conn.close() # Ensure connection is closed on early exit
        raise HTTPException(status_code=404, detail=f"Moment with ID {log_data.moment_id} not found.")

    # Generate embeddings for optional text fields if they exist and are not empty
    # The `strip()` ensures that strings with only whitespace are treated as empty
    # If generate_embedding returns an empty list for empty/None input, this needs to be handled
    # by ensuring the DB column accepts empty vector or None. Here, we send None.
    mood_embedding = await generate_embedding(log_data.mood) if log_data.mood and log_data.mood.strip() else None
    context_embedding = await generate_embedding(log_data.context) if log_data.context and log_data.context.strip() else None
    triggers_embedding = await generate_embedding(log_data.triggers) if log_data.triggers and log_data.triggers.strip() else None
    extra_details_embedding = await generate_embedding(log_data.extra_details) if log_data.extra_details and log_data.extra_details.strip() else None

    # Use current server-side timestamp for accuracy and consistency
    current_timestamp = datetime.now()

    try:
        # Insert the new log entry, including generated embeddings
        # Ensure the number of columns in INSERT matches the number of values in VALUES
        # And ensure all fields required by ProblemLog model are in RETURNING
        row = await conn.fetchrow(
            """
            INSERT INTO problem_logs(
                moment_id, timestamp, mood, context, triggers, extra_details,
                mood_embedding, context_embedding, triggers_embedding, extra_details_embedding, created_at
            )
            VALUES($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            RETURNING id, moment_id, timestamp, mood, context, triggers, extra_details,
                      mood_embedding, context_embedding, triggers_embedding, extra_details_embedding, created_at
            """,
            log_data.moment_id,
            current_timestamp, # $2: Use current_timestamp for 'timestamp'
            log_data.mood,     # $3
            log_data.context,  # $4
            log_data.triggers, # $5
            log_data.extra_details, # $6
            mood_embedding,        # $7
            context_embedding,     # $8
            triggers_embedding,    # $9
            extra_details_embedding, # $10
            current_timestamp      # $11: Use current_timestamp for 'created_at' too
        )

        if row is None:
            # This case implies the insert didn't return a row, which is unexpected with RETURNING.
            raise HTTPException(status_code=500, detail="Failed to retrieve newly created log from database after insertion.")

    finally:
        await conn.close()

    # Convert asyncpg.Record to a standard dictionary before Pydantic validation.
    # This ensures Pydantic works with an unambiguous dictionary, resolving the "missing field" error
    # even when the record's __repr__ shows the fields.
    return ProblemLog.model_validate(dict(row))

@router.get("/api/problem-logs", response_model=List[ProblemLog])
async def get_problem_logs(moment_id: Optional[int] = Query(None, description="Filter logs by a specific moment ID.")):
    """Retrieves problem logs, optionally filtered by a specific moment ID."""
    conn = await get_db_connection()
    try:
        select_clause = """
            id, moment_id, timestamp, mood, context, triggers, extra_details, created_at,
            mood_embedding, context_embedding, triggers_embedding, extra_details_embedding
        """
        if moment_id:
            rows = await conn.fetch(f"SELECT {select_clause} FROM problem_logs WHERE moment_id = $1 ORDER BY timestamp DESC", moment_id)
        else:
            # For general logs page, limit for performance to avoid huge data transfer
            rows = await conn.fetch(f"SELECT {select_clause} FROM problem_logs ORDER BY timestamp DESC LIMIT 500")
    finally:
        await conn.close()
    # Apply the same dict conversion when fetching multiple rows for consistency
    return [ProblemLog.model_validate(dict(row)) for row in rows]

# --- Endpoint to Update Moment's Solution Status ---
@router.put("/api/moments/{moment_id}/solution-status", status_code=200)
async def update_moment_solution_status(moment_id: int, status_update: SolutionStatusUpdate):
    """Updates the 'has_solution' status for a specific moment."""
    conn = await get_db_connection()
    try:
        result = await conn.execute(
            "UPDATE moments SET has_solution=$1 WHERE id=$2",
            status_update.has_solution, moment_id
        )
    finally:
        await conn.close()
    if result == 'UPDATE 0':
        raise HTTPException(status_code=404, detail="Moment not found")
    return {"message": "Solution status updated successfully"}

# --- Endpoint for Summary of No-Solution Moments ---
@router.get("/api/summary/no-solution-moments", response_model=List[NoSolutionSummary])
async def get_no_solution_summary():
    """Retrieves a summary of moments marked as 'no solution', including log counts."""
    conn = await get_db_connection()
    try:
        rows = await conn.fetch(
            """
            SELECT
                m.id,
                m.moment,
                m.solution,
                COUNT(pl.id) AS log_count,
                MAX(pl.timestamp) AS last_logged
            FROM
                moments m
            LEFT JOIN -- Use LEFT JOIN to include moments even if they have no logs
                problem_logs pl ON m.id = pl.moment_id
            WHERE
                m.has_solution = FALSE
            GROUP BY
                m.id, m.moment, m.solution
            ORDER BY
                last_logged DESC NULLS LAST, log_count DESC
            """
        )
    finally:
        await conn.close()
    # Apply the same dict conversion here for consistency, though it might not be strictly needed for this model
    return [NoSolutionSummary.model_validate(dict(row)) for row in rows]

# --- ML Prediction Endpoint (Now using actual ML service) ---
import app.ml_service as ml_service # Import the new ML service

@router.get("/api/predictions/risk-alerts", response_model=List[PredictedRisk])
async def get_risk_alerts():
    """
    Retrieves predicted high-risk recurring problems using the trained ML model.
    This endpoint calls the ML service to get real-time predictions.
    """
    predictions = await ml_service.get_predictions_for_display()
    return predictions