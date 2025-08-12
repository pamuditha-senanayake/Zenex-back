import pandas as pd
import numpy as np
from datetime import datetime, timedelta,timezone
import joblib
import os
import asyncio # For async operations within a synchronous script
import asyncpg # For direct database access from ML module

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from typing import Optional, List

# --- Configuration ---
# Ensure DATABASE_URL environment variable is set before running the script or FastAPI app
DATABASE_URL = os.getenv("DATABASE_URL")

# Directory and filename for the trained ML model
MODEL_DIR = "ml_models"
MODEL_FILENAME = "xgb_risk_predictor.joblib"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# Create the model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Global variables to hold the loaded model and its feature names
# These are loaded once on FastAPI startup to avoid reloading for every prediction request
_ml_model = None
_ml_features = None

# --- Helper for database connection (for ML service) ---
async def _get_db_connection_ml():
    """Establishes an asynchronous connection to the PostgreSQL database."""
    if not DATABASE_URL:
        # This error should have been caught earlier by direct execution of ml_service
        # or by checking env vars before starting FastAPI.
        raise ValueError("DATABASE_URL environment variable is not set for ML service.")
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        return conn
    except Exception as e:
        print(f"Error connecting to database from ML service: {e}")
        raise # Re-raise to ensure connection failure is propagated

# --- Feature Engineering Functions ---

async def _fetch_raw_ml_data():
    """
    Fetches all necessary raw data from 'moments' and 'problem_logs' tables.
    Data is ordered by moment_id and timestamp for sequential feature engineering.
    """
    conn = await _get_db_connection_ml()
    try:
        print(f"DEBUG: Attempting to fetch raw ML data from DB...")
        rows = await conn.fetch("""
            SELECT
                pl.id AS log_id,
                pl.moment_id,
                pl.timestamp,
                pl.mood_embedding,
                pl.context_embedding,
                pl.triggers_embedding,
                pl.extra_details_embedding,
                m.moment,
                m.has_solution
            FROM problem_logs pl
            JOIN moments m ON pl.moment_id = m.id
            ORDER BY pl.moment_id, pl.timestamp
        """)

        # --- Debugging prints for fetched data ---
        print(f"DEBUG: Fetched {len(rows)} rows from the database.")
        if len(rows) > 0:
            print(f"DEBUG: First record (asyncpg.Record) keys: {rows[0].keys()}")
            print(f"DEBUG: First record values: {rows[0].values()}")
            if 'moment_id' in rows[0]:
                print(f"DEBUG: 'moment_id' is directly accessible in first record.")
            else:
                print(f"DEBUG: 'moment_id' is NOT directly accessible in first record (this is unexpected).")
        # --- End Debugging prints ---

        # *** CRITICAL FIX HERE ***
        # Convert each asyncpg.Record object to a dictionary before passing to DataFrame
        # This ensures Pandas correctly infers column names from dictionary keys.
        rows_as_dicts = [dict(row) for row in rows]
        df = pd.DataFrame(rows_as_dicts)
        # *** END CRITICAL FIX ***

        # --- Debugging prints for DataFrame ---
        print(f"DEBUG: Pandas DataFrame created. Empty: {df.empty}")
        print(f"DEBUG: DataFrame columns: {df.columns.tolist()}")
        if not df.empty:
            print(f"DEBUG: First 5 rows of df_raw:\n{df.head()}")
        # --- End Debugging prints ---

        return df
    except Exception as e:
        print(f"ERROR: Exception during _fetch_raw_ml_data: {e}")
        raise # Re-raise to ensure the error propagates and is seen
    finally:
        await conn.close()

def _prepare_features(df_input: pd.DataFrame, training_mode: bool = True, known_features: Optional[List[str]] = None):
    """
    Prepares and engineers features from the raw data for ML training or prediction.
    Ensures consistency of feature columns between training and inference.

    Args:
        df_input (pd.DataFrame): Raw data (for training) or a single row for prediction.
        training_mode (bool): True if preparing data for training, False for prediction.
        known_features (Optional[List[str]]): List of feature names from training, used in prediction mode.

    Returns:
        pd.DataFrame: Feature matrix (X)
        pd.Series: Target variable (y) (only in training_mode)
        List[str]: Names of the generated feature columns (only in training_mode)
    """

    if df_input.empty:
        if training_mode:
            return pd.DataFrame(), pd.Series(), []
        else:
            return pd.DataFrame() # Empty DataFrame for prediction mode

    df = df_input.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df = df.sort_values(by=['moment_id', 'timestamp']).reset_index(drop=True)

    # --- Temporal Features ---
    df['time_since_last_occurrence'] = df.groupby('moment_id')['timestamp'].diff().dt.total_seconds().fillna(0) / (24 * 3600) # Days
    df['day_of_week'] = df['timestamp'].dt.dayofweek # 0=Monday, 6=Sunday
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_year'] = df['timestamp'].dt.dayofyear # Day of the year (1-366)
    df['week_of_year'] = df['timestamp'].dt.isocalendar().week.astype(int) # Week of the year (1-53)

    # --- Frequency Features (Rolling Counts) ---
    df['rolling_7_day_count'] = df.groupby('moment_id').rolling('7D', on='timestamp')['log_id'].count().reset_index(level=0, drop=True)
    df['rolling_30_day_count'] = df.groupby('moment_id').rolling('30D', on='timestamp')['log_id'].count().reset_index(level=0, drop=True)

    # --- Categorical/Boolean Features ---
    df['has_solution'] = df['has_solution'].astype(int) # Convert boolean to integer (0 or 1)

    # --- Embedding Features ---
    embedding_dim = 768 # Assuming text-embedding-004 dimension

    for col_prefix in ['mood', 'context', 'triggers', 'extra_details']:
        col_name = f"{col_prefix}_embedding"
        # Safely convert to numpy array, handling None or potentially empty lists
        # Ensures that embeddings are always of the correct dimension, filling with zeros if missing or incorrect
        df[col_name] = df[col_name].apply(
            lambda x: np.array(x) if (isinstance(x, list) and len(x) == embedding_dim) else np.zeros(embedding_dim)
        )

        # Expand embeddings into separate columns
        expanded_embeddings = pd.DataFrame(df[col_name].tolist(), index=df.index,
                                           columns=[f'{col_prefix}_emb_{i}' for i in range(embedding_dim)])
        df = pd.concat([df, expanded_embeddings], axis=1)
        df = df.drop(columns=[col_name]) # Drop the original list-of-floats column

    # --- Target Variable Calculation (only in training mode) ---
    if training_mode:
        df['next_occurrence_timestamp'] = df.groupby('moment_id')['timestamp'].shift(-1)
        df['time_to_next_occurrence'] = (df['next_occurrence_timestamp'] - df['timestamp']).dt.total_seconds() / (24 * 3600)
        df['target_7_day_recurrence'] = ((df['time_to_next_occurrence'] > 0) & (df['time_to_next_occurrence'] <= 7)).astype(int)

        df_final = df.dropna(subset=['target_7_day_recurrence']).copy()

        feature_cols = [col for col in df_final.columns if col.startswith((
            'time_', 'day_of_', 'hour_of_', 'week_of_', 'rolling_', 'has_solution',
            'mood_emb_', 'context_emb_', 'triggers_emb_', 'extra_details_emb_'
        ))]

        X = df_final[feature_cols]
        y = df_final['target_7_day_recurrence']

        return X, y, feature_cols

    # --- Feature Preparation for Prediction Mode ---
    else: # Prediction mode
        X_pred_aligned = pd.DataFrame(columns=known_features) # Create empty DF with all training features

        for col in known_features:
            if col in df.columns:
                X_pred_aligned[col] = df[col].iloc[0] # Take the value from the current row
            else:
                X_pred_aligned[col] = 0 # Fill with 0 if feature was not present in this specific log

        return X_pred_aligned

# --- Model Training ---
async def train_model_async():
    """
    Orchestrates fetching data, preparing features, training, and saving the model.
    This function should be run as a separate background task or cron job.
    """
    global _ml_model, _ml_features

    print(f"[{datetime.now()}] Starting ML model training...")
    df_raw = await _fetch_raw_ml_data() # Fetch raw data

    # --- ROBUST DATA CHECKS BEFORE TRAINING ---
    # 1. Check if DataFrame is completely empty after fetching
    if df_raw.empty:
        print("No raw data fetched from the database. Skipping training.")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH) # Remove any stale model file if no data
        _ml_model = None
        _ml_features = None
        return

    # 2. Check if 'moment_id' column exists. This should only be hit if df_raw is NOT empty,
    # but somehow the column didn't form correctly (unlikely with dict conversion).
    if 'moment_id' not in df_raw.columns:
        print("The fetched data does not contain a 'moment_id' column. Check database schema and query. Skipping training.")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        _ml_model = None
        _ml_features = None
        return

    # 3. Check for sufficient unique moments for meaningful training (at least 2)
    if len(df_raw['moment_id'].unique()) < 2:
        print("Not enough unique moments (less than 2) to train a meaningful model. Skipping training.")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        _ml_model = None
        _ml_features = None
        return

    # Prepare features and target variable from the raw data
    X, y, feature_cols = _prepare_features(df_raw, training_mode=True)

    # 4. Final check after feature preparation (e.g., if target calculation resulted in all NaNs)
    if X.empty or y.empty or y.nunique() < 2:
        print("After feature preparation, data is insufficient or target has no variance (all 0s or all 1s). Skipping training.")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        _ml_model = None
        _ml_features = None
        return

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)

    # Calculate `scale_pos_weight` for handling class imbalance (if occurrences are rare)
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum() if y_train.sum() > 0 else 1.0

    # Initialize and train the XGBoost Classifier model
    model = XGBClassifier(
        objective='binary:logistic',     # For binary classification
        eval_metric='logloss',           # Evaluation metric for early stopping/logging
        use_label_encoder=False,         # Suppress a deprecation warning (for older XGBoost)
        n_estimators=200,                # Number of boosting rounds (trees)
        learning_rate=0.05,              # Step size shrinkage to prevent overfitting
        max_depth=5,                     # Maximum depth of a tree
        subsample=0.8,                   # Subsample ratio of the training instance (for bagging)
        colsample_bytree=0.8,            # Subsample ratio of columns when constructing each tree
        random_state=42,                 # For reproducibility
        scale_pos_weight=scale_pos_weight # To balance positive and negative classes
    )

    model.fit(X_train, y_train)

    # Evaluate model performance on training and test sets
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    print(f"[{datetime.now()}] Model training complete. Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Save the trained model and its associated feature names
    model_data = {
        'model': model,
        'features': feature_cols # It's crucial to save the feature names for consistent prediction
    }
    joblib.dump(model_data, MODEL_PATH)
    print(f"[{datetime.now()}] Model saved to {MODEL_PATH}")

    # Update global variables for the running FastAPI application
    _ml_model = model
    _ml_features = feature_cols

# --- Model Loading (for FastAPI startup) ---
def load_ml_model():
    """
    Loads the trained ML model and its features into global memory.
    This function is synchronous and intended to be called during FastAPI startup.
    """
    global _ml_model, _ml_features
    if os.path.exists(MODEL_PATH):
        try:
            print(f"[{datetime.now()}] Loading ML model from {MODEL_PATH}...")
            model_data = joblib.load(MODEL_PATH)
            _ml_model = model_data['model']
            _ml_features = model_data['features']
            print(f"[{datetime.now()}] ML model loaded successfully.")
        except Exception as e:
            print(f"[{datetime.now()}] Error loading ML model: {e}")
            _ml_model = None
            _ml_features = None
    else:
        print(f"[{datetime.now()}] No ML model found at {MODEL_PATH}. Prediction service will be inactive.")
        _ml_model = None
        _ml_features = None

# --- Prediction Inference ---
async def get_predictions_for_display():
    """
    Generates real-time predictions for recurring problems based on the loaded ML model
    and the most recent state of problem logs for each moment.
    """
    if _ml_model is None or _ml_features is None:
        print(f"[{datetime.now()}] ML model not loaded or features not available. Cannot generate predictions.")
        return []

    conn = await _get_db_connection_ml()
    try:
        # Fetch the most recent log entry for each unique moment_id
        # This represents the 'current state' from which we want to predict future occurrences.
        rows = await conn.fetch("""
            WITH LastLogs AS (
                SELECT
                    pl.moment_id,
                    MAX(pl.timestamp) AS last_timestamp
                FROM problem_logs pl
                GROUP BY pl.moment_id
            )
            SELECT
                m.id AS moment_id,
                m.moment,
                m.has_solution,
                pl.timestamp, -- This is the timestamp of the LAST log for this moment
                pl.mood_embedding,
                pl.context_embedding,
                pl.triggers_embedding,
                pl.extra_details_embedding
            FROM moments m
            JOIN LastLogs ll ON m.id = ll.moment_id
            JOIN problem_logs pl ON ll.moment_id = pl.moment_id AND ll.last_timestamp = pl.timestamp
            ORDER BY m.id
        """)

        predictions_output = []
        current_time = datetime.now(timezone.utc) # The time at which we are making the prediction

        for row in rows:
            # Convert asyncpg.Record to dictionary for consistent DataFrame creation
            row_dict = dict(row)

            # Create a DataFrame for a single prediction instance from the current row data
            current_moment_df = pd.DataFrame([row_dict])

            # Recalculate time_since_last_occurrence relative to current_time
            current_moment_df['time_since_last_occurrence'] = (current_time - current_moment_df['timestamp']).dt.total_seconds() / (24 * 3600)

            # Cyclical time features based on the *current time* (prediction time)
            current_moment_df['day_of_week'] = current_time.weekday()
            current_moment_df['hour_of_day'] = current_time.hour
            current_moment_df['day_of_year'] = current_time.timetuple().tm_yday
            current_moment_df['week_of_year'] = current_time.isocalendar().week


            # Rolling counts for current prediction:
            # These are challenging to compute accurately for a single row without more history.
            # For this simplified example, setting to 0. A robust solution might pre-calculate these daily.
            current_moment_df['rolling_7_day_count'] = 0 # Placeholder for simplicity
            current_moment_df['rolling_30_day_count'] = 0 # Placeholder for simplicity

            # Prepare features using the shared _prepare_features function (in prediction mode)
            X_pred = _prepare_features(current_moment_df, training_mode=False, known_features=_ml_features)

            # Ensure X_pred has all the expected features from training, filling missing with 0
            missing_cols = set(_ml_features) - set(X_pred.columns)
            for c in missing_cols:
                X_pred[c] = 0
            X_pred = X_pred[_ml_features] # Ensure order of columns is correct for model input

            try:
                # Predict the probability of recurrence for the next 7 days
                prediction_proba = _ml_model.predict_proba(X_pred)[:, 1][0]

                # Filter for high-risk moments based on a probability threshold
                if prediction_proba > 0.6: # Example threshold (can be tuned)
                    predictions_output.append({
                        "moment_id": row_dict['moment_id'], # Use dict for consistency
                        "moment_text": row_dict['moment'],
                        "predicted_risk": float(prediction_proba),
                        "prediction_time": "within the next 7 days" # Reflects the target definition
                    })
            except Exception as e:
                print(f"[{datetime.now()}] Error predicting for moment {row_dict['moment_id']}: {e}")
                continue

        # Sort predictions by risk level (highest first)
        predictions_output.sort(key=lambda x: x['predicted_risk'], reverse=True)
        return predictions_output
    finally:
        await conn.close()

# --- Entry point for manual training (e.g., cron job or command line) ---
if __name__ == "__main__":
    # To run training manually from terminal: python -m app.ml_service
    # Ensure DATABASE_URL env var is set in your environment
    asyncio.run(train_model_async())