import asyncpg
import os
from dotenv import load_dotenv
import pathlib
from pgvector.asyncpg import register_vector # <--- Add this import!

# Load .env two levels above this file
env_path = pathlib.Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=str(env_path))

DB_HOST = os.getenv('PG_HOST', 'localhost')
DB_PORT = int(os.getenv('PG_PORT', 5432))
DB_NAME = os.getenv('PG_DATABASE', 'yourdbname')
DB_USER = os.getenv('PG_USER', 'yourdbuser')
DB_PASS = os.getenv('PG_PASSWORD', 'yourdbpass')

async def get_db_connection():
    conn = await asyncpg.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME
    )
    # Register the vector type for THIS SPECIFIC connection
    await register_vector(conn) # <--- Add this line!
    return conn

# No need for close_db_connection_pool if you're not using a pool,
# but remember to close individual connections in your FastAPI endpoints!
# Your existing FastAPI code already does this with await conn.close() in a finally block.