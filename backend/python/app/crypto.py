import os
import httpx
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

# Simple sentiment based on price movement
def price_sentiment(prev_price: float, current_price: float) -> str:
    diff = current_price - prev_price
    if diff > 0.5:
        return "Bullish"
    elif diff < -0.5:
        return "Bearish"
    else:
        return "Neutral"

class CryptoRequest(BaseModel):
    symbol: str  # e.g., BTCUSD

@router.post("/crypto-sentiment")
async def crypto_sentiment(req: CryptoRequest):
    # Gemini public API URL for last trade price
    url = f"https://api.gemini.com/v1/pubticker/{req.symbol.lower()}"
    
    async with httpx.AsyncClient() as client:
        res = await client.get(url)
        data = res.json()

    last_price = float(data["last"])
    prev_price = float(data["prev_close_price"]) if "prev_close_price" in data else last_price

    sentiment = price_sentiment(prev_price, last_price)

    # TODO: Log user query & sentiment to DB here

    return {
        "symbol": req.symbol.upper(),
        "last_price": last_price,
        "previous_close": prev_price,
        "sentiment": sentiment
    }
