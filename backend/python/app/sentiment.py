from fastapi import APIRouter
from pydantic import BaseModel
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Ensure VADER lexicon is downloaded (optional, just for safety)
nltk.download('vader_lexicon')

router = APIRouter()
sia = SentimentIntensityAnalyzer()

class InputData(BaseModel):
    text: str

def improved_sentiment_analysis(text: str) -> str:
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return 'Positive'
    elif compound <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

@router.post("/analyze")
async def analyze(input_data: InputData):
    sentiment = improved_sentiment_analysis(input_data.text)
    return {"text": input_data.text, "sentiment": sentiment}



# def dummy_sentiment_analysis(text: str) -> str:
#     positive_words = ['good', 'happy', 'great', 'awesome', 'love']
#     negative_words = ['bad', 'sad', 'terrible', 'awful']
#     text_lower = text.lower()
#     if any(word in text_lower for word in positive_words):
#         return 'Positive'
#     elif any(word in text_lower for word in negative_words):
#         return 'Negative'
#     else:
#         return 'Neutral'

