from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from typing import Dict
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from utils import extract_article_content, split_text_into_chunks

app = FastAPI()

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins. In production, replace "*" with specific domains.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# clf = pipeline("text-classification", model="jy46604790/Fake-News-Bert-Detect")

clf = pipeline("text-classification", model="dhruvpal/fake-news-ber")

  
class URLRequest(BaseModel):
    url: str


@app.post("/predict", response_model=Dict[str, float])
async def predict(request: URLRequest):
    url = request.url

    # Extract article content
    try:
        title, content = extract_article_content(url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error extracting content: {e}")

    # Split content into manageable chunks
    chunks = split_text_into_chunks(content)

    results = []
    for chunk in chunks:
        result = clf(chunk)
        results.append(result[0])  # Extract the first (and only) result

    # If any chunk is labeled as true (LABEL_1), return true with its score
    for result in results:
        if result['label'] == 'LABEL_1':
            return {"true": result['score']}

    # Otherwise, return false with the highest fake score from chunks labeled as false (LABEL_0)
    highest_fake_score = max(result['score'] for result in results if result['label'] == 'LABEL_0')
    return {"false": highest_fake_score}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
