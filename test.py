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
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


clf = pipeline("text-classification", model="jy46604790/Fake-News-Bert-Detect")

class URLRequest(BaseModel):
    url: str

class TextRequest(BaseModel):
    text: str


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/news", response_model=Dict[str, float])
async def news(request: URLRequest):
    url = request.url

    
    try:
        title, content = extract_article_content(url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error extracting content: {e}")

    
    chunks = split_text_into_chunks(content)

    results = []
    for chunk in chunks:
        result = clf(chunk)
        results.append(result[0])

    
    for result in results:
        if result['label'] == 'LABEL_1':
            return {"true": result['score']}

    
    highest_fake_score = max(result['score'] for result in results if result['label'] == 'LABEL_0')
    return {"false": highest_fake_score}

@app.post("/social", response_model=Dict[str, float])
async def social(request: TextRequest):
    text = request.text

    
    chunks = split_text_into_chunks(text)

    results = []
    for chunk in chunks:
        result = clf(chunk)
        results.append(result[0])  

    
    for result in results:
        if result['label'] == 'LABEL_1':
            return {"true": result['score']}

    
    highest_fake_score = max(result['score'] for result in results if result['label'] == 'LABEL_0')
    return {"false": highest_fake_score}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
