
from fastapi import FastAPI, HTTPException, File, UploadFile, Depends
from pydantic import BaseModel
from transformers import pipeline
from typing import Dict
import uvicorn
import shutil
import os
from fastapi.middleware.cors import CORSMiddleware
from utils import extract_article_content, split_text_into_chunks
import easyocr

app = FastAPI()

# Configure CORS middleware
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins. In production, specify allowed domains.
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Explicitly allow OPTIONS requests
    allow_headers=["*"],
)



clf = pipeline("text-classification", model="dhruvpal/fake-news-bert")

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
    sanitized_text = text.replace('"', "'").replace("\\", " ")  # Replace quotes & backslashes
    sanitized_text = " ".join(sanitized_text.split())

    
    chunks = split_text_into_chunks(sanitized_text)

    results = []
    for chunk in chunks:
        result = clf(chunk)
        results.append(result[0])  

    
    for result in results:
        if result['label'] == 'LABEL_1':
            return {"true": result['score']}

    
    highest_fake_score = max(result['score'] for result in results if result['label'] == 'LABEL_0')
    return {"false": highest_fake_score}

reader = easyocr.Reader(['en'])

@app.post("/getimage")
async def get_image(file: UploadFile = File(...)):  # Change "image" → "file"
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Read text from the image using EasyOCR
        result = reader.readtext(temp_file_path)
        paragraph = " ".join([detection[1] for detection in result])

        if not paragraph:
            raise HTTPException(status_code=400, detail="No text detected in the image.")

        # Split text into chunks for model inference
        chunks = split_text_into_chunks(paragraph)

        # Analyze each chunk
        results = []
        for chunk in chunks:
            result = clf(chunk)
            results.append(result)

        # Flatten results
        flattened_results = [item for sublist in results for item in sublist]

        # Check if any chunk is labeled as 'LABEL_1'
        for result in flattened_results:
            if result['label'] == 'LABEL_1':
                return {"true": result['score']}  # Ensure it returns a JSON object

        # If no 'LABEL_1' is found, find the highest 'LABEL_0' score
        highest_fake_score = max(result['score'] for result in flattened_results if result['label'] == 'LABEL_0')
        return {"false": highest_fake_score}

    finally:
        os.remove(temp_file_path)
  


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
