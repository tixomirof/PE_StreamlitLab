
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn

from brains import TranslateAndEmotion

app = FastAPI(title="PE Streamlit Lab API")

# global model instance; created on startup to avoid heavy work at import time
model = None


@app.on_event("startup")
def startup_event():
	global model
	model = TranslateAndEmotion()


class TextRequest(BaseModel):
	text: str


@app.get("/")
async def root():
	return {"status": "ok", "message": "PE Streamlit Lab API"}


@app.post("/classify")
async def classify_text(req: TextRequest):
	if model is None:
		raise HTTPException(status_code=503, detail="Model not loaded")
	sentences = model.get_sentences_from_text(req.text)
	translations = model.translate_all_sentences(sentences)
	emotions = model.classified_emotions_from_data(translations)
	counts, _ = model.count_emotions(emotions)
	return {
		"sentences": sentences,
		"translations": translations,
		"emotions": emotions,
		"counts": counts,
	}


@app.post("/classify-file")
async def classify_file(file: UploadFile = File(...)):
	if model is None:
		raise HTTPException(status_code=503, detail="Model not loaded")
	content = await file.read()
	try:
		text = content.decode("utf-8")
	except Exception:
		raise HTTPException(status_code=400, detail="File must be UTF-8 text")
	sentences = model.get_sentences_from_text(text)
	translations = model.translate_all_sentences(sentences)
	emotions = model.classified_emotions_from_data(translations)
	counts, _ = model.count_emotions(emotions)
	return {
        "text": text,
		"sentences": sentences,
		"translations": translations,
		"emotions": emotions,
		"counts": counts,
	}


@app.get("/classify-comments")
async def classify_comments():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    comments = model.get_comments_data()
    sentences = [comment.strip() for comment in comments if comment.strip()]
    translations = model.translate_all_sentences(sentences)
    emotions = model.classified_emotions_from_data(translations)
    counts, _ = model.count_emotions(emotions)
    return {
        "sentences": sentences,
        "translations": translations,
        "emotions": emotions,
        "counts": counts,
    }


@app.post("/add-comment")
async def add_comment(req: TextRequest):
	if model is None:
		raise HTTPException(status_code=503, detail="Model not loaded")
	model.set_new_comment_in_data(req.text)
	return {"status": "ok"}


if __name__ == "__main__":
	uvicorn.run("api:app", host="0.0.0.0", port=8000)

