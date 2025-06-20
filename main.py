from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
# from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel
from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration
import torch
import json
import time
import threading
from transformers import TextIteratorStreamer

app = FastAPI()

# Mount templates directory
templates = Jinja2Templates(directory="templates")

model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

class TranslationRequest(BaseModel):
    text: str
    source_language: str
    target_language: str

def translate_text(text: str, src_lang: str, tgt_lang: str) -> str:
    try:
        tokenizer.src_lang = src_lang
        encoded = tokenizer(text, return_tensors="pt")
        
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.get_lang_id(tgt_lang),
            max_length=100
        )
        
        translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def stream_translation(text: str, src_lang: str, tgt_lang: str):
    try:
        tokenizer.src_lang = src_lang
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
        gen_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": 100,
            "forced_bos_token_id": tokenizer.get_lang_id(tgt_lang),
            "num_beams": 1,
        }

        # Run generation in background
        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        # Stream tokens in real-time
        for token in streamer:
            if token.strip():
                yield token + " "
                time.sleep(0.05)

    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.post("/translate")
async def translate(request: TranslationRequest):
    try:
        translation = translate_text(request.text, request.source_language, request.target_language)
        return {"translation": translation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate/stream")
async def translate_stream(request: TranslationRequest):
    try:
        generator = stream_translation(request.text, request.source_language, request.target_language)
        return StreamingResponse(
            generator,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/translation_page", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("translation_page.html", {"request": request})

def stream_hardcoded_text():
    hardcoded_text = "This is a hardcoded string that will be streamed to the user character by character. Hello from the streaming API!"
    for char in hardcoded_text:
        # response_data = {"text": char}
        yield f"{char}"
        # Add a small delay to make the streaming visible
        time.sleep(0.1)

@app.get("/stream-text")
async def stream_text():
    try:
        return StreamingResponse(
            stream_hardcoded_text(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
