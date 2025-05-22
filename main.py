import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration, TextStreamer
import time
import asyncio
from typing import Optional

app = FastAPI(title="M2M100 Translation API")

# Load model and tokenizer
model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

class TranslationRequest(BaseModel):
    text: str
    source_language: str
    target_language: str

class CustomStreamer(TextStreamer):
    def __init__(self, tokenizer, skip_special_tokens: bool = True):
        super().__init__(tokenizer, skip_special_tokens)
        self.current_text = ""
    
    def put(self, value):
        if isinstance(value, str):
            self.current_text += value
        else:
            # Decode tensor values before concatenating
            decoded_text = self.tokenizer.decode(value, skip_special_tokens=True)
            self.current_text += decoded_text
        return value

async def translate_stream(text: str, src_lang: str, tgt_lang: str):
    try:
        tokenizer.src_lang = src_lang
        encoded = tokenizer(text, return_tensors="pt")
        
        # Create a custom streamer
        streamer = CustomStreamer(tokenizer, skip_special_tokens=True)
        
        # Generate with streaming
        generated = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.get_lang_id(tgt_lang),
            streamer=streamer,
            max_length=512,
            num_beams=1,
            return_dict_in_generate=True,
            output_scores=False
        )
        
        # Return the final translation
        final_translation = tokenizer.decode(generated.sequences[0], skip_special_tokens=True)
        return final_translation
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate/stream")
async def translate_text(request: TranslationRequest):
    """
    Translate text from source language to target language.
    """
    try:
        translation = await translate_stream(
            request.text,
            request.source_language,
            request.target_language
        )
        return {"translation": translation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to M2M100 Translation API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
