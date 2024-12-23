from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import time

# Initialize FastAPI app
app = FastAPI()

# Load model and tokenizer
model = M2M100ForConditionalGeneration.from_pretrained('facebook/m2m100_418M')
tokenizer = M2M100Tokenizer.from_pretrained('facebook/m2m100_418M')

# Define request schema
class TranslationRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

# Define response schema
class TranslationResponse(BaseModel):
    translation: str
    time_taken: float

@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    start_time = time.time()

    try:
        # Set source language
        tokenizer.src_lang = request.source_lang

        # Encode the input text
        inputs = tokenizer(request.text, return_tensors="pt")

        # Generate translation
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[request.target_lang]
        )

        # Decode translation
        translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Calculate time taken
    end_time = time.time()
    time_taken = end_time - start_time

    return TranslationResponse(translation=translation, time_taken=time_taken)

# Run the app
# Use this command to run the server: uvicorn filename:app --reload