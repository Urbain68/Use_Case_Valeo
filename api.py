from fastapi import FastAPI, HTTPException
from transformers import pipeline
from fine_tuning import fine_tune_model


app = FastAPI()

model, tokenizer = fine_tune_model(quantize=True    )

# Load fine-tuned model and tokenizer
model_pipeline = pipeline('text-classification', model=model, tokenizer=tokenizer)

@app.post("/predict/")
async def predict(text: str):
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Model pipeline is not initialized")

    # Perform the prediction
    prediction = model_pipeline(text)
    return {"prediction": prediction}