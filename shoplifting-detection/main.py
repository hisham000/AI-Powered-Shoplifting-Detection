import io
from typing import Dict

from fastapi import FastAPI, File, UploadFile
from PIL import Image
from tensorflow.keras.models import load_model

# Create FastAPI instance
app = FastAPI()


@app.get("/health")
def health_check() -> Dict[str, str]:
    """
    Health check endpoint to verify if the service is running.
    """
    return {"status": "healthy"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, float]:
    """
    Predict the probability of shoplifting using the pre-trained model.

    Args:
        file (UploadFile): The uploaded image file.

    Returns:
        Dict[str, float]: Probability of shoplifting.
    """
    # Read the image from the uploaded file
    contents = await file.read()
    Image.open(io.BytesIO(contents))

    # Load the pre-trained model
    load_model("model.h5")

    # Preprocess the image and make predictions (dummy implementation)
    # In a real scenario, you would preprocess the image for your model
    # For now, we return a dummy prediction
    prediction = 0.85
    return {"probability_of_shoplifting": prediction}
