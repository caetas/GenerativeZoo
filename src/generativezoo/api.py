from os import getenv
from typing import List

import config
from fastapi import FastAPI, File, UploadFile
from loguru import logger
from pydantic import BaseModel

app = FastAPI(title="My API", version="0.1", description="Description of my API.")

# https://fastapi.tiangolo.com/tutorial/response-model/
# FIXME: this is a reference pydantic model to represent a prediction oioi


class Prediction(BaseModel):
    prediction: bool


@app.get("/")
def read_root():
    return {"message": "Welcome from the API."}


# Public endpoint
@app.get("/api/v1/healthcheck")
async def health_check():
    """Health check."""
    return {"message": "API is live!"}


@app.post("/predict/text")  # , response_model=Prediction) To document the type of the response
async def predict(
    text_file: UploadFile = File(...),
):
    """Predict with a text file input."""
    text_contents = await text_file.read()
    text_input = text_contents.decode("utf-8").split("\n")
    output = str(text_input)  # = invoke_my_predict_function(text_input)
    logger.info(output)
    return output


@app.post("/predict/image-list")
async def predict_images(
    image_list: List[UploadFile] = File(...),
):
    """Get annotations from image file"""
    for image_file in image_list:
        image_content = await image_file.read()
        image_content.process()
        """
        (image, height, width) = read_image(image_content)

        detections = get_inferences(image)

        output = annotations2json(
            image_name=file.filename,
            image_np=image,
            gt_boxes=detections["detection_boxes"],
            class_list=detections["detection_classes"].astype(str),
            normalized_coordinates=True,
        )

    logger.info(output)
    """
    return {"prediction": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        debug=False,
        workers=1,
    )
