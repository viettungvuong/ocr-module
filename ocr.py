from fastapi import FastAPI, UploadFile, File
from pdf2image import convert_from_bytes
from PIL import Image
from surya.layout import LayoutPredictor
from surya.texify import TexifyPredictor
import shutil
import json
import re
import requests
import base64
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow React frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

layout_predictor = LayoutPredictor()
texify_predictor = TexifyPredictor()


def parse_bbox_data(input_text):
    """Extract bounding box details from Surya layout output."""
    bbox_pattern = r"LayoutBox\(polygon=(\[\[.*?\]\]).*?confidence=([\d\.]+).*?label='(.*?)'.*?position=(\d+).*?top_k=({.*?}).*?bbox=(\[.*?\])"
    bboxes_data = re.findall(bbox_pattern, input_text, re.DOTALL)

    bboxes = []
    for polygon_str, confidence, label, position, top_k_str, bbox_str in bboxes_data:
        polygon = json.loads(polygon_str.replace("'", '"'))
        confidence = float(confidence)
        position = int(position)
        top_k = json.loads(top_k_str.replace("'", '"'))
        bbox = json.loads(bbox_str)

        bboxes.append(
            {
                "polygon": polygon,
                "confidence": confidence,
                "label": label,
                "position": position,
                "top_k": top_k,
                "bbox": bbox,
            }
        )

    return {"bboxes": bboxes}


def clean_latex(latex):
    """Remove HTML-like <math>...</math> tags from LaTeX expressions."""
    cleaned_latex = re.sub(r"<math.*?>(.*?)</math>", r"\1", latex, flags=re.DOTALL)
    return cleaned_latex


@app.post("/extract-math/")
async def extract_math(file: UploadFile = File(...)):
    """Process uploaded PDF, detect equations, and extract LaTeX."""

    # Save the uploaded file temporarily
    temp_pdf_path = f"temp_{file.filename}"
    with open(temp_pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Convert PDF to images
    images = convert_from_bytes(open(temp_pdf_path, "rb").read())

    # Layout Prediction
    layout_predictions = layout_predictor(images)

    # Parse bounding boxes
    pages = [parse_bbox_data(str(pred)) for pred in layout_predictions]

    # Find math regions
    math_regions = {}
    for i, page in enumerate(pages):
        for bbox in page["bboxes"]:
            if bbox["label"] in ["TextInLineMath", "Equation"]:
                math_regions.setdefault(i, []).append(bbox)

    # Crop detected math regions
    cropped_images = []
    for page_idx, regions in math_regions.items():  # for every page
        page_image = images[page_idx]
        for region in regions:
            left, top, right, bottom = region["bbox"]
            cropped_image = page_image.crop((left, top, right, bottom))

            # Convert image to Base64
            buffered = BytesIO()
            cropped_image.save(buffered, format="PNG")
            encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

            info = (page_idx, region["bbox"])
            cropped_images.append((info, encoded_image))

    # Process extracted regions with TexifyPredictor
    results = []
    for info, encoded_image in cropped_images:
        # Decode the image back to PIL format for processing
        image_data = base64.b64decode(encoded_image)
        image = Image.open(BytesIO(image_data))

        result = texify_predictor.predict(image)
        latex = clean_latex(result.text)

        insert = {
            "page": info,
            "res": latex,
            "image": encoded_image,  # Returning the Base64 encoded image
        }
        results.append(insert)

    return {"extracted_latex": results}
