
# fastapi for SAM model

from fastapi import FastAPI, File, UploadFile
from typing import List
import os
import cv2
import numpy as np
import supervision as sv
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import matplotlib.pyplot as plt
import torch
import asyncio
from uuid import uuid4 


samapi = FastAPI()
def process_image(image_path: str) -> str:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"File not found at: {image_path}")
    HOME = os.getcwd()
    CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
    MODEL_TYPE = "vit_h"
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam)

    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    sam_result = mask_generator.generate(image_rgb)

    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    detections = sv.Detections.from_sam(sam_result=sam_result)
    annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

    masks = [
        mask['segmentation']
        for mask
        in sorted(sam_result, key=lambda x: x['area'], reverse=True)
    ]
    composite_mask = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        composite_mask |= mask

    background_mask = ~composite_mask
    modified_image = annotated_image.copy()
    background_color = [0, 0, 0]

    modified_image[composite_mask] = background_color
    modified_image[background_mask] = [128, 0, 128]  # Non-masked color

    modified_image_path = os.path.join(HOME, "data", "modified_image.png")
    cv2.imwrite(modified_image_path, cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB))

    return modified_image_path

@samapi.post("/segment-image/")
async def segment_image(image_files: List[UploadFile] = File(...)):
    for image_file in image_files:
        contents = await image_file.read()
        with open("temp_image.png", "wb") as temp_image:
            temp_image.write(contents)

    modified_image_path = process_image("temp_image.png")

    return {"modified_image_path": modified_image_path}
if __name__ == "__main__":
    uvicorn.run("samapi:app", host="0.0.0.0", port=8005)
