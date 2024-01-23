

#FastAPI for combining the SAM and dumpster placement code
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import os
import cv2
import numpy as np
import math
import io
from uuid import uuid4
import uvicorn
from typing import List
import supervision as sv
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import matplotlib.pyplot as plt
import torch
# import asyncio


# Define the FastAPI app
app = FastAPI()

# Helper functions
def meters_to_yards(distance_in_meters):
    yards_conversion_factor = 1.09361
    distance_in_yards = distance_in_meters * yards_conversion_factor
    return distance_in_yards

def feet_to_meters(feet):
    meters = feet * 0.305
    return meters


def draw_rectangle(bg_image_path, org_path, height_in_feet, width_in_feet, x_coordinate, y_coordinate, tilt):
    x_coordinate = math.ceil(x_coordinate)  # Define your desired x-coordinate here
    y_coordinate = math.ceil(y_coordinate)

    height_in_feet = math.ceil(height_in_feet)  # Define your desired x-coordinate here
    width_in_feet = math.ceil(width_in_feet)




    bg_image = cv2.cvtColor(bg_image_path, cv2.COLOR_BGR2RGB)
    bg_image = cv2.resize(bg_image, (491, 255))

    org_image = cv2.cvtColor(org_path, cv2.COLOR_BGR2RGB)
    org_image = cv2.resize(org_image, (491, 255))


    while True:
      # height_in_feet = float(input("Enter height in feet (>=2): "))
      if height_in_feet >= 3:
          purple_area=0
          break
      elif height_in_feet==2:
          purple_area=2
          break
      else:
          break



    while True:       
     if width_in_feet >= 3:
        purple_area = 0
        break
     elif width_in_feet == 2:
        purple_area = 2
        break
     else:
        break

            

    # Read the background image
    bg_image = bg_image
    bg_image = cv2.resize(bg_image, (491, 255))

    object_height = feet_to_meters(height_in_feet)
    object_width = feet_to_meters(width_in_feet)



    image_height_pixels = bg_image.shape[0]
    image_width_pixels = bg_image.shape[1]

    object_height_meters = meters_to_yards(object_height)
    object_width_meters = meters_to_yards(object_width)

    object_height_pixels = int(object_height_meters * (image_height_pixels / 35))
    object_width_pixels = int(object_width_meters * (image_width_pixels / 90))

    rect_center = (x_coordinate, y_coordinate)
    rect_size = (object_width_pixels, object_height_pixels)
    angle = tilt

    #output_image = np.zeros_like(bg_image)

    if (rect_center[0] - object_width_pixels // 2 >= 0 and
        rect_center[0] + object_width_pixels // 2 <= image_width_pixels and
        rect_center[1] - object_height_pixels // 2 >= 0 and
        rect_center[1] + object_height_pixels // 2 <= image_height_pixels):
        start_x = rect_center[0] - object_width_pixels // 2
        end_x = rect_center[0] + object_width_pixels // 2
        start_y = rect_center[1] - object_height_pixels // 2
        end_y = rect_center[1] + object_height_pixels // 2

        roi = bg_image[start_y:end_y, start_x:end_x]
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_purple = np.array([120, 50, 50])
        upper_purple = np.array([150, 255, 255])

        purple_mask = cv2.inRange(roi_hsv, lower_purple, upper_purple)
        purple_area = np.sum(purple_mask == 255)+purple_area
        # roi_height_pixels = end_y - start_y
        # roi_width_pixels = end_x - start_x
        object_area_pixels = int(object_height_meters * object_width_meters * (image_height_pixels / 55) * (image_width_pixels / 100))

        if purple_area >= object_area_pixels:
            # print("Condition Met: Drawing Object")
            rect_points = cv2.boxPoints(((rect_center[0], rect_center[1]), (rect_size[0], rect_size[1]), angle))
            rect_points = np.int0(rect_points)

            org = org_image
            resized_image = cv2.resize(org, (491, 255))
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB) 
            cv2.drawContours(resized_image, [rect_points], 0, (0, 255, 0), 1, cv2.LINE_AA)
            status_message = "Condition Met: Drawing Object"
            # cv2_imshow(resized_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        else:
            status_message = "Condition Not Met: Not Drawing Object"
            org=org_image
            resized_image = cv2.resize(org, (491, 255))
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB) 

            return status_message
            # print("Condition Not Met: Not Drawing Object")
    else:
        status_message="Object cannot be placed entirely within the image"
        org=org_image
        resized_image = cv2.resize(org, (491, 255))
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB) 
        # print("Object cannot be placed entirely within the image")
    return resized_image,status_message



def process_image(org_image_cv2: np.ndarray) -> str:  # Change the input type to np.ndarray
    HOME = os.getcwd()
    CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
    MODEL_TYPE = "vit_h"
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam)

    # Assuming the image is already in BGR format, no need for cv2.cvtColor
    image_rgb = cv2.cvtColor(org_image_cv2, cv2.COLOR_BGR2RGB)
    sam_result = mask_generator.generate(image_rgb)

    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    detections = sv.Detections.from_sam(sam_result=sam_result)
    annotated_image = mask_annotator.annotate(scene=org_image_cv2.copy(), detections=detections)

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


@app.post("/process_image")
async def process_image_endpoint(org_image: UploadFile = File(...),
                        height_in_feet: float = 18,
                        width_in_feet: float = 7.5,
                        x_coordinate: float = 97,
                        y_coordinate: float = 133,
                        tilt: float = -15):
    if height_in_feet < 2:
        raise HTTPException(status_code=400, detail="Height should be greater than or equal to 2")
    
    if width_in_feet < 2:
        raise HTTPException(status_code=400, detail="Width should be greater than or equal to 2")

    if x_coordinate < 0 or y_coordinate < 0:
        raise HTTPException(status_code=400, detail="Coordinates should be greater than or equal to 0")

    if x_coordinate >= 450:
        raise HTTPException(status_code=400, detail="Coordinates should be less than or equal to 450")

    if y_coordinate >= 250:
        raise HTTPException(status_code=400, detail="Coordinates should be less than or equal to 250")
    
    org_image_data = await org_image.read()
    org_image_array = np.frombuffer(org_image_data, np.uint8)
    org_image_cv2 = cv2.imdecode(org_image_array, cv2.IMREAD_COLOR)  # Decode the uploaded image

    # Process the original image and get the modified image path
    modified_image_path = process_image(org_image_cv2)  
    
    # Read the processed image to create bg_image
    bg_image_cv2 = cv2.imread(modified_image_path)
    
    # Use bg_image and org_image_cv2 in draw_rectangle function to perform further operations
    processed_image, status = draw_rectangle(bg_image_cv2, org_image_cv2, height_in_feet, width_in_feet, x_coordinate, y_coordinate, tilt)
    
    if processed_image is not None:
        img_bytes = cv2.imencode(".png", processed_image)[1].tobytes()
        return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png", status_code=200, headers={"status": status})
    else:
        return {"status": status}

# Run the FastAPI application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)



