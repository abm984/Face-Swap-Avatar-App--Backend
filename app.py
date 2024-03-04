from datetime import datetime
import numpy as np
import os
import os.path as osp
import glob
import cv2
import insightface
import onnxruntime
from PIL import Image
import cv2
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import FileResponse
from pydantic import BaseModel
import base64
from io import BytesIO
import uvicorn
from fastapi import Depends, FastAPI
from mangum import Mangum
import boto3
import random
#from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import warnings
from fastapi.middleware.cors import CORSMiddleware
import json
from PIL import Image
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import matplotlib.pyplot as plt
import numpy as np

# Inside your try block


warnings.filterwarnings("ignore")
def save_base64_to_s3(base64_data, bucket_name, file_name):
    s3 = boto3.client(
        's3',
        aws_access_key_id="XXXXXXXXXXXXXXXXXXXX",
        aws_secret_access_key="nFb2/XXXXXXXXXXXXXXXXXXXX" # Add full S3 access to these keys
    )
    decoded_data = base64.b64decode(base64_data)

    try:
        s3.put_object(Body=decoded_data, Bucket=bucket_name, Key=file_name, ACL='public-read')
        
        #url = s3.generate_presigned_url(
        #ClientMethod='get_object', 
        #Params={'Bucket': bucket_name, 'Key': file_name},
        #ExpiresIn=21600)
        
        object_url = f"https://{bucket_name}.s3.amazonaws.com/{file_name}"
        return object_url
    except Exception as e:
        return f'Error uploading to S3: {str(e)}'


def s3_uploader(base64_data):
    body = base64_data
    # Generate a random timestamp within a reasonable range
    random_timestamp = datetime.fromtimestamp(random.randint(0, 2**31-1))
    
    # Format the timestamp as a string
    formatted_timestamp = random_timestamp.strftime("%Y%m%d%H%M%S")
    
    # Generate a random number to add uniqueness
    random_number = random.randint(1000, 9999)
    bucket_name = "ai-processed-images"  # Replace with your specific S3 bucket name
    file_name = f"uploaded-image-{formatted_timestamp}-{random_number}.png"  # Replace with your desired file name

    object_url = save_base64_to_s3(body, bucket_name, file_name)
    return object_url






app = FastAPI()
handler = Mangum(app)



from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:5175",
      "ANY_OR_ALL_front_end_url"  # Update this to your frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


api_key_1 = "a1b2c3d4e5"



class ImageInput(BaseModel):
    image_base64: str
    mask_base64: str 
@app.post("/process_images")
async def process_images(image: ImageInput, key: str = Body(embed=True)):
    if key != api_key_1:
        raise HTTPException(status_code=401, detail="Invalid API key")
    try:


        img_path = BytesIO(base64.b64decode(image.image_base64))
        mask_path = BytesIO(base64.b64decode(image.mask_base64))

        img1 = Image.open(img_path).convert("RGB")
        img2 = Image.open(mask_path).convert("RGB")
        img1_np = np.array(img1)
        img2_np = np.array(img2)
        app = FaceAnalysis(name='buffalo_l')
        app.prepare(ctx_id=0, det_size=(640, 640))
        swapper = insightface.model_zoo.get_model('inswapper_128.onnx',
                                        download=False,
                                        download_zip=False)

        face1 = app.get(img1_np)[0]
        face2 = app.get(img2_np)[0]
        img1_ = img1_np.copy()
        img2_ = img2_np.copy()
        #img1_ = swapper.get(img1_, face1, face2, paste_back=True)
        img2_ = swapper.get(img2_, face2, face1, paste_back=True)
        processed_image_io = BytesIO()
        Image.fromarray(img2_).save(processed_image_io, format="PNG")
        processed_image_b64 = base64.b64encode(processed_image_io.getvalue())

        public_url = s3_uploader(processed_image_b64)

        return JSONResponse(content={"message": "Image processed successfully", "result_image": public_url})
    except Exception as e:
        return JSONResponse(content={"error": f"Error processing image: {str(e)}"}, status_code=500)
if __name__ == "__main__":
#   uvicorn.run(app)
   uvicorn.run(app, host="127.0.0.1", port=8080)
