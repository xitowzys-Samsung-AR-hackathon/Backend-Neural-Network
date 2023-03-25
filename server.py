# from flask import Flask, request, jsonify
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from pathlib import Path
import time
import base64
import os
from io import BytesIO
import glob

from pymongo import MongoClient

from coam import DataBase, CoamModel

app = FastAPI()

client = MongoClient("mongodb://localhost:27017/")
db = client["mydatabase"]
collection = db["mycollection"]

TARGETS_DIR = "./targets/"

Path(TARGETS_DIR).mkdir(parents=True, exist_ok=True)

# initilize database and network here
path = glob.glob('./pretrainedmodels/effnetB1_ep86.pth')[0]
coamModel = CoamModel(path)

# Create database
dataBase = DataBase(coamModel)
# load targets
dataBase.add_targets(TARGETS_DIR)


@app.post('/add')
async def add_to_database(image: UploadFile = File(...)):
    img = Image.open(BytesIO(await image.read()))
    timestamp = str(time.time())
    filepath = TARGETS_DIR + timestamp + '.' + img.format
    img.save(filepath)
    dataBase.add_target(filepath)

    result = collection.insert_one({"filepath": filepath})
    doc_id = result.inserted_id

    return {
        "msg": "success",
        "id": str(doc_id)
    }


@app.post('/find_matching')
async def find_matching(image: UploadFile = File(...)):
    img = Image.open(BytesIO(await image.read()))
    timestamp = str(time.time())
    filepath = timestamp + '.' + img.format
    img.save(filepath)

    target_id, score = dataBase.get_best_matching(filepath, batch_size=4)
    best_match = dataBase.targets[target_id]

    buffered = BytesIO()
    best_match.save(buffered, format="PNG")
    data = base64.b64encode(buffered.getvalue()).decode()

    if os.path.exists(filepath):
        os.remove(filepath)

    filepath = dataBase.filepaths[target_id // dataBase.num_of_rot]

    doc = collection.find_one({"filepath": filepath})
    doc_id = doc["_id"]

    return {
        'msg': 'success',
        'score': score,
        'img_path': filepath,
        'id': str(doc_id)
    }


if __name__ == "__main__":
    app.run(debug=True)
