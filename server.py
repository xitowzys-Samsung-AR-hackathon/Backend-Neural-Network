from flask import Flask, request, jsonify
from PIL import Image
from pathlib import Path
import time
import base64
import os
from io import BytesIO
import glob

app = Flask(__name__)

TARGETS_DIR = "./targets/"

Path(TARGETS_DIR).mkdir(parents=True, exist_ok=True)

#initilize database and network here
path = glob.glob('./coam/pretrainedmodels/effnetB1_ep86.pth')[0]
coamModel = CoamModel(path)

#Create database
dataBase = DataBase(coamModel)
#load targets
dataBase.add_targets(TARGETS_DIR)


@app.route('/add', methods=['POST'])
def add_to_database():
    file = request.files['image']
    img = Image.open(file.stream)

    timestamp = str(time.time())
    filepath = TARGETS_DIR + timestamp + '.' + img.format
    img.save(filepath)

    dataBase.add_target(filepath)

    return {"msg": "success"}

@app.route('/find_matching', methods=['POST'])
def find_matching():
    file = request.files['image']
    img = Image.open(file.stream)
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

    return jsonify({
        'msg': 'success',
        'score': score,
        'img': data
    })


if __name__ == "__main__":
    app.run(debug=True)