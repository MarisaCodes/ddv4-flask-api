from io import BytesIO
from base64 import b64decode
from flask import Flask, request, jsonify
import keras
from PIL import Image
import numpy as np

model_path = keras.utils.get_file(
    "ddv4.h5", origin="https://github.com/MarisaCodes/ddv4/raw/main/model-resnet_custom_v4.h5")

model = keras.models.load_model(model_path)


def process_pimg(pimg: Image.Image):
    (w, h) = pimg.size
    img = Image.new(pimg.mode, (max(w, h), max(w, h)))
    img.paste(pimg)
    img = img.resize((512, 512), Image.Resampling.LANCZOS)
    img = np.array(img, dtype=np.float64)
    img /= 255.0
    img = img.reshape((1, 512, 512, 3))
    return img


def classify(pimg: Image.Image):
    # loading tags
    with open("./tags/tags.txt", "r") as fd:
        all_tags = fd.readlines()
        all_tags = [tag.strip() for tag in all_tags]
    with open("./tags/tags-character.txt", "r") as fd:
        char_tags = fd.readlines()
        char_tags = [tag.strip() for tag in char_tags]

    # processing image
    img = process_pimg(pimg=pimg)

    # prediction
    pred = model.predict(img)

    # organizing general result and result for characters
    res = {}
    chars = {}
    for tag, score in zip(all_tags, pred[0]):
        if score > 0.5:
            res[tag] = f'{score:.3f}'
            try:
                i = char_tags.index(tag)
                chars[char_tags[i]] = f'{score:.3f}'
            except ValueError as e:
                pass
    return {"tags": res, "characters": chars}


app = Flask(__name__)


@app.post("/api")
def index():
    data = request.get_json()
    pimg = Image.open(BytesIO(b64decode(data["image"]))).convert("RGB")
    return jsonify(classify(pimg))
