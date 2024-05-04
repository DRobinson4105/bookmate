from pyzbar.pyzbar import decode
from PIL import Image
from ultralytics import YOLO
from flask import Flask, request, jsonify
from flask_cors import CORS
from waitress import serve
from dotenv import load_dotenv
import os
import io

# init
model = YOLO("barcodeDetection.pt")
model.fuse()

app = Flask(__name__)
CORS(app)

load_dotenv()

print('API Ready')

@app.route("/api/getISBNs", methods=["POST"])
def get_isbns():
    file = request.files.get('file')
    image_stream = io.BytesIO(file.read())
    image = Image.open(image_stream)
    results = model(image, save=True)

    result = []

    boxes = [box.cpu().numpy().tolist() for box in results[0].obb.xyxyxyxy]
    boxes = [[[int(num) for num in pair] for pair in box] for box in boxes]

    for box in boxes:
        left = 1e9
        top = 1e9
        right = 0
        bottom = 0

        for pair in box:
            left = min(left, pair[0])
            top = min(top, pair[1])
            right = max(right, pair[0])
            bottom = max(bottom, pair[1])

        left, top, right, bottom = left-50, top-50, right+50, bottom+50
        decoded_objects = decode(image.crop((left, top, right, bottom)))

        if len(decoded_objects) == 0: continue

        isbn = decoded_objects[0].data.decode("utf-8")
        result.append([isbn, str(left), str(top), str(right), str(bottom)])
    
    return jsonify(result)
    # return "test"

if __name__ == "__main__":
    serve(app, host="127.0.0.1", port=os.environ["FLASK_PORT"])