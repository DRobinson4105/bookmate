from pyzbar.pyzbar import decode
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from flask import Flask, request, jsonify
from flask_cors import CORS
from waitress import serve
from dotenv import load_dotenv
import os
import io
import base64

app = Flask(__name__)
CORS(app)

load_dotenv()

# get api directory
dir = os.path.dirname(os.path.abspath(__file__))

# ISBN Retrieval

model = YOLO(os.path.join(dir, "barcodeDetection.pt"))
model.fuse()

@app.route("/api/getISBNs", methods=["POST"])
def get_isbns():
    file = request.files.get('file')
    image_stream = io.BytesIO(file.read())
    image = Image.open(image_stream)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 72)
    results = model(image, save=True)

    result = []

    # get bounding box coordinates
    boxes = [box.cpu().numpy().tolist() for box in results[0].obb.xyxyxyxy]
    boxes = [[[int(num) for num in pair] for pair in box] for box in boxes]

    for box in boxes:
        left = 1e9
        top = 1e9
        right = 0
        bottom = 0

        # widen bounding box
        for pair in box:
            left = min(left, pair[0])
            top = min(top, pair[1])
            right = max(right, pair[0])
            bottom = max(bottom, pair[1])

        left, top, right, bottom = left-50, top-50, right+50, bottom+50

        # get isbn from cropped image
        decoded_objects = decode(image.crop((left, top, right, bottom)))

        if len(decoded_objects) == 0: continue
        
        # draw bounding box on image
        draw.rectangle((left, top, right, bottom), outline="black", fill=None, width=5)

        # display isbn on image
        isbn = decoded_objects[0].data.decode("utf-8")
        position = (left, top - 75)
        text = 'ISBN: ' + isbn
        left, top, right, bottom = draw.textbbox(position, text, font=font)
        draw.rectangle((left-5, top-5, right+5, bottom+5), fill="white")
        draw.text(position, text, fill="black", font=font)

        result.append(isbn)
    
    # save new image with isbns and bounding boxes
    img_io = io.BytesIO()
    image.save(img_io, 'JPEG')
    img_io.seek(0)
    img_io_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
    
    return jsonify({
        'image': f"data:image/jpeg;base64,{img_io_base64}",
        'isbns': result
    })

if __name__ == "__main__":
    print("API Ready")
    serve(app, host="127.0.0.1", port=os.environ["FLASK_PORT"])