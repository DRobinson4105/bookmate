from pyzbar.pyzbar import decode
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

model = YOLO("best.pt")
model.fuse()

def get_isbns(image_path):
    results = model(image_path, save=True)
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    # font = ImageFont.truetype('arial.ttf', 56)
    font = ImageFont.load_default(size=56)
    isbns = []

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

        im1 = image.crop((left - 50, top - 50, right + 50, bottom + 50))

        decoded_objects = decode(im1)
        if len(decoded_objects) == 0: continue

        left, top, right, bottom = left-50, top-50, right+50, bottom+50
        draw.rectangle((left, top, right, bottom), outline="black", fill=None, width=5)

        isbn = decoded_objects[0].data.decode("utf-8")
        position = (left, top - 75)
        text = 'ISBN: ' + isbn
        left, top, right, bottom = draw.textbbox(position, text, font=font)
        draw.rectangle((left-5, top-5, right+5, bottom+5), fill="white")
        draw.text(position, text, fill="black", font=font)
        isbns.append(int(isbn))
    
    image.save('test_result.jpg')
    return isbns

if __name__ == "__main__":
    print(get_isbns("test.jpg"))