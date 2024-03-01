from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Directory to save images
SAVE_IMAGE_DIR = 'received_images'

# Create the directory if it doesn't exist
os.makedirs(SAVE_IMAGE_DIR, exist_ok=True)

@app.post('/anpr')
def anpr():
    try:
        print("received")
        # Read image from request
        file = request.files['image']
        img_stream = file.stream
        img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Get image height and width
        height, width = img.shape[:2]

        # Rotate the image
        center = (width // 2, height // 2)
        angle = -90  # Rotate 90 degrees clockwise
        scale = 1.0
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
        # Image.

        # Save the received image
        image_path = os.path.join(SAVE_IMAGE_DIR, 'received_image.jpg')
        cv2.imwrite(image_path, rotated_img)
        
        

        # Apply image processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        min_intensity = np.min(gray)
        max_intensity = np.max(gray)
        new_min = 0
        new_max = 255
        stretched = ((gray - min_intensity) / (max_intensity - min_intensity)) * (new_max - new_min) + new_min
        stretched = stretched.astype(np.uint8)
        kernal = np.ones((1, 1), np.uint8)
        image = cv2.dilate(stretched, kernal, iterations=1)
        kernal = np.ones((1, 1), np.uint8)
        image = cv2.erode(image, kernal, iterations=1)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernal)
        image = cv2.medianBlur(image, 3)
        img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Load the models
        model_LP = YOLO('best_lp.pt')  # licence plate model
        model_OCR = YOLO('best_ocr.pt')  # OCR model

        results_LP = model_LP(img, conf=0.5)

        for r1 in results_LP:
            pass

        if r1.boxes.cls.numel() != 0:
            x1, y1, x2, y2 = r1.boxes.numpy().xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            roi = img[y1:y2, x1:x2]

            results_OCR = model_OCR(roi, conf=0.41)

            ids = []
            positions = []

            # Process results list
            for r in results_OCR:
                pass

            for i in range(len(r.boxes.cls)):
                x = int(r.boxes.xyxy[i][0])
                positions.append(x)

            for id in r.boxes.cls:
                ids.append(int(id))

            if len(ids) != 0:
                arranged_lists = sorted(zip(positions, ids))
                sorted_positions, sorted_ids = zip(*arranged_lists)

                plate = ''
                for j in sorted_ids:
                    plate += r.names[j]

                return jsonify({'licence_plate': plate})
        else:
            return jsonify({'message': 'No Licence Plate Detected'}), 500
    except Exception as e:
        print(e)
        return jsonify({'message': 'Internal Server Error: No Licence Plate Detected', 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, host="0.0.0.0")
