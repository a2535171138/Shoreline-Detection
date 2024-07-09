from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import torch
from model import predict, fill_holes, skeletonize
from datetime import datetime

app = Flask(__name__)
CORS(app)

def predict_image(image):
    checkpoint_path = "C:\\Users\\padra\\Newcoaste-detect\\backend\\29_model.pth"
    predicted_img = predict(checkpoint_path, image)
    # _, binary_image = cv2.threshold(predicted_img, 200, 255, cv2.THRESH_BINARY)
    # filled_image = fill_holes(binary_image)
    # skeleton = skeletonize(filled_image)
    return predicted_img

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    result = predict_image(image)

    _, buffer = cv2.imencode('.png', result)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    processing_time = datetime.utcnow().isoformat() + 'Z'

    return jsonify({'result': encoded_image, 'processingTime': processing_time})

if __name__ == '__main__':
    app.run(debug=True)
