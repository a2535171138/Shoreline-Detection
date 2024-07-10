from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from datetime import datetime
import logging
from werkzeug.utils import secure_filename
import os
from predict import Dexined_predict  # 确保这个导入是正确的
import json

app = Flask(__name__)
CORS(app)

# 配置
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传大小为16MB
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.jpeg']
app.config['CHECKPOINT_PATH'] = "/home/yiting/coaste-detect/backend/29_model.pth"
app.config['THRESHOLD'] = 200  # 可以根据需要调整阈值

# 配置日志
logging.basicConfig(level=logging.DEBUG)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not os.path.splitext(file.filename)[1].lower() in app.config['UPLOAD_EXTENSIONS']:
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        # 读取图像
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # 使用预测函数
        binary_result, color_result, pixels_result = Dexined_predict(
            image,
            app.config['CHECKPOINT_PATH'],
            app.config['THRESHOLD']
        )

        # 将结果转换为base64编码
        _, binary_buffer = cv2.imencode('.png', binary_result)
        binary_encoded = base64.b64encode(binary_buffer).decode('utf-8')

        # 将 BGR 转换为 RGB
        color_result = cv2.cvtColor(color_result, cv2.COLOR_BGR2RGB)
        _, color_buffer = cv2.imencode('.png', color_result)
        color_encoded = base64.b64encode(color_buffer).decode('utf-8')

        processing_time = datetime.utcnow().isoformat() + 'Z'

        # 如果 pixels_result 是 numpy 数组，转换为列表
        if isinstance(pixels_result, np.ndarray):
            pixels_result = pixels_result.tolist()

        result = {
            'binary_result': binary_encoded,
            'color_result': color_encoded,
            'pixels_result': pixels_result,
            'processingTime': processing_time
        }

        return json.dumps(result, cls=NumpyEncoder), 200, {'Content-Type': 'application/json'}

    except Exception as e:
        app.logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)