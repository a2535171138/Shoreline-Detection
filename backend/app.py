from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import base64
from datetime import datetime
import logging
from werkzeug.utils import secure_filename
import os
from uaed_predict import UAED_predict
from quality import evaluate_image_quality
from classify import classify_image
from PIL import Image
import json
import io
import zipfile


app = Flask(__name__)
CORS(app)

# 配置
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传大小为16MB
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.jpeg']
app.config['THRESHOLD'] = 200  # 可以根据需要调整阈值
app.config['CLASSIFICATION_MODEL_PATH'] = "/app/backend/coast_classifier.pth"
app.config['ENABLE_QUALITY_CHECK'] = False

# 模型路径配置
# MODEL_PATHS = {
#     'General': "/app/backend/General.pth",
#     'Narrabeen': "/app/backend/Narrabeen.pth",
#     'Gold Coast': "/app/backend/GoldCoast.pth",
#     'CoastSnap': "/app/backend/CoastSnap.pth"
# }

MODEL_PATHS = {
    'General': "/home/yiting/coaste-detect/backend/General.pth",
    'Narrabeen': "/home/yiting/coaste-detect/backend/Narrabeen.pth",
    'Gold Coast': "/home/yiting/coaste-detect/backend/GoldCoast.pth",
    'CoastSnap': "/home/yiting/coaste-detect/backend/CoastSnap.pth"
}
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


# 全局变量来存储处理结果
processed_results = []


@app.route('/toggle_quality_check', methods=['POST'])
def toggle_quality_check():
    app.config['ENABLE_QUALITY_CHECK'] = not app.config['ENABLE_QUALITY_CHECK']
    return jsonify({'enabled': app.config['ENABLE_QUALITY_CHECK']})


@app.route('/predict/<scene>', methods=['POST'])
def predict_route(scene):
    app.logger.info(f"Quality check enabled: {app.config['ENABLE_QUALITY_CHECK']}")
    app.logger.info(f"Selected scene: {scene}")

    if scene not in MODEL_PATHS:
        return jsonify({'error': 'Invalid scene name'}), 400

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

        if app.config['ENABLE_QUALITY_CHECK']:
            # 质量控制
            quality_result = evaluate_image_quality(image)
            if quality_result != 0:
                quality_messages = {1: "Low Contrast", 2: "Underexposed", 3: "Overexposed"}
                return jsonify({
                    'error': f'Image quality check attention: {quality_messages.get(quality_result, "Unknown issue")}'}), 400

            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            classification_result = classify_image(pil_image, app.config['CLASSIFICATION_MODEL_PATH'])
            if classification_result != 1:  # 假设1表示海岸线图片
                return jsonify({'error': 'Image is not classified as a coastline'}), 400

        # 使用预测函数
        model_path = MODEL_PATHS[scene]
        binary_result, color_result, pixels_result, confidence = UAED_predict(
            image,
            model_path,
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

        result = {
            'filename': secure_filename(file.filename),
            'binary_result': binary_encoded,
            'color_result': color_encoded,
            'pixels_result': pixels_result,
            'processingTime': processing_time,
            'confidence': confidence
        }

        # 添加结果到全局列表
        global processed_results
        processed_results.append(result)

        app.logger.info(f"Returning result with confidence: {confidence}")

        return json.dumps(result, cls=NumpyEncoder), 200, {'Content-Type': 'application/json'}

    except Exception as e:
        app.logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/download_all', methods=['GET'])
def download_all():
    if not processed_results:
        return jsonify({'error': 'No results to download'}), 400
    csv_content = "path,rectified site,camera,type,obstructi,downward,low,shadow,label\n"  # CSV 头部
    for item in processed_results:
        # 使用文件名作为路径
        path = item['filename']
        # 使用像素结果作为 MULTILINESTRING 数据
        wkt = item['pixels_result']
        # 其他列留空
        csv_content += f"{path},,,,,,,,{wkt}\n"

    memory_file = io.BytesIO()
    memory_file.write(csv_content.encode())
    memory_file.seek(0)

    return send_file(
        memory_file,
        mimetype='text/csv',
        as_attachment=True,
        download_name='all_results.csv'
    )


@app.route('/download_all/<type>', methods=['GET'])
def download_all_type(type):
    if not processed_results:
        return jsonify({'error': 'No results to download'}), 400

    if type == 'pixels' or type == 'all':
        csv_content = "path,rectified site,camera,type,obstructi,downward,low,shadow,label\n"
        for item in processed_results:
            filename = item['filename']
            pixels_result = item['pixels_result']
            csv_content += f"{filename},,,,,,,,{pixels_result}\n"

        memory_file = io.BytesIO()
        memory_file.write(csv_content.encode('utf-8'))
        memory_file.seek(0)

        if type == 'pixels':
            return send_file(
                memory_file,
                mimetype='text/csv',
                as_attachment=True,
                download_name='all_pixels_results.csv'
            )

    if type == 'all' or type in ['binary', 'color']:
        zip_memory_file = io.BytesIO()
        with zipfile.ZipFile(zip_memory_file, 'w') as zf:
            if type == 'all' or type == 'pixels':
                zf.writestr('all_pixels_results.csv', csv_content)

            for item in processed_results:
                if type == 'binary' or type == 'all':
                    binary_data = base64.b64decode(item['binary_result'])
                    zf.writestr(f"{item['filename']}_binary.png", binary_data)
                if type == 'color' or type == 'all':
                    color_data = base64.b64decode(item['color_result'])
                    zf.writestr(f"{item['filename']}_color.png", color_data)

        zip_memory_file.seek(0)
        return send_file(
            zip_memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'all_{type}_results.zip'
        )


@app.route('/clear_results', methods=['POST'])
def clear_results():
    global processed_results
    processed_results = []
    return jsonify({'message': 'All results cleared'}), 200


@app.route('/delete_result/<filename>', methods=['DELETE'])
def delete_result(filename):
    global processed_results
    processed_results = [result for result in processed_results if result['filename'] != filename]
    return jsonify({'message': f'Result for {filename} deleted successfully'}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
