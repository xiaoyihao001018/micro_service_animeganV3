from flask import Flask, request, send_file
import onnxruntime as ort
import cv2
import numpy as np
import io
from PIL import Image
import os
import sys
import logging
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:4200"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# 添加OPTIONS请求处理
@app.route('/convert', methods=['OPTIONS'])
def handle_options():
    return '', 200

def get_resource_path(relative_path):
    """获取资源文件的绝对路径"""
    try:
        # PyInstaller创建临时文件夹,将路径存储在_MEIPASS中
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)

def load_model(model_path):
    try:
        # 获取模型文件的完整路径
        full_path = get_resource_path(model_path)
        logger.info(f"模型路径: {full_path}")
        
        # 检查文件是否存在
        if not os.path.exists(full_path):
            # 列出目录内容以进行调试
            dir_path = os.path.dirname(full_path)
            logger.info(f"目录内容 {dir_path}:")
            for file in os.listdir(dir_path):
                logger.info(f"- {file}")
            raise FileNotFoundError(f"模型文件不存在: {full_path}")
        
        # 加载模型
        session = ort.InferenceSession(
            full_path, 
            providers=['CPUExecutionProvider']
        )
        logger.info("模型加载成功！")
        return session
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise

def process_image(img, model_name):
    h, w = img.shape[:2]
    def to_8s(x):
        if 'tiny' in model_name:
            return 256 if x < 256 else x - x % 16
        else:
            return 256 if x < 256 else x - x % 8
    img = cv2.resize(img, (to_8s(w), to_8s(h)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/ 127.5 - 1.0
    return img

def convert_image(image_data, model):
    # 将图片数据转换为 OpenCV 格式
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 处理图片
    processed_img = process_image(img, "AnimeGANv3_PortraitSketch_25.onnx")
    processed_img = np.expand_dims(processed_img, axis=0)
    
    # 模型推理
    x = model.get_inputs()[0].name
    fake_img = model.run(None, {x: processed_img})
    
    # 后处理
    fake_img = (np.squeeze(fake_img[0]) + 1.) * 127.5
    fake_img = np.clip(fake_img, 0, 255).astype(np.uint8)
    fake_img = cv2.cvtColor(fake_img, cv2.COLOR_RGB2BGR)
    
    # 转换为字节流
    _, buffer = cv2.imencode('.png', fake_img)
    return io.BytesIO(buffer.tobytes())

@app.route('/convert', methods=['POST'])
def convert():
    try:
        # 调试信息
        logger.info(f"收到请求，Content-Type: {request.content_type}")
        logger.info(f"请求文件: {request.files}")
        
        if 'image' not in request.files:
            logger.warning("没有找到图片文件")
            return {'error': 'No image uploaded'}, 400
        
        file = request.files['image']
        if file.filename == '':
            logger.warning("文件名为空")
            return {'error': 'No selected file'}, 400
        
        # 检查文件类型
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            logger.warning(f"不支持的文件类型: {file.filename}")
            return {'error': 'Unsupported file type'}, 400
        
        # 读取和处理图片
        image_data = file.read()
        if not image_data:
            logger.warning("图片数据为空")
            return {'error': 'Empty file'}, 400
            
        logger.info(f"开始处理图片，大小: {len(image_data)} bytes")
        result = convert_image(image_data, model)
        logger.info("图片处理完成")
        
        return send_file(
            result,
            mimetype='image/png',
            as_attachment=True,
            download_name='anime_style.png'
        )
        
    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        return {'error': str(e)}, 500

# 添加全局CORS处理
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:4200')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

if __name__ == '__main__':
    try:
        model = load_model("AnimeGANv3_PortraitSketch_25.onnx")
        # 添加debug=True以查看更多信息
        app.run(host='0.0.0.0', port=8602, debug=True)
    except Exception as e:
        logger.error(f"启动失败: {str(e)}")
        input("按任意键退出...")