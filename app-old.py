#!/usr/bin/env python
# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, send_file, jsonify, send_from_directory
import os
import sys
import logging
import time
import uuid
import shutil
import json
import parselmouth
import numpy as np
from werkzeug.utils import secure_filename
from matplotlib import pyplot as plt
import librosa
import librosa.display
import soundfile as sf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['STATIC_FOLDER'] = 'static'
app.secret_key = 'your-secret-key-here'

# 获取当前文件所在目录
cur_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cur_root, 'src'))

# 确保目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['STATIC_FOLDER'], 'images'), exist_ok=True)

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_alignment(input_wav_path: str, input_text: str, output_tg_path: str) -> dict:
    """执行对齐操作的Python函数"""
    try:
        from Charsiu import charsiu_forced_aligner
        
        start_time = time.time()
        logger.info(f"开始对齐处理: {input_wav_path}")
        
        # 初始化模型
        pre_model = 'pretrained_model/charsiu-zh_w2v2_tiny_fc_10ms'
        charsiu = charsiu_forced_aligner(aligner=pre_model, lang='zh')
        
        # 执行对齐
        charsiu.serve(audio=input_wav_path, text=input_text, save_to=output_tg_path)
        
        # 计算处理时间
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info(f"对齐完成，结果保存到: {output_tg_path}")
        logger.info(f"处理用时: {total_time:.2f}秒")
        
        return {
            "status": "success",
            "message": "对齐完成",
            "processing_time": total_time,
            "output_file": output_tg_path
        }
    except Exception as e:
        logger.error(f"对齐过程中发生错误: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

def visualize_alignment(audio_path, textgrid_path, output_image_path):
    """生成音频和TextGrid的可视化图像"""
    try:
        # 读取音频文件
        sound = parselmouth.Sound(audio_path)
        y, sr = librosa.load(audio_path)
        duration = len(y) / sr
        
        # 读取TextGrid文件
        tg = parselmouth.Data.read(textgrid_path)
        
        # 创建图形
        plt.figure(figsize=(14, 8))
        
        # 绘制波形图
        plt.subplot(2, 1, 1)
        librosa.display.waveshow(y, sr=sr, color='b')
        plt.title('Waveform with TextGrid Alignment')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        # 绘制频谱图
        plt.subplot(2, 1, 2)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram with TextGrid Alignment')
        
        # 在两张图上绘制相同的TextGrid标记
        for subplot in [1, 2]:
            plt.subplot(2, 1, subplot)
            
            # 绘制每个层级的边界
            for tier in range(1, tg.number_of_tiers + 1):
                tier_obj = tg.get_tier(tier)
                
                # 处理区间层级
                if tier_obj.is_interval_tier():
                    for interval in tier_obj:
                        plt.axvline(x=interval.xmin(), color='r', linestyle='--', alpha=0.7)
                        plt.axvline(x=interval.xmax(), color='r', linestyle='--', alpha=0.7)
                        plt.text((interval.xmin() + interval.xmax()) / 2, 
                                plt.ylim()[1] * 0.9 if subplot == 1 else 10000,
                                interval.text,
                                ha='center', va='center', 
                                bbox=dict(facecolor='white', alpha=0.7))
                
                # 处理点层级
                elif tier_obj.is_point_tier():
                    for point in tier_obj:
                        plt.axvline(x=point.time, color='g', linestyle='-', alpha=0.7)
                        plt.text(point.time, 
                                plt.ylim()[1] * 0.8 if subplot == 1 else 8000,
                                point.text,
                                ha='center', va='center',
                                bbox=dict(facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(output_image_path)
        plt.close()
        
        return True
    except Exception as e:
        logger.error(f"可视化过程中发生错误: {str(e)}")
        return False

@app.route('/', methods=['GET'])
def index():
    """主页面"""
    return render_template('upload_form.html')

@app.route('/align/', methods=['POST'])
def align_audio_text():
    """
    音频文本对齐接口
    """
    # 检查文件是否上传
    if 'audio_file' not in request.files:
        return jsonify({"status": "error", "message": "没有上传文件"}), 400
    
    audio_file = request.files['audio_file']
    text_content = request.form.get('text_content', '')
    output_filename = request.form.get('output_filename', 'output.TextGrid')
    
    # 验证输入
    if not audio_file.filename:
        return jsonify({"status": "error", "message": "没有选择文件"}), 400
    
    if not text_content:
        return jsonify({"status": "error", "message": "文本内容不能为空"}), 400
    
    # 生成唯一ID用于临时文件
    unique_id = str(uuid.uuid4())
    temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], unique_id)
    os.makedirs(temp_dir, exist_ok=True)
    
    # 保存上传的音频文件
    filename = secure_filename(audio_file.filename)
    input_wav_path = os.path.join(temp_dir, filename)
    audio_file.save(input_wav_path)
    
    # 设置输出文件路径
    output_tg_path = os.path.join(temp_dir, output_filename)
    
    # 执行对齐
    result = run_alignment(input_wav_path, text_content, output_tg_path)
    
    if result["status"] == "success":
        # 生成可视化图像
        # image_path = os.path.join(app.config['STATIC_FOLDER'], 'images', f'{unique_id}.png')
        # visualize_alignment(input_wav_path, output_tg_path, image_path)
        
        # 返回结果
        return jsonify({
            "status": "success",
            "message": "对齐完成",
            "audio_url": f"/uploads/{unique_id}/{filename}",
            "textgrid_url": f"/uploads/{unique_id}/{output_filename}",
            "image_url": f"/static/images/{unique_id}.png",
            "unique_id": unique_id
        })
    else:
        # 返回错误信息
        return jsonify(result), 400

@app.route('/visualize/', methods=['POST'])
def visualize():
    """生成可视化结果"""
    if 'audio_file' not in request.files or 'textgrid_file' not in request.files:
        return jsonify({"status": "error", "message": "缺少文件"}), 400
    
    audio_file = request.files['audio_file']
    textgrid_file = request.files['textgrid_file']
    
    # 保存临时文件
    unique_id = str(uuid.uuid4())
    temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], unique_id)
    os.makedirs(temp_dir, exist_ok=True)
    
    audio_path = os.path.join(temp_dir, secure_filename(audio_file.filename))
    textgrid_path = os.path.join(temp_dir, secure_filename(textgrid_file.filename))
    image_path = os.path.join(app.config['STATIC_FOLDER'], 'images', f'{unique_id}.png')
    
    audio_file.save(audio_path)
    textgrid_file.save(textgrid_path)
    
    # 生成可视化图像
    success = visualize_alignment(audio_path, textgrid_path, image_path)
    
    if success:
        return jsonify({
            "status": "success",
            "audio_url": f"/uploads/{unique_id}/{secure_filename(audio_file.filename)}",
            "textgrid_url": f"/uploads/{unique_id}/{secure_filename(textgrid_file.filename)}",
            "image_url": f"/static/images/{unique_id}.png",
            "unique_id": unique_id
        })
    else:
        return jsonify({"status": "error", "message": "生成可视化失败"}), 500

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """提供上传的文件"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)