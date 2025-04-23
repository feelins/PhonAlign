#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask, render_template, request, send_from_directory
import os
import uuid
import librosa
import parselmouth
import matplotlib.pyplot as plt
import numpy as np
from werkzeug.utils import secure_filename
import logging
import time
from logging import handlers
from datetime import datetime
import soundfile as sf
from tqdm import tqdm

# 从Charsiu导入对齐功能
try:
    from Charsiu import charsiu_forced_aligner, charsiu_predictive_aligner
except ImportError:
    # 如果直接导入失败，尝试添加路径
    import sys
    cur_root = os.path.split(os.path.realpath(__file__))[0]
    sys.path.append(os.path.join(cur_root, 'src'))
    from Charsiu import charsiu_forced_aligner, charsiu_predictive_aligner

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3'}

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 设置日志
def setup_logger(log_dir):
    level = logging.INFO
    stamp = int(time.time())
    log_filename = 'log_' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(stamp)) + '.log'
    
    logger = logging.getLogger(log_filename)
    logger.setLevel(level)
    
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, log_filename)
    
    formatter = logging.Formatter('%(asctime)s - %(pathname)s - %(levelname)s: %(message)s')
    
    ch = logging.StreamHandler()
    ch.setLevel(level)
    
    fh = handlers.TimedRotatingFileHandler(
        filename=log_file_path,
        when='D',
        backupCount=0,
        encoding='utf-8')
    fh.setLevel(level)
    
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_spectrogram(audio_path, output_path):
    """生成频谱图并保存"""
    y, sr = librosa.load(audio_path)
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, y_axis='linear', x_axis='time', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def visualize_textgrid(textgrid_path, audio_path, output_path):
    """可视化TextGrid并保存"""
    try:
        snd = parselmouth.Sound(audio_path)
        tg = parselmouth.praat.call('Read from file', textgrid_path)
        
        plt.figure(figsize=(10, 6))
        
        # 绘制波形
        plt.subplot(2, 1, 1)
        plt.plot(snd.xs(), snd.values.T)
        plt.xlim([snd.xmin, snd.xmax])
        plt.xlabel("time [s]")
        plt.ylabel("amplitude")
        plt.title("Waveform")
        
        # 绘制TextGrid
        plt.subplot(2, 1, 2)
        # tiers = tg.get_number_of_tiers()
        tiers = parselmouth.praat.call(tg, "Get number of tiers")
        y_positions = list(range(1, tiers + 1))
        
        for tier in range(1, tiers + 1):
            # tier_name = tg.get_tier_name(tier)
            # intervals = tg.get_interval_texts(tier)
            intervals = parselmouth.praat.call(tg, "Get number of intervals", tier)
            
            for interval in range(1, intervals + 1):
                label = parselmouth.praat.call(tg, "Get label of interval", tier, interval).strip()
                start = parselmouth.praat.call(tg, "Get start point", tier, interval)
                end = parselmouth.praat.call(tg, "Get end point", tier, interval)
                # start = tg.get_start_time(tier, interval)
                # end = tg.get_end_time(tier, interval)
                # label = tg.get_label_of_interval(tier, interval)
                
                plt.hlines(y=tier, xmin=start, xmax=end, linewidth=10, color='blue')
                plt.text((start + end)/2, tier, label, ha='center', va='center', 
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        plt.yticks(y_positions, [parselmouth.praat.call(tg, "Get tier name", i) for i in range(1, tiers + 1)])
        plt.xlabel("time [s]")
        plt.ylabel("tiers")
        plt.xlim([snd.xmin, snd.xmax])
        plt.title("TextGrid Alignment")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    except Exception as e:
        print(f"Error visualizing TextGrid: {str(e)}")
        raise

def perform_alignment(audio_path, text_path, output_path, logger=None):
    """执行音频文本对齐"""
    start_time = time.time()
    
    if logger:
        logger.info('Starting alignment process')
    
    try:
        # 初始化模型
        cur_root = os.path.split(os.path.realpath(__file__))[0]
        pre_model_path = os.path.join(os.path.dirname(cur_root), 'pretrained_model')
        pre_model = 'pretrained_model/charsiu-zh_w2v2_tiny_fc_10ms'
        
        if logger:
            logger.info(f'Loading model from: {pre_model}')
        
        charsiu = charsiu_forced_aligner(aligner=pre_model, lang='zh')
        
        # 读取文本内容
        with open(text_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
        
        if logger:
            logger.info(f'Performing alignment for: {audio_path}')
            logger.info(f'Text content: {text_content[:50]}...')  # 只记录前50个字符
        
        # 执行对齐
        charsiu.serve(audio=audio_path, text=text_content, save_to=output_path)
        
        if logger:
            logger.info(f'Alignment saved to: {output_path}')
        
        end_time = time.time()
        total_time = end_time - start_time
        
        if logger:
            logger.info(f'Alignment completed in {total_time:.2f} seconds')
        
        return True
    except Exception as e:
        if logger:
            logger.error(f'Alignment failed: {str(e)}')
        return False

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 检查文件是否上传
        if 'audio_file' not in request.files:
            return render_template('index.html', error='No file uploaded')
        
        audio_file = request.files['audio_file']
        text_content = request.form.get('text_content', '')
        output_name = request.form.get('output_name', 'output')
        
        if audio_file.filename == '':
            return render_template('index.html', error='No selected file')
        
        if not allowed_file(audio_file.filename):
            return render_template('index.html', error='Invalid file type')
        
        # 创建唯一ID和目录
        unique_id = str(uuid.uuid4())
        upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], unique_id)
        os.makedirs(upload_dir, exist_ok=True)
        
        # 设置日志
        log_dir = os.path.join(upload_dir, 'logs')
        logger = setup_logger(log_dir)
        
        try:
            # 保存上传的文件
            audio_filename = secure_filename(audio_file.filename)
            audio_path = os.path.join(upload_dir, audio_filename)
            audio_file.save(audio_path)
            
            # 保存文本内容
            text_path = os.path.join(upload_dir, 'input.txt')
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            
            output_path = os.path.join(upload_dir, f'{output_name}.TextGrid')
            
            # 执行对齐
            logger.info('Starting alignment process')
            success = perform_alignment(audio_path, text_path, output_path, logger)
            
            if not success:
                return render_template('index.html', error='Alignment failed. Check logs for details.')
            
            # 生成可视化
            spectrogram_path = os.path.join(upload_dir, 'spectrogram.png')
            generate_spectrogram(audio_path, spectrogram_path)
            
            textgrid_viz_path = os.path.join(upload_dir, 'textgrid.png')
            visualize_textgrid(output_path, audio_path, textgrid_viz_path)
            
            # 准备结果
            results = {
                'audio_file': audio_filename,
                'textgrid_file': f'{output_name}.TextGrid',
                'spectrogram': 'spectrogram.png',
                'textgrid_viz': 'textgrid.png',
                'unique_id': unique_id
            }
            
            return render_template('index.html', results=results)
            
        except Exception as e:
            logger.error(f'Error in processing: {str(e)}')
            return render_template('index.html', error=f'Processing failed: {str(e)}')
    
    return render_template('index.html')

@app.route('/download/<unique_id>/<filename>')
def download_file(unique_id, filename):
    return send_from_directory(
        os.path.join(app.config['UPLOAD_FOLDER'], unique_id),
        filename,
        as_attachment=True
    )

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)