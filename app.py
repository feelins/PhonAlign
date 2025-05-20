#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask, render_template, request, send_from_directory, jsonify, send_file
import os
import uuid
import librosa
import parselmouth
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
from werkzeug.utils import secure_filename
import logging
import time
from logging import handlers
import soundfile as sf
import sys
import zipfile
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import json

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

def resample_audio(input_path, output_path, target_sr=16000, res_type='kaiser_best'):
    # 加载音频（保留原始采样率）
    y, orig_sr = librosa.load(input_path, sr=None)
    
    # 重采样
    y_resampled = librosa.resample(y, orig_sr=orig_sr, 
                                 target_sr=target_sr,
                                 res_type=res_type)
    
    # 保存
    sf.write(output_path, y_resampled, target_sr)
    
    return y_resampled, target_sr

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_spectrogram(audio_path, output_path):
    """Generate spectrogram and save to file"""
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
    """Visualize TextGrid with waveform"""
    try:
        # Read audio file
        snd = parselmouth.Sound(audio_path)
        
        # Read TextGrid file
        textgrid = parselmouth.Data.read(textgrid_path)
        
        plt.figure(figsize=(12, 8))
        
        # Plot waveform
        plt.subplot(2, 1, 1)
        plt.plot(snd.xs(), snd.values.T)
        plt.xlim([snd.xmin, snd.xmax])
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("Waveform")
        
        # Plot TextGrid
        plt.subplot(2, 1, 2)
        
        # Get number of tiers
        n_tiers = parselmouth.praat.call(textgrid, "Get number of tiers")
        
        # Create positions for each tier
        y_positions = list(range(n_tiers))
        y_ticks = []
        y_labels = []
        
        for tier_idx in range(1, n_tiers + 1):
            # tier = textgrid.tier(tier_idx)
            tier_name = parselmouth.praat.call(textgrid, "Get tier name", tier_idx)
            y_ticks.append(tier_idx - 1)
            y_labels.append(tier_name)
            
            # Handle interval tiers
            # if tier.is_interval_tier():
            intervals = parselmouth.praat.call(textgrid, "Get number of intervals", tier_idx)
            for interval in range(1, intervals + 1):
                label = parselmouth.praat.call(textgrid, "Get label of interval", tier_idx, interval).strip()
                start = parselmouth.praat.call(textgrid, "Get start point", tier_idx, interval)
                end = parselmouth.praat.call(textgrid, "Get end point", tier_idx, interval)
                
                plt.hlines(y=tier_idx-1, xmin=start, xmax=end, 
                            linewidth=40, color='dodgerblue', alpha=0.7)
                plt.text((start + end)/2, tier_idx-1, label, 
                        ha='center', va='center',fontsize=12,
                        bbox=dict(facecolor='white', alpha=0.8, pad=1))
                # 在每个interval的结尾处添加红色垂直线
                if label == '[SIL]':
                    plt.vlines(x=start, ymin=tier_idx-1.5, ymax=tier_idx-0.5,  # 调整ymin/ymax可以控制线的高度
                            colors='red', linewidth=2, linestyles='solid')
                    plt.vlines(x=end, ymin=tier_idx-1.5, ymax=tier_idx-0.5,  # 调整ymin/ymax可以控制线的高度
                            colors='red', linewidth=2, linestyles='solid')
        
        plt.yticks(y_ticks, y_labels)
        plt.xlabel("Time (s)")
        plt.ylabel("Tiers")
        plt.xlim([snd.xmin, snd.xmax])
        plt.ylim([-0.5, n_tiers - 0.5])
        plt.title("TextGrid Alignment")
        plt.grid(True, axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error visualizing TextGrid: {str(e)}")
        raise

def perform_alignment(audio_path, text_content, output_path, logger=None):
    """Perform audio-text alignment"""
    start_time = time.time()
    
    if logger:
        logger.info('Starting alignment process')
    
    try:
        # Initialize model
        from Charsiu import charsiu_forced_aligner, charsiu_predictive_aligner
        cur_root = os.path.dirname(os.path.realpath(__file__))
        pre_model_path = os.path.join(cur_root, 'pretrained_model')
        pre_model = 'pretrained_model/charsiu-zh_w2v2_tiny_fc_10ms'
        
        if logger:
            logger.info(f'Loading model from: {pre_model}')
        
        charsiu = charsiu_forced_aligner(aligner=pre_model, lang='zh')
        
        if logger:
            logger.info(f'Performing alignment for: {audio_path}')
            logger.info(f'Text content: {text_content[:50]}...')
        
        # Perform alignment
        charsiu.serve(audio=audio_path, text=text_content, save_to=output_path)
        
        if logger:
            logger.info(f'Alignment saved to: {output_path}')
        
        end_time = time.time()
        total_time = end_time - start_time
        
        if logger:
            logger.info(f'Alignment completed in {total_time:.2f} seconds')
        
        return True, total_time
    except Exception as e:
        if logger:
            logger.error(f'Alignment failed: {str(e)}')
        return False, 0

def process_batch_item(audio_path, text_path, output_dir, output_prefix, job_id, logger=None):
    """处理单个批量对齐任务"""
    try:
        # 读取文本内容
        with open(text_path, 'r', encoding='utf-8') as f:
            text_content = f.read().strip()
        
        if not text_content:
            raise ValueError("Text file is empty")
        
        # 准备输出文件名
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        output_name = f"{output_prefix}_{base_name}"
        output_textgrid = os.path.join(output_dir, f"{output_name}.TextGrid")
        
        # 重采样音频
        tmp_audio_path = os.path.join(output_dir, f"tmp_{base_name}.wav")
        resampled_audio, sr = resample_audio(audio_path, tmp_audio_path, 16000)
        
        # 执行对齐
        success, processing_time = perform_alignment(tmp_audio_path, text_content, output_textgrid, logger)
        
        if not success:
            raise ValueError("Alignment failed")
        
        # 生成可视化
        spectrogram_path = os.path.join(output_dir, f"{output_name}_spectrogram.png")
        generate_spectrogram(audio_path, spectrogram_path)
        
        textgrid_viz_path = os.path.join(output_dir, f"{output_name}_textgrid.png")
        visualize_textgrid(output_textgrid, audio_path, textgrid_viz_path)
        
        # 更新任务状态
        batch_jobs[job_id]['processed'] += 1
        batch_jobs[job_id]['results'].append({
            'status': 'success',
            'filename': base_name,
            'audio_file': os.path.basename(audio_path),
            'text_file': os.path.basename(text_path),
            'textgrid_file': f"{output_name}.TextGrid",
            'spectrogram': f"{output_name}_spectrogram.png",
            'textgrid_viz': f"{output_name}_textgrid.png",
            'processing_time': processing_time
        })
        
        return True
    except Exception as e:
        batch_jobs[job_id]['processed'] += 1
        batch_jobs[job_id]['error_count'] += 1
        batch_jobs[job_id]['results'].append({
            'status': 'error',
            'filename': os.path.splitext(os.path.basename(audio_path))[0],
            'error': str(e)
        })
        return False

def process_batch_zip(zip_path, output_dir, output_prefix, job_id, logger=None):
    """处理ZIP格式的批量上传"""
    try:
        # 解压ZIP文件
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        # 收集音频和文本文件
        audio_files = []
        text_files = []
        
        for root, _, files in os.walk(output_dir):
            for file in files:
                file_lower = file.lower()
                if file_lower.endswith(('.wav', '.mp3')):
                    audio_files.append(os.path.join(root, file))
                elif file_lower.endswith('.txt'):
                    text_files.append(os.path.join(root, file))
        
        # 创建文件对 (音频文件 -> 对应的文本文件)
        file_pairs = []
        for audio_file in audio_files:
            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            matching_text = next((t for t in text_files 
                               if os.path.splitext(os.path.basename(t))[0] == base_name), None)
            if matching_text:
                file_pairs.append((audio_file, matching_text))
            else:
                batch_jobs[job_id]['error_count'] += 1
                batch_jobs[job_id]['results'].append({
                    'status': 'error',
                    'filename': base_name,
                    'error': f"No matching text file found for {audio_file}"
                })
        
        # 更新总任务数
        batch_jobs[job_id]['total'] = len(file_pairs)
        
        # 使用线程池处理文件对
        futures = []
        for audio_path, text_path in file_pairs:
            future = executor.submit(
                process_batch_item, 
                audio_path, text_path, output_dir, output_prefix, job_id, logger
            )
            futures.append(future)
        
        # 等待所有任务完成
        for future in futures:
            future.result()
        
        # 创建结果ZIP
        result_zip_path = os.path.join(output_dir, f"{output_prefix}_results.zip")
        with zipfile.ZipFile(result_zip_path, 'w') as zipf:
            for root, _, files in os.walk(output_dir):
                for file in files:
                    if file.endswith(('.TextGrid', '.png')):
                        zipf.write(os.path.join(root, file), file)
        
        batch_jobs[job_id]['result_zip'] = os.path.basename(result_zip_path)
        batch_jobs[job_id]['status'] = 'completed'
        
    except Exception as e:
        batch_jobs[job_id]['status'] = 'failed'
        batch_jobs[job_id]['message'] = str(e)
        if logger:
            logger.error(f"Batch processing failed: {str(e)}")


def cleanup_temp_files():
    """清理过期的临时文件"""
    batch_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'batch')
    if not os.path.exists(batch_dir):
        return
    
    now = time.time()
    for job_id in os.listdir(batch_dir):
        job_dir = os.path.join(batch_dir, job_id)
        if os.path.isdir(job_dir):
            # 检查目录是否过期
            dir_time = os.path.getmtime(job_dir)
            if now - dir_time > app.config['BATCH_TEMP_EXPIRE']:
                try:
                    shutil.rmtree(job_dir)
                    # 同时从内存中删除任务记录
                    if job_id in batch_jobs:
                        del batch_jobs[job_id]
                except Exception as e:
                    print(f"Failed to cleanup {job_dir}: {str(e)}")

cur_root = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(cur_root, 'src'))


# 设置matplotlib使用支持中文的字体
try:
    # 尝试使用系统已有的中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows
    # plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac
    # plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']  # Linux
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except:
    # 如果找不到上述字体，尝试使用内置的DejaVu Sans
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 * 100  # 100MB limit for batch
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3'}
app.config['BATCH_TEMP_EXPIRE'] = 3600 * 6  # 6小时临时文件过期时间
app.config['MAX_WORKERS'] = 4  # 最大并发工作线程数

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 全局线程池
executor = ThreadPoolExecutor(max_workers=app.config['MAX_WORKERS'])

# 任务状态存储
batch_jobs = {}
logger = setup_logger('logs')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    
    if request.method == 'POST':
        # Check if file was uploaded
        if 'audio_file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        audio_file = request.files['audio_file']
        text_content = request.form.get('text_content', '')
        output_name = request.form.get('output_name', 'output')
        
        if audio_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(audio_file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Create unique ID and directory
        unique_id = str(uuid.uuid4())
        upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], unique_id)
        os.makedirs(upload_dir, exist_ok=True)
        
        try:
            # Save uploaded file
            audio_filename = secure_filename(audio_file.filename)
            audio_path = os.path.join(upload_dir, audio_filename)
            audio_file.save(audio_path)
            tmp_audio_path = os.path.join(upload_dir, 'tmp.wav')
            
            output_path = os.path.join(upload_dir, f'{output_name}.TextGrid')
            
            # Perform alignment
            logger.info('Starting alignment process')
            resampled_audio, sr = resample_audio(audio_path, tmp_audio_path, 16000)
            success, processing_time = perform_alignment(tmp_audio_path, text_content, output_path, logger)
            
            if not success:
                return jsonify({'error': 'Alignment failed'}), 500
            
            # Generate visualizations
            spectrogram_path = os.path.join(upload_dir, 'spectrogram.png')
            generate_spectrogram(audio_path, spectrogram_path)
            
            textgrid_viz_path = os.path.join(upload_dir, 'textgrid.png')
            visualize_textgrid(output_path, audio_path, textgrid_viz_path)
            
            # Prepare results
            results = {
                'status': 'success',
                'processing_time': f'{processing_time:.2f}',
                'audio_file': audio_filename,
                'textgrid_file': f'{output_name}.TextGrid',
                'spectrogram': 'spectrogram.png',
                'textgrid_viz': 'textgrid.png',
                'unique_id': unique_id
            }
            
            return jsonify(results)
            
        except Exception as e:
            logger.error(f'Error in processing: {str(e)}')
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/batch_align/', methods=['POST'])
def batch_align():
    """处理批量对齐请求"""
    try:
        # 创建唯一任务ID和目录
        job_id = str(uuid.uuid4())
        upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'batch', job_id)
        os.makedirs(upload_dir, exist_ok=True)
        
        # 初始化任务状态
        batch_jobs[job_id] = {
            'status': 'processing',
            'total': 0,
            'processed': 0,
            'success_count': 0,
            'error_count': 0,
            'start_time': time.time(),
            'results': [],
            'result_zip': None,
            'output_dir': upload_dir
        }
        
        # 获取表单数据
        output_prefix = request.form.get('output_prefix', 'batch_output')
        upload_type = request.form.get('uploadType', 'zip')

        logger.info(f'Starting batch alignment job {job_id}')
        
        # 处理ZIP上传
        if upload_type == 'zip' and 'batch_zip' in request.files:
            zip_file = request.files['batch_zip']
            if zip_file.filename == '':
                return jsonify({'error': 'No ZIP file selected'}), 400
            
            if not zip_file.filename.lower().endswith('.zip'):
                return jsonify({'error': 'Invalid file type, only ZIP allowed'}), 400
            
            zip_path = os.path.join(upload_dir, 'batch_files.zip')
            zip_file.save(zip_path)
            
            # 在后台处理ZIP文件
            executor.submit(
                process_batch_zip,
                zip_path, upload_dir, output_prefix, job_id, logger
            )
            
        # 处理分别上传的音频和文本文件
        elif upload_type == 'dir' and 'batch_files' in request.files:
            # 保存上传的文件
            uploaded_files = request.files.getlist('batch_files')
            
            # 创建文件对 (音频文件 -> 对应的文本文件)
            file_pairs = []
            audio_files = []
            text_files = []
            
            # 分类音频和文本文件
            for file in uploaded_files:
                filename = secure_filename(file.filename)
                filepath = os.path.join(upload_dir, filename)
                file.save(filepath)
                
                if filename.lower().endswith(('.wav', '.mp3')):
                    audio_files.append(filepath)
                elif filename.lower().endswith('.txt'):
                    text_files.append(filepath)
            
            # 配对文件
            for audio_path in audio_files:
                base_name = os.path.splitext(os.path.basename(audio_path))[0]
                matching_text = next((t for t in text_files 
                                   if os.path.splitext(os.path.basename(t))[0] == base_name), None)
                if matching_text:
                    file_pairs.append((audio_path, matching_text))
                else:
                    batch_jobs[job_id]['error_count'] += 1
                    batch_jobs[job_id]['results'].append({
                        'status': 'error',
                        'filename': base_name,
                        'error': f"No matching text file found for {audio_path}"
                    })
            
            # 更新总任务数
            batch_jobs[job_id]['total'] = len(file_pairs)
            
            # 使用线程池处理文件对
            for audio_path, text_path in file_pairs:
                executor.submit(
                    process_batch_item, 
                    audio_path, text_path, upload_dir, output_prefix, job_id, logger
                )
        
        else:
            return jsonify({'error': 'Invalid upload type or missing files'}), 400
        
        return jsonify({
            'status': 'processing',
            'job_id': job_id,
            'message': 'Batch processing started'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/batch_status/<job_id>')
def batch_status(job_id):
    """获取批量任务状态"""
    if job_id not in batch_jobs:
        return jsonify({'error': 'Invalid job ID'}), 404
    
    job = batch_jobs[job_id]
    
    # 计算进度百分比
    progress = 0
    if job['total'] > 0:
        progress = min(100, int((job['processed'] / job['total']) * 100))
    
    # 准备响应数据
    response = {
        'status': job['status'],
        'progress': progress,
        'total': job['total'],
        'processed': job['processed'],
        'success_count': len([r for r in job['results'] if r['status'] == 'success']),
        'error_count': len([r for r in job['results'] if r['status'] == 'error']),
        'latest_results': job['results'][-10:],  # 返回最近10个结果
        'processing_time': time.time() - job['start_time']
    }
    
    # 如果任务完成，添加结果ZIP信息
    if job['status'] in ['completed', 'success'] and job.get('result_zip'):
        response['result_zip'] = job['result_zip']
    
    # 如果所有任务已完成但状态未更新（保险机制）
    # if job['status'] == 'processing' and job['processed'] >= job['total']:
    #     job['status'] = 'completed'
    #     response['status'] = 'completed'
    
    return jsonify(response)

@app.route('/download_batch/<job_id>/<filename>')
def download_batch(job_id, filename):
    """下载批量处理结果"""
    if job_id not in batch_jobs:
        return jsonify({'error': 'Invalid job ID'}), 404
    
    job_dir = batch_jobs[job_id]['output_dir']
    return send_from_directory(job_dir, filename, as_attachment=True)

@app.route('/calculate_duration/', methods=['POST'])
def calculate_duration():
    """计算目录下所有音频文件的时长"""
    try:
        directory_path = request.form.get('directory_path', '').strip()
        
        if not directory_path:
            return jsonify({"status": "error", "message": "目录路径不能为空"})
        
        if not os.path.isdir(directory_path):
            return jsonify({"status": "error", "message": "目录不存在或不可访问"})
        
        # 支持的音频格式
        audio_extensions = ('.wav', '.mp3', '.ogg', '.flac')
        total_seconds = 0
        file_count = 0
        error_files = []
        
        start_time = time.time()
        logger.info('processing ' + directory_path)
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith(audio_extensions):
                    file_path = os.path.join(root, file)
                    try:
                        # 使用librosa获取音频时长
                        duration = librosa.get_duration(path=file_path)
                        total_seconds += duration
                        file_count += 1
                    except Exception as e:
                        error_files.append({
                            "filename": file_path,
                            "error": str(e)
                        })
                        logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
        
        # 计算总小时数
        total_hours = total_seconds / 3600
        
        return jsonify({
            "status": "success",
            "directory_path": directory_path,
            "total_duration_seconds": total_seconds,
            "total_duration_hours": total_hours,
            "file_count": file_count,
            "error_files": error_files,
            "processing_time": time.time() - start_time
        })
        
    except Exception as e:
        logger.error(f"计算时长时出错: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/download/<unique_id>/<filename>')
def download_file(unique_id, filename):
    return send_from_directory(
        os.path.join(app.config['UPLOAD_FOLDER'], unique_id),
        filename,
        as_attachment=True
    )



if __name__ == '__main__':
    # 启动时清理旧文件
    # cleanup_temp_files()
    app.run(host='127.0.0.1', port=8000, debug=True)