#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask, render_template, request, send_from_directory, jsonify
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
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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
                            linewidth=10, color='dodgerblue', alpha=0.7)
                plt.text((start + end)/2, tier_idx-1, label, 
                        ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.8, pad=1))
        
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
        from Charsiu import charsiu_forced_aligner, charsiu_predictive_aligner
        # Initialize model
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
        
        return True
    except Exception as e:
        if logger:
            logger.error(f'Alignment failed: {str(e)}')
        return False

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
        
        # Setup logger
        log_dir = os.path.join('./', 'logs')
        logger = setup_logger(log_dir)
        
        try:
            # Save uploaded file
            audio_filename = secure_filename(audio_file.filename)
            audio_path = os.path.join(upload_dir, audio_filename)
            audio_file.save(audio_path)
            
            output_path = os.path.join(upload_dir, f'{output_name}.TextGrid')
            
            # Perform alignment
            logger.info('Starting alignment process')
            success = perform_alignment(audio_path, text_content, output_path, logger)
            
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

@app.route('/download/<unique_id>/<filename>')
def download_file(unique_id, filename):
    return send_from_directory(
        os.path.join(app.config['UPLOAD_FOLDER'], unique_id),
        filename,
        as_attachment=True
    )

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)