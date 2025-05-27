import { pollBatchStatus, updateProgressUI } from '../services/utils.js';

export function initBatch() {
  const form = document.getElementById('batchForm');
  if (!form) return;

  // 修改上传方式切换逻辑
  document.querySelectorAll('input[name="uploadType"]').forEach(radio => {
    radio.addEventListener('change', function() {
      const zipSection = document.getElementById('zipUploadSection');
      const dirSection = document.getElementById('dirUploadSection');
      
      if (this.value === 'zip') {
        zipSection.style.display = 'block';
        dirSection.style.display = 'none';
      } else {
        zipSection.style.display = 'none';
        dirSection.style.display = 'block';
      }
    });
  });
  
  // 批量处理表单提交
  form.addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const submitBtn = document.getElementById('batchSubmitBtn');
    const spinner = document.getElementById('batchSpinner');
    const progressSection = document.getElementById('batchProgressSection');
    const progressBar = document.getElementById('batchProgressBar');
    const progressText = document.getElementById('batchProgressText');
    const batchResults = document.getElementById('batchResults');
    const downloadSection = document.getElementById('batchDownloadSection');
    
    // 重置UI
    submitBtn.disabled = true;
    spinner.classList.remove('d-none');
    batchResults.innerHTML = '';
    progressBar.style.width = '0%';
    progressText.textContent = '0/0';
    downloadSection.classList.add('d-none');
    
    // 准备表单数据
    const formData = new FormData(this);
    const uploadType = document.querySelector('input[name="uploadType"]:checked').value;
    formData.delete('batch_dir'); // 清除之前的字段

    // 根据上传类型处理不同输入
    if (uploadType === 'zip') {
      const zipFile = document.getElementById('batchZipFile').files[0];
      if (!zipFile) {
        alert('请选择ZIP文件');
        submitBtn.disabled = false;
        spinner.classList.add('d-none');
        return;
      }
      formData.append('batch_zip', zipFile);
    }

    if (uploadType === 'dir') {
      const dirInput = document.getElementById('batchDirPath');
      
      if (dirInput.files.length === 0) {
        alert('请选择目录');
        submitBtn.disabled = false;
        spinner.classList.add('d-none');
        return;
      }
      
      // 添加目录中的所有文件到FormData
      for (let i = 0; i < dirInput.files.length; i++) {
        formData.append('batch_files', dirInput.files[i]);
      }
    }
    
    try {
      const response = await fetch('/batch_align/', {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        throw new Error(`HTTP错误: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.error) {
        throw new Error(data.error);
      }
      
      // 显示进度区域
      progressSection.classList.remove('d-none');
      
      // 开始轮询处理状态
      const jobId = data.job_id;
      await pollBatchStatus(jobId, progressBar, progressText, batchResults, downloadSection);
      
    } catch (error) {
      console.error('批量处理错误:', error);
      batchResults.innerHTML = `
        <div class="alert alert-danger">
          <i class="bi bi-exclamation-triangle me-2"></i>
          批量处理失败: ${error.message}
        </div>
      `;
      progressSection.classList.remove('d-none');
    } finally {
      submitBtn.disabled = false;
      spinner.classList.add('d-none');
    }
  });
}