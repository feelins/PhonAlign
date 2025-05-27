export function initAlignment() {
  const form = document.getElementById('alignmentForm');
  if (!form) return;

  form.addEventListener('submit', function(e) {
    e.preventDefault();
    
    // Show loading state
    const submitBtn = document.querySelector('#alignmentForm button[type="submit"]');
    const spinner = document.getElementById('loadingSpinner');
    const progressBar = document.getElementById('progressBar');
    const resultSection = document.getElementById('resultSection');
    const statusMessage = document.getElementById('statusMessage');
    
    submitBtn.disabled = true;
    spinner.style.display = 'inline-block';
    progressBar.style.display = 'block';
    resultSection.style.display = 'block';
    
    // Prepare form data
    const formData = new FormData();
    formData.append('audio_file', document.getElementById('audio_file').files[0]);
    formData.append('text_content', document.getElementById('text_content').value);
    formData.append('output_name', document.getElementById('output_name').value);
    
    // Send request
    fetch('/', {
      method: 'POST',
      body: formData,
      headers: {
        'Accept': 'application/json'
      }
    })
    .then(response => {
      if (!response.ok) {
        return response.json().then(err => {
          throw new Error(err.error || '服务器返回错误状态');
        }).catch(() => {
          throw new Error(`HTTP错误: ${response.status}`);
        });
      }
      return response.json();
    })
    .then(data => {
      if (data.error) {
        throw new Error(data.error);
      }
      
      if (data.status !== 'success') {
        throw new Error('处理未成功完成');
      }
      
      // Update status message
      statusMessage.className = 'alert alert-success';
      statusMessage.innerHTML = `<i class="bi bi-check-circle"></i> 对齐处理成功完成！耗时 <span class="processing-time">${data.processing_time}秒</span>`;
      
      // Set download links
      const downloadTextgridBtn = document.getElementById('downloadTextgridBtn');
      const downloadAudioBtn = document.getElementById('downloadAudioBtn');
      
      downloadTextgridBtn.href = `/download/${data.unique_id}/${data.textgrid_file}`;
      downloadTextgridBtn.style.display = 'inline-block';
      
      downloadAudioBtn.href = `/download/${data.unique_id}/${data.audio_file}`;
      downloadAudioBtn.style.display = 'inline-block';
      
      // Display visualizations
      document.getElementById('spectrogram-img').src = `/download/${data.unique_id}/${data.spectrogram}`;
      document.getElementById('textgrid-img').src = `/download/${data.unique_id}/${data.textgrid_viz}`;
    })
    .catch(error => {
      console.error('Error:', error);
      statusMessage.className = 'alert alert-danger';
      statusMessage.innerHTML = `<i class="bi bi-exclamation-triangle"></i> 处理失败: ${error.message}`;
    })
    .finally(() => {
      submitBtn.disabled = false;
      spinner.style.display = 'none';
      progressBar.style.display = 'none';
    });
  });
}