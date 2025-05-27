export function initDuration() {
  const form = document.getElementById('durationForm');
  if (!form) return;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const calculateBtn = document.getElementById('calculateBtn');
    const spinner = document.getElementById('durationSpinner');
    const resultDiv = document.getElementById('durationResult');
    const errorAlert = document.getElementById('durationErrorAlert');
    const errorMessage = document.getElementById('durationErrorMessage');
    
    // 重置UI
    resultDiv.classList.add('d-none');
    errorAlert.classList.add('d-none');
    calculateBtn.disabled = true;
    spinner.classList.remove('d-none');
    
    const formData = new FormData(e.target);
    
    try {
      const startTime = new Date();
      const response = await fetch('/calculate_duration/', {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      const endTime = new Date();
      const processTime = (endTime - startTime) / 1000;
      
      document.getElementById('durationProcessTime').textContent = `处理时间: ${processTime.toFixed(2)}秒`;
      
      if (data.status === 'success') {
        document.getElementById('totalDuration').textContent = 
          data.total_duration_seconds.toFixed(2);
        document.getElementById('totalHours').textContent = 
          data.total_duration_hours.toFixed(2);
        document.getElementById('fileCount').textContent = data.file_count;
        document.getElementById('dirPath').textContent = data.directory_path;
        
        // 显示错误文件（如果有）
        if (data.error_files && data.error_files.length > 0) {
          const errorSection = document.getElementById('durationErrorSection');
          const errorList = document.getElementById('durationErrorList');
          
          errorList.innerHTML = '';
          data.error_files.forEach(file => {
            const li = document.createElement('li');
            li.className = 'list-group-item d-flex justify-content-between align-items-start';
            li.innerHTML = `
              <div class="ms-2 me-auto">
                <div class="fw-bold">${file.filename}</div>
                ${file.error}
              </div>
              <span class="badge bg-danger rounded-pill">!</span>
            `;
            errorList.appendChild(li);
          });
          
          errorSection.classList.remove('d-none');
        } else {
          document.getElementById('durationErrorSection').classList.add('d-none');
        }
        
        resultDiv.classList.remove('d-none');
      } else {
        errorMessage.textContent = data.message;
        errorAlert.classList.remove('d-none');
      }
    } catch (err) {
      errorMessage.textContent = `请求失败: ${err.message}`;
      errorAlert.classList.remove('d-none');
    } finally {
      calculateBtn.disabled = false;
      spinner.classList.add('d-none');
    }
  });
}