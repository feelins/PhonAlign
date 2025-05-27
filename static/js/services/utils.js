// 轮询批量处理状态
export async function pollBatchStatus(jobId, progressBar, progressText, batchResults, downloadSection) {
  let completed = false;
  let retryCount = 0;
  const maxRetries = 10;
  
  while (!completed && retryCount < maxRetries) {
    try {
      const response = await fetch(`/batch_status/${jobId}`);
      const data = await response.json();
      
      if (data.status === 'completed') {
        // 处理完成
        completed = true;
        updateProgressUI(data, progressBar, progressText, batchResults);
        
        // 显示下载按钮
        if (data.result_zip) {
          const downloadBtn = document.getElementById('downloadAllBtn');
          downloadBtn.onclick = () => {
            window.location.href = `/download_batch/${jobId}/${data.result_zip}`;
          };
          downloadSection.classList.remove('d-none');
        }
        
        // 添加成功消息
        batchResults.innerHTML += `
          <div class="alert alert-success">
            <i class="bi bi-check-circle me-2"></i>
            批量处理完成！共处理 ${data.total} 个文件，成功 ${data.success_count} 个，失败 ${data.error_count} 个。
          </div>
        `;
      } else if (data.status === 'processing') {
        // 更新进度
        updateProgressUI(data, progressBar, progressText, batchResults);
        await new Promise(resolve => setTimeout(resolve, 2000)); // 2秒后再次检查
      } else if (data.status === 'failed') {
        // 处理失败
        completed = true;
        batchResults.innerHTML = `
          <div class="alert alert-danger">
            <i class="bi bi-exclamation-triangle me-2"></i>
            批量处理失败: ${data.message || '未知错误'}
          </div>
        `;
      }
    } catch (error) {
      console.error('轮询错误:', error);
      retryCount++;
      if (retryCount >= maxRetries) {
        batchResults.innerHTML += `
          <div class="alert alert-danger">
            <i class="bi bi-exclamation-triangle me-2"></i>
            获取处理状态失败: ${error.message}
          </div>
        `;
        break;
      }
      await new Promise(resolve => setTimeout(resolve, 2000)); // 错误后等待2秒重试
    }
  }
}

// 更新进度UI
export function updateProgressUI(data, progressBar, progressText, batchResults) {
  const progress = Math.round((data.processed / data.total) * 100);
  progressBar.style.width = `${progress}%`;
  progressText.textContent = `${data.processed}/${data.total}`;
  
  // 更新结果列表
  if (data.latest_results && data.latest_results.length > 0) {
    data.latest_results.forEach(result => {
      const resultDiv = document.createElement('div');
      resultDiv.className = `batch-result-item ${result.status === 'error' ? 'error' : ''}`;
      resultDiv.innerHTML = `
        <div class="p-3">
          <h6 class="mb-1">
            ${result.filename}
            <span class="badge ${result.status === 'success' ? 'bg-success' : 'bg-danger'} float-end">
              ${result.status === 'success' ? '成功' : '失败'}
            </span>
          </h6>
          <small class="text-muted">${result.message || ''}</small>
          ${result.download_link ? `
          <div class="mt-2">
            <a href="${result.download_link}" class="btn btn-sm btn-outline-primary">
              <i class="bi bi-download me-1"></i>下载
            </a>
          </div>
          ` : ''}
        </div>
      `;
      batchResults.appendChild(resultDiv);
    });
  }
}

export function displayComparisonResults(onlyInA, onlyInB, common) {
    const summaryDiv = document.getElementById('compareSummary');
    const onlyInADiv = document.getElementById('onlyInA');
    const onlyInBDiv = document.getElementById('onlyInB');
    const commonDiv = document.getElementById('commonPhonemes');

    // 清空之前的结果
    onlyInADiv.innerHTML = '';
    onlyInBDiv.innerHTML = '';
    commonDiv.innerHTML = '';

    // 设置摘要信息
    if (onlyInA.length === 0 && onlyInB.length === 0) {
        summaryDiv.className = 'alert alert-success';
        summaryDiv.innerHTML = '<i class="bi bi-check-circle me-2"></i>两个音素列表<b>完全相同</b>';
    } else {
        summaryDiv.className = 'alert alert-warning';
        let summary = '<i class="bi bi-exclamation-triangle me-2"></i>两个音素列表<b>有差异</b><br>';

        if (onlyInA.length > 0) {
            summary += `A多出 ${onlyInA.length} 个音素: ${onlyInA.join(', ')}<br>`;
        }

        if (onlyInB.length > 0) {
            summary += `B多出 ${onlyInB.length} 个音素: ${onlyInB.join(', ')}<br>`;
        }

        summary += `共有 ${common.length} 个相同音素`;
        summaryDiv.innerHTML = summary;
    }

    // 显示A独有的音素
    onlyInA.forEach(phoneme => {
        const badge = document.createElement('span');
        badge.className = 'badge bg-danger';
        badge.textContent = phoneme;
        onlyInADiv.appendChild(badge);
    });

    // 显示B独有的音素
    onlyInB.forEach(phoneme => {
        const badge = document.createElement('span');
        badge.className = 'badge bg-danger';
        badge.textContent = phoneme;
        onlyInBDiv.appendChild(badge);
    });

    // 显示共同音素
    common.forEach(phoneme => {
        const badge = document.createElement('span');
        badge.className = 'badge bg-success';
        badge.textContent = phoneme;
        commonDiv.appendChild(badge);
    });

    // 显示结果区域
    document.getElementById('compareResults').classList.remove('d-none');
}