export function initPhoneme() {
  const form = document.getElementById('compareForm');
  if (!form) return;

  form.addEventListener('submit', function(e) {
    e.preventDefault();

    const resultsDiv = document.getElementById('compareResults');
    const errorDiv = document.getElementById('compareError');

    // 重置UI
    resultsDiv.classList.add('d-none');
    errorDiv.classList.add('d-none');

    try {
      // 获取输入并处理为音素数组
      const textA = document.getElementById('phonemesA').value.trim();
      const textB = document.getElementById('phonemesB').value.trim();

      if (!textA || !textB) {
        throw new Error('请输入两个音素列表进行比较');
      }

      // 分割音素（支持空格、换行、逗号分隔）
      const phonemesA = textA.split(/[\s,\n]+/).filter(p => p.trim());
      const phonemesB = textB.split(/[\s,\n]+/).filter(p => p.trim());

      if (phonemesA.length === 0 || phonemesB.length === 0) {
        throw new Error('音素列表不能为空');
      }

      // 去重
      const setA = new Set(phonemesA);
      const setB = new Set(phonemesB);

      // 比较结果
      const onlyInA = [...setA].filter(p => !setB.has(p));
      const onlyInB = [...setB].filter(p => !setA.has(p));
      const common = [...setA].filter(p => setB.has(p));

      // 显示结果
      displayComparisonResults(onlyInA, onlyInB, common);

    } catch (err) {
      errorDiv.textContent = err.message;
      errorDiv.classList.remove('d-none');
    }
  });

  function displayComparisonResults(onlyInA, onlyInB, common) {
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
}