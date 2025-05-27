import { displayComparisonResults } from '../services/utils.js';

// 音素比较功能
export function compareList() {
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
}
        