// 导入各功能模块
import { initAlignment } from './modules/alignment.js';
import { initBatch } from './modules/batch.js';
import { initDuration } from './modules/duration.js';
import { initPhoneme } from './modules/phoneme.js';
import { compareList } from './modules/compare.js';

// 页面加载完成后初始化各模块
document.addEventListener('DOMContentLoaded', () => {
  initAlignment();
  initBatch();
  initDuration();
  initPhoneme();
  compareList();
});