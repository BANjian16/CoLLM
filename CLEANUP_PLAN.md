# CoLLM 项目清理清单

## 建议保留

- `data/CMAPSS/`：原始 CMAPSS 数据。
- `datasets/`：FD001/FD003 数据集读取与预处理代码。
- `models/`：Small、Large、CoLLM、Fuzzy、Reflection 等模型代码。
- `train/`：FD001 当前正式训练权重与训练脚本。
- `train_fd003/`：FD003 当前正式训练权重。
- `results_test/FD001_*`：FD001 正式测试图。
- `results_test/FD003_*`：FD003 正式测试图。
- `ppt_work/output/CoLLM_复现项目汇报.pptx`：汇报 PPT。
- `ppt_work/src/build_deck.mjs`：PPT 可复现生成脚本。
- `config.py`、`eval_test.py`、`main.py`、`environment.yml`、`README.md`。
- 论文 PDF 与 `paper_text.txt`。

## 可删除的临时/重复文件

- `__pycache__/`
- `datasets/__pycache__/`
- `models/__pycache__/`
- `train/__pycache__/`
- `tmp/`
- `train_smoke/`
- `train_fd003_smoke/`
- `train_repro/`
- `train_repro_correct/`
- `train_repro_ft/`
- `train_repro_improved/`
- `train_repro_random/`
- `train_fd003_combined/`
- `train_backup_initial/`
- `result_test/`
- `results/tau_search/`
- `ppt_work/scratch/`
- `ppt_work/node_modules/`
- `results_test/test_rul_comparison.*`
- `results_test/test_error_vs_rul.*`
- `results_test/test_error_distribution.*`

## 清理后预期结构

```text
CoLLM/
  data/
  datasets/
  models/
  train/
  train_fd003/
  results/
  results_test/
  ppt_work/
    output/
    src/
  config.py
  eval_test.py
  main.py
  environment.yml
  README.md
  paper_text.txt
  CoLLM_Industrial_...pdf
```
