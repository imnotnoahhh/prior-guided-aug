# Result of Code Cleanup Analysis & Proposal

为了确保代码库符合学术论文开源标准（Clean, Reproducible, Professional），我进行了全面的从静态分析和关键词扫描。以下是必须处理的清理项清单。

## 1. 移除遗留的版本标记与调试代码

我们在早期开发中使用了 `v4`, `v5`, `v6` 等标记，现需全部移除，使代码看起来像是一个成熟的整体版本。

### 需清理文件清单：
*   **`src/augmentations.py`**:
    *   移除底部的 `if __name__ == "__main__":` 测试代码块。其中包含 `print("\n[v5-1] ...")` 等调试信息。
    *   更新注释引用的文档版本：`Reference: research_plan_v4.md` -> `Reference: docs/research_plan.md`。
*   **`src/models.py`**:
    *   去除注释：`... per No-NAS constraint (research_plan_v4.md Section 5)` -> `Constraint: No-NAS (ResNet-18 only).`
*   **`src/dataset.py` & `src/utils.py`**:
    *   同样修正所有指向 `research_plan_v4.md` 的引用。

## 2. Pylint 静态分析发现的问题

Pylint 扫描发现以下代码质量问题，建议修复：

*   **`run_phase0_calibration.py`**:
    *   **未使用变量**: `train_loss`, `val_loss`, `top5_acc` 在 `line 227-230` 被赋值但未使用。建议改为 `_` 占位符。
    *   **未使用导入**: `import os`, `from typing import List, Tuple`。建议删除。
*   **`src/augmentations.py`**:
    *   函数 `train_static_policy` (被导入到 main 中) 内部有重复导入 `from src.augmentations import build_transform_with_ops`。建议删除内部导入。

## 3. 文档与注释标准化

*   **Docstrings**: `main_phase_c.py` 的顶部 docstring 仍包含 `Phase C: ... (v7)`。建议移除 `(v7)`。
*   **TODOs**: 虽然 grep 未扫到显眼的 TODO，建议再一次人工过目所有 `# NOTE` 注释，确保没有开发阶段的临时笔记。

## 4. 建议的清理行动计划

如果你同意，我将执行以下操作：

1.  **Refactor**: 修正所有 `src/*.py` 中的文档引用，移除 `_v4`/`_v5` 后缀。
2.  **Clean**: 删除 `src/augmentations.py` 底部的测试代码块。
3.  **Fix**: 修复 `run_phase0_calibration.py` 中的 unused variables/imports。
4.  **Final Polish**: 统一 `main_phase_c.py` 等脚本的头部说明，移除内部的重复 import。

请确认是否执行此清理计划。
