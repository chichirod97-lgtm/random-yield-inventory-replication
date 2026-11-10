# 随机交付库存问题：部分复现与损失函数扩展

**RICHINA ORTANCIA DOMINIQUE（杜米妮）**

Reports: [中文](index.html) · [English](report_en.html)

Public report: https://chichirod97-lgtm.github.io/random-yield-inventory-replication/

[Installation, parameters, seeds and rerun instructions / 安装与复跑](REPRODUCING.md) · [Actual AI assistance record / AI 使用记录](AI_USAGE.md)

New grid runner: `python code/run_grid.py` prints the plan only. `--execute` explicitly starts training. Full run = 24 configurations × 3 seeds × 3 methods (216 trainings). Duration is unmeasured; run a pilot first. The mean-loss trainer is newly reconstructed; historical configuration-ID mapping and full training archives are unavailable. This is partial reproduction plus a loss-function extension, not exact reproduction of the original paper.

课程作业。原论文：Yan, Chen, Fu & Bi, Computers & Operations Research 186 (2026), 107305。DOI: https://doi.org/10.1016/j.cor.2025.107305

**主要阅读文件：[HTML完整报告](index.html)**。下载后用浏览器打开；若仓库启用GitHub Pages，也可直接浏览该目录页面。GitHub文件页默认显示HTML源码。

内容包括：公式（3）—（5）解释、24配置/3种子的实验、均值与风险分析、局限、汇总数据和校验记录。提供中文报告与英文对应报告。

本作业为部分复现与方法扩展，不声称完整复现原论文432实例。平均损失目标是对论文指数损失的修改。使用AI辅助编程和报告整理，提交者须按课程要求披露并理解内容。

核验汇总：`python code/verify_summary.py`。

公开文件不包含原论文PDF、私人账户信息、完整历史Notebook输出或全部模型二进制快照。
