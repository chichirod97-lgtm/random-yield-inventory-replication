# Submission verification / 提交核查

Preparation date: 2026-11-10. Author: RICHINA ORTANCIA DOMINIQUE（杜米妮）.

- `python code/verify_summary.py`: PASS; 216 records and saved receipts. Mean LIR gaps: mean loss −5.1152934308%, exponential loss −1.2632257348%, GDSH −1.8512844996%. This does not execute training or independently reproduce saved model evaluations.
- `python code/check_submission.py`: PASS for local HTML link targets and fragments, Python syntax, grid size and summary checks.
- `python code/run_grid.py`: PASS, dry-run only, 24 configurations × 3 seeds × 3 methods = 216 models and 64,800 epochs at defaults. Training runtime is unmeasured; no training launched. NumPy is available locally; SciPy and Torch are unavailable in the preparation runtime. Thus model execution and dependency installation are not validated here.
- Browser: installed Microsoft Edge through Playwright, headless, desktop width 1280 and mobile width 390. Both pages have no broken images; English had no page overflow. Chinese mobile formula overflow was found, fixed and rechecked: no page overflow. Opening screenshots were visually inspected at mobile width; this is not a visual inspection of every section at every size.
- GitHub Pages root returned HTTP 200 before publication. Post-publication availability must be checked separately because deployment can be delayed.
- Local secret-pattern scan found no credential pattern in tracked text. Public author name added; no student number or key deliberately included. Private session logs were not copied.

本次仅核查轻量结果与页面，没有执行完整训练。历史风险指标缺少原始逐轨迹数据，未重新计算。新脚本的编号和平均损失入口明确标为本次建立，不能视作已找回历史源代码。重训耗时需先测量。
