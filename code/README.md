# 代码说明 / Code guide

代码为本次项目使用的实现，不宣称是论文作者提供的原始实现。

- `core.py`：状态、需求和交付生成、LIR/WNH、模拟器、离散DP。
- `learning.py`：GDSH、基础网络和可微成本。
- `learning_v2.py`：双精度、初始化、缩放和指数损失训练。
- `verify.py`、`verify_v2.py`：基础与数值验证。
- `dp_study.py`：有限动作小问题演示，不代表连续动作问题的精确最优值。
- `verify_summary.py`：只读核对公开汇总，不需Torch，无训练。

运行：`python code/verify_summary.py`（仓库中该课程目录下）。

在代码目录运行 `python -m unittest verify verify_v2 -v` 需要 NumPy、SciPy、PyTorch；`verify_v2` 含极小训练。历史打包环境为Python3.13.16、NumPy2.1.3、SciPy1.16.3、Torch2.11.0；不是所有历史训练环境的完整证明。

本公开目录是课程提交材料，未包含全部360个二进制模型或所有原始逐轨迹成本。完整本地快照另行保留。所有原始论文公式和思想归原作者；未附作者PDF，代码的进一步分发许可未作额外授权。
