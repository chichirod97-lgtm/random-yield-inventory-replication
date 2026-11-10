# AI assistance and correction record / AI 使用与纠正记录

Author: **RICHINA ORTANCIA DOMINIQUE（杜米妮）**

This is a bounded reconstruction from the conversation visible during submission preparation, public code comments, the Chinese report and public evidence JSON. It is not a complete transcript. Missing prompts, dates, model identities and intermediate steps have not been reconstructed as facts. No private session logs, student number or credentials are included.

本记录依据本次可见对话、公开代码注释、中文报告和证据文件整理，不是完整逐字历史。未补造缺失提示词、日期、模型身份及中间步骤，不公开私人会话、学号或密钥。

## Prompts actually visible / 可见的真实提示

1. “ON COMMENCE POUR LA PREMIERE ETAPE POUR LAUTRE ARTICLE 解释公式3到公式5的推导 ...” — The assistant read the new random-yield article's equations and explained the target stock approximation, WNH scenario average and GDSH weight generalization in Chinese and French. This is evidence of methodological explanation, not proof of training execution.
2. “我现在希望把这篇文章完整的复现出来…先写复现计划，不要写代码。” — The supplied package name referred to the older E2E article. The assistant audited that older package. This scope drift must not be presented as random-yield experimental work.
3. “我没有办法拿到需要的数据…Google, colab, pro…先列计划，我同意后开始执行” and “方案可以，开始执行” — Planning and execution authorization are visible, but the subsequent audit concerned the older E2E data. Those results are not evidence for this repository's 24-configuration training.
4. The current submission request asks for an English HTML version, actual AI prompt/correction documentation, dependencies, a 24 x 3 runner, link/result/display checks, publication of the author's name, and exclusion of student number and API keys. It explicitly states that `verify_summary.py` passed but did not verify complete training.

## Corrections visible in the history / 历史中可见的纠正

The older-package audit initially counted physical CSV lines and incorrectly reported 3,364 observations. A proper CSV reader returned 3,000 because quoted fields contain newlines. The assistant corrected the audit prose, although an old table still showed the incorrect count at that time. This is an AI-audit error and correction from the older project, not a random-yield result.

旧项目曾把 CSV 物理行数误当记录数，后用 CSV 解析器确认 3,000 条；该纠正不能作为本项目训练证据。

## Changes evidenced by public artifacts, with missing prompts / 公开文件支持但缺少提示词的变更

- `learning.py` explicitly marks the added nonnegative order projection, nonnegative GDSH softplus weights, and finite-Adam-state checks. These modify/diagnose the implementation and are not verbatim author code.
- `learning_v2.py` contains float64, teacher initialization, state scaling, finite-parameter/gradient/optimizer checks and checkpoint identity checks. The report describes diagnostics at instances 13/21 and mean-loss diagnostics at 0/12. The underlying conversations for these interventions are unavailable here; no exact prompt or author attribution is asserted.
- The report describes replacing exponential loss with mean discounted cost. This is a loss-function extension, not exact paper reproduction. Evidence files record two repeated CPU training examples; they do not establish that all 216 headline trainings were rerun from scratch.

上述代码及报告支持“发生过这些实现变更”，但无法确认缺失提示词或每项操作的具体执行者。平均损失是扩展，不是原论文损失的忠实复现。

## Submission preparation performed now / 本次提交整理

Added an English companion report and language links, installation instructions, compatible dependency ranges, an opt-in grid runner and a reconstructed mean-loss training entry point. Recorded the missing historical configuration-ID manifest and missing original mean-loss source. Kept all submitted numerical results unchanged. Lightweight verification receipts are documented separately. No costly training was launched. AI assists implementation and writing; the named author remains responsible for understanding and approving the submitted work.

本次新增英文报告、语言导航、依赖安装说明、显式启动的批量脚本及重建的平均损失训练入口；保留历史结果，注明缺失来源。本次未执行昂贵训练。
