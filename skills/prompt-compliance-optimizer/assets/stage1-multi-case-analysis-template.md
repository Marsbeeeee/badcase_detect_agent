# 多 Case Badcase 分析结论

## 分析范围

- 输入文件/目录：`{{input_path}}`
- 总 Case 数：`{{record_count}}`
- 成功得出结论：`{{analyzed_count}}`
- 证据不足：`{{failed_count}}`
- 分析阶段：Stage 1，仅分析，未运行实验，未修改 prompt

## 结论总览

| # | Case ID | Badcase 类型 | 核心结论 | 终止类型 |
|---:|---|---|---|---|
| {{case_number}} | `{{case_id}}` | `{{error_type}}` | {{short_conclusion}} | {{termination_type}} |

---

## Case {{case_number}}：结论

> {{causal_conclusion_in_two_or_three_sentences}}

- 正确处理：{{correct_behavior_conclusion}}
- 错误起因：{{root_cause_conclusion}}
- 首次偏航位置：{{first_reliable_bad_word_or_phrase_and_its_semantic_effect_for_confirmed_badcase_or_boundary_unavailable}}
- Logprob 能说明什么：{{answer_only_the_supported_logprob_questions_and_merge_the_evidence_into_a_clear_plain_language_conclusion}}
- 模型对比结论：{{comparison_conclusion_or_insufficient_evidence}}
- 终止结论：{{termination_conclusion}}
- 最终归因：{{natural_language_final_attribution_explaining_state_interpretation_branch_and_result}}
- 影响因素：{{contributing_factor_and_amplifier_in_plain_language_or_none}}

---

<!-- 为每个成功得出结论的 Case 重复以上结论块。

首次偏航位置必须按错误形态填写：
- 错误陈述：填写首次形成错误含义的正常词或最小短语，并说明它如何使回复进入错误语义或分支。
- 必要内容遗漏：填写使遗漏变得不可恢复的收尾词、转折词或新动作；该词本身可以没有事实错误，但它在必需内容出现前启动了关闭或转向。
- 无法可靠定位：填写“词级边界不可用”，说明是 token 缺失、序列不完整还是无法与回复对齐；不要强行选择最低 logprob token。

Logprob 分析时，只回答当前记录能够提供证据的问题：
1. 错误发生时，模型是在犹豫，还是明显偏向错误方向？
2. 正确方向有没有出现在候选中？如果出现，是第几候选？
3. 选错以后，模型是稳定沿错误路线生成，还是仍在摇摆或尝试恢复？
4. 如果概率值饱和、token 缺失或无法与回复对齐，直接说明哪些判断无法进行。

最终将证据合并成清晰的自然语言结论，不限制句数；按当前证据充分表达即可。不要输出分类标签，也不要强行回答所有问题。没有可校准证据时明确写出证据限制，不要为了填满字段推测采样影响。

不要展示逐 turn 对话、完整回复原文、token 表、原始 logprob 数值或 Meta 参数表。不要用 taxonomy 标签或代码式标识代替自然语言归因。 -->

## 暂无结论的 Case

| Case ID | 原因 |
|---|---|
| `{{case_id}}` | {{exact_missing_evidence_or_parse_error}} |

## 阶段状态

- Stage 1 multi-case analysis complete
- Experiments not run
- Prompt not modified
