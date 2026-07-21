# {{report_title}} Badcase 分析

## 分析对象

- Case ID：`{{case_id}}`
- 数据文件：`{{source_file}}`
- 目标轮次：Turn {{target_turn}}
- Badcase 模型：`{{target_model}}`
- 当前用户输入：`{{target_user_message}}`

## 结论

{{causal_conclusion_covering_wrong_behavior_primary_cause_correct_route_and_termination}}

{{business_or_technical_termination_conclusion}}

## 关键对话状态

目标轮次前的可观察状态为：

```text
{{observable_workflow_state}}
```

{{workflow_state_interpretation}}

## 正确路由

正确路由是：

1. {{correct_route_step_1}}
2. {{correct_route_step_2}}
3. {{correct_route_step_3_or_omit}}

满足评测口径的核心回复应为：

> {{expected_core_response}}

## 目标模型的实际输出

{{target_model_short_name}} 输出：

> {{badcase_response}}

{{visible_behavior_difference_and_wrong_branch_evidence}}

## 原因分析

### 1. 核心原因：{{primary_cause_name}}

{{primary_cause_analysis}}

### 2. {{secondary_cause_name}}

{{secondary_cause_analysis}}

### 3. {{contributing_factor_name_or_omit}}

{{contributing_factor_analysis_or_omit}}

### 4. {{rule_priority_error_name_or_omit}}

{{rule_priority_error_analysis_or_omit}}

<!-- 原因小节按证据增删，不要求固定为四项；每项都必须说明状态、触发解释、分支选择或规则优先级如何导致实际错误。 -->

## 与其他模型的对比

| 候选模型 | 主要行为 | Human | Judge | 是否关闭通话/流程 |
| --- | --- | ---: | ---: | --- |
| `{{candidate_model}}` | {{candidate_behavior}} | {{human_score}} | {{judge_score}} | {{candidate_termination_behavior}} |

{{candidate_comparison_conclusion}}

## Meta 信息分析

| Meta 项 | 目标模型记录 | 对原因判断的意义 |
| --- | --- | --- |
| `task_success` | {{task_success}} | {{task_success_interpretation}} |
| 输出 token 数 | {{output_token_count}} | {{output_length_interpretation}} |
| `max_completion_tokens` | {{max_completion_tokens}} | {{completion_limit_interpretation}} |
| `temperature` / `top_p` | {{temperature}} / {{top_p}} | {{sampling_config_interpretation}} |
| `n_samples` | {{n_samples}} | {{sample_count_interpretation}} |
| PPL | {{ppl}} | {{ppl_interpretation}} |
| 首次偏航词/短语 | {{first_reliable_bad_word_or_phrase}} | {{first_bad_span_interpretation}} |
| 偏航边界附近候选 | {{boundary_alternatives_summary}} | {{boundary_alternatives_interpretation}} |
| 末 token / 停止状态 | {{final_token_and_stop_summary}} | {{final_token_and_stop_interpretation}} |

{{meta_causal_conclusion}}

<!-- 必须按顺序扫描目标模型的完整 token-logprob 序列后再填写结论。不要把最低 logprob token 当作首次偏航，不要把 logprob/PPL 当作正确性证据；缺失或无法对齐时明确写出证据限制。 -->

## 为什么不是技术性提前停止

{{technical_termination_analysis}}

因此终止类型为：

> `{{termination_type}}`

<!-- 若证据支持技术终止，将本节标题改为“终止类型判断”，并准确写明技术终止证据。若只是结束当前工作流阶段而非结束对话，也要明确区分。 -->

## 待验证原因假设

以下均为基于观察的可证伪假设，尚未经过实验验证。

### H1：{{hypothesis_1_name}}

- 单一变量：{{hypothesis_1_single_variable}}
- 预期行为：{{hypothesis_1_expected_behavior}}
- 不变控制：{{hypothesis_1_controls}}
- 通过标准：{{hypothesis_1_pass_criterion}}

### H2：{{hypothesis_2_name_or_omit}}

- 单一变量：{{hypothesis_2_single_variable_or_omit}}
- 预期行为：{{hypothesis_2_expected_behavior_or_omit}}
- 不变控制：{{hypothesis_2_controls_or_omit}}
- 通过标准：{{hypothesis_2_pass_criterion_or_omit}}

<!-- 仅保留有证据支持且可由单一变量验证的假设；按需增删 H2、H3 等。Stage 1 不执行实验。 -->

## 最终归因

{{natural_language_final_attribution_explaining_state_interpretation_branch_and_result}}

- Primary：`{{primary_label}}`
- Workflow error：`{{workflow_error_label_or_none}}`
- Semantic subtype：`{{semantic_subtype_label_or_none}}`
- Contributing factor：`{{contributing_factor_label_or_none}}`
- Rule error：`{{rule_error_label_or_none}}`
- Excluded：{{excluded_causes}}
- Termination type：`{{termination_type}}`

## 阶段状态

- `Stage 1 analysis complete`
- `experiments not run`
- `prompt not modified`

<!-- 单 case 报告应完整展示因果分析所需证据，但只引用证明路由所必需的最小对话和 prompt 片段。不要泄露无关原始数据，也不要把观察性假设写成已验证结论。 -->
