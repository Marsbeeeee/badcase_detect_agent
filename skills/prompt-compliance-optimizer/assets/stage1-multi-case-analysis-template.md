# 多 Case Badcase 分析报告

## 分析范围

- 输入文件/目录：`{{input_path}}`
- 总记录数：`{{record_count}}`
- 成功分析：`{{analyzed_count}}`
- 解析失败/证据不足：`{{failed_count}}`
- 目标模型：`{{target_model}}`
- 分析阶段：Stage 1，仅分析，未运行实验，未修改 prompt

## 总体概览

| # | Case ID | Turn | Badcase 类型 | 终止类型 | 核心归因 |
|---:|---|---:|---|---|---|
| {{case_number}} | `{{case_id}}` | {{target_turn}} | `{{error_type}}` | {{termination_type}} | {{short_cause}} |

---

## Case {{case_number}}

### 1. 分析对象、正确流程和实际对比

- Case ID：`{{case_id}}`
- 数据文件：`{{source_file}}`
- 目标轮次：`{{target_turn}}`
- 目标模型：`{{target_model}}`
- 当前用户输入：

> {{target_user_message}}

正确流程：

> {{correct_route_in_one_or_two_sentences}}

目标模型实际回复：

> {{badcase_response}}

实际偏差：

> {{behavior_difference_in_one_or_two_sentences}}

### 2. 原因分析

#### 原因一：{{primary_cause_name}}

{{primary_cause_in_two_or_three_sentences}}

#### 原因二：{{secondary_cause_name_optional}}

{{secondary_cause_in_two_or_three_sentences_or_omit}}

#### Meta 证据：{{meta_evidence_type}}

{{meta_causal_interpretation_in_two_or_three_sentences}}

### 3. 最终归因

> {{final_attribution_in_one_or_two_sentences}}

- Primary：`{{primary_label}}`
- Subtype：`{{subtype_label}}`
- Contributing factor：`{{contributing_factor_or_none}}`
- Amplifier：`{{amplifier_or_none}}`
- 终止类型：`{{termination_type}}`

---

<!-- 为每个成功解析的 Case 重复以上三个部分。 -->

## 未完成分析的 Case

| Case ID | 状态 | 原因 |
|---|---|---|
| `{{case_id}}` | {{parse_or_evidence_status}} | {{exact_error_or_missing_evidence}} |

## 阶段状态

- Stage 1 multi-case analysis complete
- Experiments not run
- Experiment suggestions omitted
- Prompt not modified
