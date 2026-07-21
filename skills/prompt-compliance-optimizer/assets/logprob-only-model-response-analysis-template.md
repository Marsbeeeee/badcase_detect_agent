# Badcase 模型回复 Logprob 专项分析

## 分析范围

- 输入文件：`{{input_path}}`
- Badcase 数：`{{badcase_count}}`
- 分析单位：一个 Case 只对应一条具体模型回复 badcase
- 数据路径：`dialog[{{dialog_index}}].evaluate.{{badcase_model}}.meta.logprobs`
- 分析范围：只分析每条 badcase 模型回复自身的 token logprobs，不展开同一 turn 下其他候选模型，不做完整工作流归因，不运行实验，不修改 prompt

## 结论总览

| # | Case ID | Turn | Badcase 模型 | 首次偏航位置 | Logprob 核心结论 |
|---:|---|---:|---|---|---|
| {{case_number}} | `{{case_id}}` | {{target_turn}} | `{{badcase_model}}` | {{first_bad_boundary_or_unavailable}} | {{short_logprob_conclusion}} |

---

## Case {{case_number}}

- Case ID：`{{case_id}}`
- 目标轮次：Turn {{target_turn}}
- Badcase 模型：`{{badcase_model}}`
- 首次偏航位置：{{first_reliable_bad_word_or_phrase_omission_boundary_or_boundary_unavailable}}

### 基本信息

**上一轮 User input**

{{previous_user_input_as_plain_text}}

**本轮 GT 回复**

{{current_turn_ground_truth_response_as_plain_text}}

**本轮模型回复**

{{current_turn_badcase_model_response_as_plain_text}}

### Logprob 能说明什么

- **偏航边界：{{boundary_conclusion}}**
  - {{boundary_word_or_minimal_phrase_and_its_semantic_effect}}
- **边界处的局部选择：{{local_choice_conclusion}}**
  - **实际选择**：{{chosen_token_or_span_logprob_and_probability_when_calibrated}}
  - **关键候选**：{{most_relevant_alternatives_ranks_logprobs_or_probabilities}}
  - **差距含义**：{{margin_relative_advantage_and_whether_the_model_was_hesitating_or_clearly_leaning}}
- **正确方向是否可见：{{correct_direction_visibility_conclusion}}**
  - {{what_the_recorded_candidate_semantically_supports}}
  - {{what_its_presence_or_absence_cannot_prove}}
- **偏航后的生成轨迹：{{post_boundary_trajectory_conclusion}}**
  - {{evidence_of_lock_in_wavering_direction_change_or_recovery}}
  - {{whether_later_competition_is_business_route_level_or_only_wording_or_punctuation_level}}
- **采样的作用：{{sampling_conclusion_or_omit}}**
  - {{supported_scope_of_sampling_effect_and_what_cannot_be_attributed_to_sampling}}
- **停止与完整性：{{stop_and_completeness_conclusion_or_omit}}**
  - {{final_token_output_length_and_stop_evidence}}
- **证据边界：{{material_evidence_limit_or_omit}}**
  - {{which_specific_confidence_ranking_or_boundary_judgment_is_blocked_and_why}}

---

<!-- 为每条具体模型回复 badcase 重复以上 Case 区块。一个 Case 必须唯一绑定一条 `dialog[i].evaluate.<badcase_model>` 回复；不要把同一 JSONL 记录下的其他候选模型回复展开为这个 Case 的子项。

必须读取当前 badcase 模型自己的 `dialog[i].evaluate.<badcase_model>.meta.logprobs`，不要误用根级模型、其他候选模型或 chronological dialog 中基础回复的 Meta。

“基本信息”必须原样读取并展示三项内容：目标轮次之前最近一条 role 为 `user` 的 content、目标轮次 `dialog[i].content` 中的 GT 回复、目标轮次 `dialog[i].evaluate.<badcase_model>.content` 中的 badcase 模型回复。使用可自然换行的普通 Markdown 文本段落，不使用代码块、行内代码或引用块。不要概括、翻译、清理 function-response 包装或删除 `<dialog-end>`；为了让尖括号内容在 Markdown 预览中可见，可以仅将 `<` 和 `>` 转义为 `&lt;` 和 `&gt;`，显示文字必须保持不变。字段缺失时明确写“未提供”。

先用 bytes（有效时）或 token 按顺序重建该 badcase 回复，并与它自己的 content 对齐。对齐是内部前置检查：成功时不要在报告中重复展示回复路径、Logprob 路径或“完整对齐”等过程信息；只有缺失或对齐失败时，才在“Logprob 能说明什么”的对应条目或“无法分析的 Badcase”中说明。必须扫描完整 token 序列。“基本信息”允许各展示一次上一轮 User input、GT 回复和模型回复；除此之外，分析结论只保留支持判断所需的关键证据，不再重复完整回复，也不展示逐 token 表、原始 logprob 列表或完整 top-k 列表。

首次偏航位置按以下规则确定：
- 错误陈述：填写首次形成错误含义的正常词或最小短语。
- 必要内容遗漏：填写使遗漏无法再补回的收尾词、转折词或新动作。
- Token 缺失、序列不完整或无法与回复对齐：填写“词级边界不可用”，并说明具体原因。

Logprob 分析时，只回答当前 badcase 回复能够提供证据的问题：
1. 错误发生时，模型是在多个方向之间犹豫，还是明显偏向实际生成的错误方向？
2. 正确方向有没有出现在候选中？如果出现，是第几候选？
3. 经过首次偏航位置以后，模型是稳定沿错误路线生成，还是仍在摇摆、改变方向或尝试恢复？
4. 最终 token、输出长度和停止状态是否支持完整作答或技术终止判断？
5. 如果概率值饱和、候选缺失、token 缺失或无法与回复对齐，哪些置信度、排名或边界判断无法进行？

优先使用上面的具名条目组织结论，让读者能直接看到“在哪里偏航、当时有多犹豫、正确方向是否进入候选、之后是否锁定错误路线、采样是否只是放大因素、回复是否完整”。删除没有证据支撑且不影响结论的可选条目，不要为了填满模板而推测；“偏航边界”应保留，无法对齐时明确写“词级边界不可用”。存在实质证据限制时保留“证据边界”，并说明它具体阻止了哪一种判断。

每个条目必须先给出自然语言结论，再给足以理解该结论的具体证据。直接从证据事实或分析结论起句，使用客观报告语气；不要写“这里所说的……”“所谓……是指……”“你可以理解为……”“换句话说……”等面向提问者解释术语的问答式引入。

不要把分析压缩成“明显偏向”“没有进入 top-k”“选错后锁定”等一句式判断。证据存在时，继续交代关键决策词或短语、实际选中项与最相关候选、差距或大致概率、候选在语义上分别会把回复带向哪里，以及后续出现的是业务路线恢复、同一路线内的措辞摇摆，还是只影响结尾形式的采样变化。说明这些细节为什么支持当前结论，并区分“业务路线竞争”和“局部用词或标点竞争”。

同一判断包含多个数据点或解释层次时，必须使用两级无序列表：一级 bullet 写可独立阅读的结论，二级缩进 bullet 分别写实际选择、关键候选、差距、语义影响或推断边界。不要把多个候选、多个概率和结论全部挤在同一个长句中。没有足够细节时可以只保留一级 bullet；不要为了满足层级而制造证据。

数值有效时，可紧随结论写选中项 logprob、概率、相关候选的排名及差值；数值不可校准时只做定性描述。不要输出分类标签、逐 token 表或完整 top-k；模型回复只在“基本信息”中完整展示一次，Logprob 分析中不要再次整段重复。Logprob 只能说明生成偏好、局部置信度和停止形态，不能单独证明回复错误，也不能直接证明完整的隐藏推理路径。

数值为选中 token `0`、其他候选 `-9999` 等饱和形式时，不得只写“概率饱和”后结束，也不得把它当作可校准概率。必须用自然语言展开说明：这种格式在记录上相当于把选中项写成概率 1、把其他项写成接近 0，但整段反复出现完全相同的极值时，更可能是记录接口使用了哨兵值、裁剪值或占位候选，而不是模型真的在每一步都具有 100% 置信度；仅凭当前文件不能进一步断定是哪一种记录机制。说明饱和覆盖整条序列还是仅覆盖关键决策点，并明确它会使哪些判断失效，例如真实候选差距、犹豫程度、正确方向排名和采样影响。token 文本仍可用于重建回复和确认生成顺序，但不能用这种连续性替代置信度证据。

正确方向未出现在记录的 top-k 时，只能说未观察到，不能推断其具体排名或概率。不要把最低 logprob token 自动当作错误起点，也不要仅凭 temperature 推断采样是原因。
-->

## 跨 Case 的 Logprob 观察

{{cross_case_qualitative_patterns_only_when_semantic_decision_points_and_evidence_are_comparable_otherwise_state_not_comparable}}

<!-- 跨 Case 只总结可对齐的定性现象。不同模型或 checkpoint 的 logprob 标度不得直接做数值高低排名；条件不一致时分别描述，不强行比较。 -->

## 无法分析的 Badcase

| Case ID | Turn | Badcase 模型 | 原因 |
|---|---:|---|---|
| `{{case_id}}` | {{target_turn}} | `{{badcase_model}}` | {{missing_content_missing_logprobs_parse_error_or_alignment_failure}} |

## 分析状态

- Logprob-only badcase analysis complete
- Experiments not run
- Prompt not modified
