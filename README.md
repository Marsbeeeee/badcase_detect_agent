# Prompt Optimizer Agent

Prompt Optimizer Agent 是一个 Streamlit 工具，用来检查对话是否严格按照 `system prompt` 执行，并把发现的 badcase 转成可迭代的 prompt 修改。

它的核心目标不是评价回复“好不好”，而是验证：

- assistant 当前轮次是否违反了 `system prompt` 的明确规则、流程步骤、分支条件、工具规则或事实约束。
- 修改 prompt 后，是否能用新版 `system prompt` 和当前对话上下文重新生成对应轮次。
- 多轮迭代后，是否仍然存在新的 hard violation。

当前页面侧栏 build：`apply-context-v88`。

## Quick Start

```bash
cd /Users/zlshlt2501003/Desktop/prompt_optimizer_agent
python3 -m venv .venv39
source .venv39/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

如果使用 OpenAI：

```bash
export OPENAI_API_KEY="your_api_key"
export PROMPT_OPTIMIZER_MODEL="gpt-4o-mini"
```

如果使用 Company API，可以设置默认值：

```bash
export COMPANY_LLM_URL="http://192.168.101.15:9898"
export COMPANY_LLM_PROVIDER="openai_api_like"
export COMPANY_LLM_MODEL="H200_01_fc_9010"
```

## Supported Data

支持两类输入。

标准格式：

```json
{
  "system_prompt": "You are a helpful customer service agent...",
  "interactions": [
    {"role": "user", "content": "Can I return this item after 40 days?"},
    {"role": "assistant", "content": "Sure, no problem!"}
  ]
}
```

公司侧 `compress/dialog` 格式也支持。应用会自动抽取：

- system prompt
- dialog turns
- tools/function definitions
- provider/model/url 等 source metadata

## Main Workflow

```text
Upload JSON
  -> choose backend/model
  -> generate badcase
  -> inspect Trace List
  -> Apply / Apply selected
  -> Working system prompt changes
  -> targeted rerun selected assistant turn(s)
  -> Updated Conversation
  -> generate badcase again
  -> repeat until no actionable trace remains
```

重要原则：

- `Apply` 必须先真的修改 `Working system prompt`，才会 rerun。
- rerun 只重新生成 badcase 对应的 assistant turn，不会重跑整段对话；如果上传 JSON 带有 tools/function definitions，rerun 会把这些工具定义传给模型，并把返回的 `tool_calls` 保存成结构化 `tool_calls` 字段，同时在 `content` 中保留 `<function-call>...` 方便页面展示和 badcase 扫描。
- 对 `missing_fresh_promotion_search` 这类明确缺工具 badcase，targeted rerun 会把 `MandiriCX_Call_Center_search_promotion` 作为 required tool 传给模型；如果后端仍返回自然语言，应用会用本地 guard 把该 turn 转成 required function-call，避免 Updated Conversation 继续保留同一类 hard violation。
- 如果 rerun 生成了 function-call，Updated Conversation 会在 assistant function-call 后插入一个独立的 `Tool` turn 用于显示工具调用状态。当前应用没有真实工具执行器，所以该 `Tool` turn 会标记为 `not_executed`，不会伪造搜索结果。
- Apply 完成 targeted rerun 后，会立即生成英文 `Conclusion`。页面会显示 `Conclusion` 标题；内容固定为 3 个短段落，三段分别说明上一步具体修改、是否改善及非表层依据、下一步优化建议。
- 第二轮之后，Apply 会基于当前 Trace List 的来源对话继续生成。如果 Trace 来自 `Updated Conversation`，下一次 rerun 会继续用 updated 上下文，不会回到 original。
- 再次点击 `generate badcase` 时，judge 会使用当前最新版 `Working system prompt`。

## Page Guide

### Input

| 控件 | 作用 |
| --- | --- |
| `Upload conversation JSON` | 上传 `.json`、`.jsonl` 或 `.txt` 对话文件。 |
| `Load sample` | 加载示例对话。 |
| `Raw JSON` | 当前原始 JSON 文本，可以手动编辑。 |
| `Check JSON` | 重新解析 `Raw JSON`，并尽量修复常见 JSON 问题。解析失败时会清空当前已加载对话状态。 |

替换上传文件会重置当前 badcase、rerun、prompt version 和结论，避免旧状态串到新文件。

### Model

| 控件 | 作用 |
| --- | --- |
| `Backend: Company API` | 使用公司模型服务。rerun、prompt edit、judge 默认都走当前 Company API。 |
| `Backend: OpenAI` | 使用 OpenAI API，需要 `OPENAI_API_KEY`。 |
| `Company API URL` | 公司模型服务地址。 |
| `Refresh company models` | 请求 `/v1/models` 并刷新模型下拉框。 |
| `Company model` | 选择或输入 `provider:model`。 |
| `Provider` | Company API provider，例如 `openai_api_like`。 |
| `Max completion tokens` | 发送给模型的最大输出 token。 |
| `Badcase judge` | 选择 badcase 检测使用当前后端，还是单独用 OpenAI。 |
| `Auto post-rerun analysis` | 开启后，Apply 完成 rerun 后会额外自动跑一次完整 updated conversation judge，并用扫描结果更新 conclusion。默认建议关闭，手动点 `generate badcase` 更可控。 |
| `Conclusion` | Apply rerun 后生成三段英文总结：上一轮具体改动、badcase 是否真实改善及依据、下一步优化建议。请求内容已按三段证据结构化，避免模型把三部分合并成两段。 |

注意：`Badcase judge: OpenAI` 只改变检测 badcase 的后端，不改变重新生成对话的后端。rerun 始终使用主 `Backend`。

### System Prompt

| 区域 | 作用 |
| --- | --- |
| `Original system prompt` | 上传文件里的原始 prompt，只读。 |
| `Working system prompt` | 当前正在优化的 prompt。后续 `Apply`、rerun、download 都基于它。 |
| `Prompt Versions` | prompt 版本历史。每次 Apply 成功都会生成新版本；点击版本可查看 system prompt diff，并让页面下方 `Trace List` 切换到该版本被 `generate badcase` 扫描出来的结果。Apply 输入 trace 和版本扫描结果会分开记录，避免把上一版本 trace 误当成当前版本 trace。 |
| `Undo last version` | 回退上一个 prompt 版本。 |
| `v0 Original` / `vN Apply` | 查看某个版本，以及该版本 diff。 |

如果你手动修改 `Working system prompt`，后续操作会使用手动修改后的版本。

### Buttons

| 按钮 | 作用 |
| --- | --- |
| `download json` | 导出当前文件。未 rerun 时只替换最新版 `Working system prompt`；已有 Updated Conversation 时会导出 updated interactions，包括结构化 `tool_calls` 字段。不会导出 trace、diff、conclusion 等实验信息。 |
| `generate badcase` | 使用当前对话版本和当前 `Working system prompt` 检测 hard violation，并刷新 `Trace List`。如果当前是 Apply 后的 Updated Conversation，会用扫描结果更新 conclusion。 |
| `Select all` | 全选 Trace List 中的 trace item。 |
| `Clear` | 只清空选择状态，不删除 trace item。 |
| `Apply selected (N)` | 批量应用选中的 trace item，修改 prompt 后 targeted rerun 对应 assistant turns。 |
| `Apply` | 单条应用当前 trace item。 |
| `Delete` | 从 Trace List 删除该 trace item。 |
| 右侧对话卡片 `Bad case` | 人工标注该轮为 badcase。输入 remark 后可保存到 Trace List。 |

### Conversation View

右侧对话区域可能有两个版本：

| 视图 | 含义 |
| --- | --- |
| `Original` | 上传文件里的原始对话。 |
| `Updated` | Apply 后，用新版 prompt 重新生成过目标 assistant turn 的对话。 |

Trace List 有内容时，右侧会锁定到 Trace 的来源版本，避免 Trace item 和右侧高亮错位。

右侧对话支持 `Translate to Chinese` 一键翻译当前 Original/Updated 视图。开启翻译后，每个 turn 的正文会在原位置直接显示中文；隐藏翻译后恢复原文。翻译结果按当前视图内容缓存，可隐藏、刷新或清空。

## Badcase Judge Standard

当前 judge 是严格的 compliance judge，不是泛质量评审。

只有满足以下条件的项才会进入 Trace List：

- assistant turn 明确违反 `system prompt` 的规则、流程步骤、分支条件、工具规则或事实约束。
- judge 返回 `hard_violation=true`。
- `severity` 为 `high` 或 `medium`。
- `confidence >= 0.65`。
- 能指出 `violated_prompt_excerpt`。

不会收录的情况：

- 回复只是“不够礼貌”。
- 回复只是“不够详细”。
- 回复只是“可以更有说服力”。
- system prompt 没有明确规定，但 judge 主观觉得可以优化。

对于用户模糊、拒绝、不确定、答非所问的回复，只有当 system prompt 明确要求澄清或 fallback，而 assistant 仍然跳步骤时，才会进入 Trace List。

## Apply And Rerun Logic

点击 `Apply` 后会发生这些事：

1. 根据 trace item 的 evidence/recommendation 修改 `Working system prompt`。
2. 如果 prompt 没有变化，会显示 warning，并跳过 rerun。
3. 如果 prompt 发生变化，会新增 prompt version。
4. 应用会记录 expected prompt hash。
5. targeted rerun 前会写 preflight log 并校验 hash。
6. 校验通过后，才会真正请求 Company API / OpenAI 重新生成对应 assistant turn。
7. 如果模型返回 tool call，Updated Conversation 会把该 assistant turn 保存为结构化 `tool_calls` 字段，并显示为 `<function-call>tool_name:{...}</function-call>`，后续 `generate badcase` 可以据此判断是否已补上工具调用。
8. Updated Conversation 会替换对应 assistant turn，其他 turn 保持当前版本不变。
9. 页面生成英文 `Conclusion`，固定为 3 个短段落并显示在 Updated Conversation 上方。改善依据会检查输出行为、token/logprob 数据是否存在、meta/tool/config 线索和 residual badcase。
10. Trace List 会清空。下一轮需要再次点击 `generate badcase`。

Prompt edit 后台有固定 style guard：

- 不允许把一个 Step 改成超长段落。
- 不允许重复等价分支。
- 分支逻辑应拆成短的 numbered substeps 或短行。
- 对模糊/拒绝/不确定用户回复，优先补当前 step 的 clarification/fallback gate。
- 明显追加式章节会被拒绝或回退，例如 `Additional reliability requirements`、`appendix`、`addendum`。

## Logs

Company API 的远程 server log 需要在 `192.168.101.15:9898` 对应服务机器上查看。本项目会写本地 outgoing request log，方便确认自己到底发了什么。

| 文件 | 作用 |
| --- | --- |
| `logs/company_api_system_prompts.log` | 最新一次 Company API 请求的 prompt 原文。每次请求会覆盖。 |
| `logs/company_api_preflight_system_prompt.log` | targeted rerun 真正发送前的 preflight prompt。不会调用模型，只用于校验。 |
| `logs/company_api_requests.jsonl` | 最新一次 Company API 请求的结构化审计信息。每次请求会覆盖。 |
| `logs/company_api_logprobs.json` | 最新一次 Company API 响应的 logprobs 诊断信息。每次请求会覆盖；如果 API 不支持，会记录 unavailable/unsupported 原因。 |
| `logs/company_api_conclusion_dialog.jsonl` | 最新一次 conclusion 请求导出的 compress dialog。用于拿外部 `generate_from_chat_demo_server.py` 复现 conclusion 请求或拉 token logprob。 |

可以实时查看：

```bash
tail -f logs/company_api_system_prompts.log
```

怎么看 log：

- `targeted_rerun_preflight` 的 `first_system_message (outgoing working system prompt)` 是即将发送给模型的新业务 prompt。
- `targeted_rerun` 的 `first_system_message (outgoing working system prompt)` 是真实发送的业务 prompt。
- `judge` 请求的 `first_system_message (judge instruction, not the working prompt)` 是 judge 指令，不是业务 prompt。
- `judge` 请求真正用来评估的业务 prompt 在 `payload.system_prompt (working prompt evaluated by judge)`。
- `prompt_edit` 请求里当前 prompt 通常在 `payload.current_system_prompt (prompt being edited)`。

Apply 成功后，preflight 和真实 rerun 请求都会校验 expected prompt hash。如果 hash 不匹配，请求会在本地被阻断，不会发送到 Company API。

可选环境变量：

```bash
export COMPANY_LLM_REQUEST_LOG="logs/company_api_requests.jsonl"
export COMPANY_LLM_PREFLIGHT_SYSTEM_PROMPT_LOG="logs/company_api_preflight_system_prompt.log"
export COMPANY_LLM_SYSTEM_PROMPT_LOG="logs/company_api_system_prompts.log"
export COMPANY_LLM_LOGPROBS_LOG="logs/company_api_logprobs.json"
export COMPANY_LLM_CONCLUSION_DIALOG_LOG="logs/company_api_conclusion_dialog.jsonl"
export COMPANY_LLM_LOG_FULL_PROMPTS=1
export COMPANY_LLM_REQUEST_LOGPROBS=1
export COMPANY_LLM_TOP_LOGPROBS=5
```

如果不想落完整 prompt 原文：

```bash
export COMPANY_LLM_LOG_FULL_PROMPTS=0
```

## Common Checks

### 我怎么确认 rerun 真的用了新 prompt？

看 `logs/company_api_preflight_system_prompt.log`：

```text
purpose=targeted_rerun_preflight
expected_hash=...
match=True
first_system_message (outgoing working system prompt) hash=...
```

`expected_hash` 和 `first_system_message` hash 一致，说明 rerun 前校验通过。

### 我怎么把 conclusion 请求拿去外部脚本跑？

每次页面生成 conclusion 前，都会把同一份请求导出为 compress dialog：

```text
logs/company_api_conclusion_dialog.jsonl
```

这个文件包含 3 轮：`system` 是 conclusion 指令，`user` 是完整实验 payload，`assistant` 是带 `<|loss_start|>` 标记的待生成目标轮。可以用外部脚本复现：

```bash
python /Users/zlshlt2501003/Downloads/generate_from_chat_demo_server.py \
  --input logs/company_api_conclusion_dialog.jsonl \
  --output outputs/conclusion_eval.jsonl \
  --url http://192.168.101.15:9898 \
  --max-tokens 512 \
  --temperature 0.1 \
  --models openai_api_like:voyager-1.6-preview-run27m4a8b4-r3
```

如果脚本所在环境支持 token logprob，可在该脚本或服务端参数里打开对应 logprob 选项；本项目导出的 dialog 已经是完整 conclusion 请求上下文。

### 我怎么确认 generate badcase 用的是新 prompt？

看 `logs/company_api_system_prompts.log` 里的 judge 请求：

```text
purpose=judge
payload.system_prompt (working prompt evaluated by judge) hash=...
```

这个 `payload.system_prompt` 才是 judge 用来评估的业务 prompt。

### 为什么 Apply 后还会再次发现同一个 turn？

可能有三种情况：

- 新 prompt 没有改到真正的流程规则。
- targeted rerun 后该 assistant turn 仍然违反新版 prompt。
- 新版 prompt 更严格，judge 发现了同一 turn 上的新 hard violation。

可以检查：

- Prompt version diff 是否真的改到对应 step。
- Updated Conversation 里该 assistant turn 是否实质变化。
- Trace evidence 的 `violated rule` 是否和之前相同。

### 为什么 Trace List 空了？

Apply 成功后会清空当前 Trace List。下一轮需要点击 `generate badcase`，用 updated conversation 和新版 prompt 再扫描。

### 为什么 generate badcase 很慢或 timeout？

judge 需要读完整当前对话和 prompt。长对话、Company API 慢、模型输出过长都可能导致 timeout。可以：

- 降低 `Max completion tokens`。
- 使用更快的 Company model。
- 暂时关闭 `Auto post-rerun analysis`。
- 检查 Company API 服务是否稳定。

## Folder Structure

```text
prompt_optimizer_agent/
├── app.py
├── requirements.txt
├── README.md
├── examples/
│   └── sample_conversation.json
├── logs/
│   ├── company_api_system_prompts.log
│   ├── company_api_preflight_system_prompt.log
│   └── company_api_requests.jsonl
├── prompt_optimizer_agent/
│   ├── __init__.py
│   ├── agent_logic.py
│   ├── company_demo_client.py
│   └── json_utils.py
└── tests/
    ├── test_agent_logic.py
    ├── test_company_demo_client.py
    └── test_json_utils.py
```
