# How To Trigger This Skill

Use this skill only when you want Codex to operate or debug the Prompt Optimizer Agent workflow.

## Recommended Explicit Trigger

```text
$prompt-compliance-optimizer 看一下这个 badcase
```

```text
Use $prompt-compliance-optimizer to inspect this prompt-compliance badcase.
```

## Common Prompts

```text
$prompt-compliance-optimizer 看一下 Turn 8 为什么 Apply 后还是 badcase
```

```text
$prompt-compliance-optimizer 调一下这个 Trace List 问题
```

```text
$prompt-compliance-optimizer 检查这次 Apply / rerun / Conclusion 为什么不对
```

```text
用 prompt compliance optimizer 看一下这个 system prompt badcase
```

```text
这个 Prompt Optimizer Agent 的 Updated Conversation 没更新，按那个 skill 的流程排查一下
```

## Fixed Templates

### Find badcases and stop for review

```text
$prompt-compliance-optimizer

对话文件：<JSON file path>

任务：
生成一轮 badcase，整理候选列表给我人工核对，先不要 apply。

要求：
- 只判断 system-prompt compliance violation，不做泛泛回答质量优化。
- 每个候选 badcase 给出：编号、turn、error_type、证据、违反的 system prompt 规则、建议修复方式。
- 建议修复方式只能是：改 prompt / rerun target turn / 两者都要。
- 必须记录或引用本轮 scan 的 Round History / logs/optimization_rounds.jsonl 记录。
- 输出候选列表后停止，等我人工确认。
```

### Apply approved badcases and conclude

```text
$prompt-compliance-optimizer

我确认修复：<编号，比如 1 或 1,3 或 全部/全修>

任务：
对我确认的 badcase 自动 apply，必要时 rerun target turn，然后验证结果。

要求：
- 不要修复我没有确认的 badcase。
- Apply 后必须验证 Updated Conversation、prompt diff、tool turn、residual scan。
- 必须记录或引用 apply 和 residual scan 的 Round History / logs/optimization_rounds.jsonl 记录。
- 必须给 Conclusion，这是最重要的交付物。
- Conclusion 必须包含：修了什么、怎么验证、Round History 记录、是否还有 residual badcase、下一步是否需要人工核对。
- 修完后停止，不要自动再扫下一轮，除非我明确要求。
```

### Run one residual scan

```text
$prompt-compliance-optimizer

任务：
基于当前 Updated Conversation 和当前 Working system prompt，再生成一轮 residual badcase scan。

要求：
- 必须读取 Round History 或 logs/optimization_rounds.jsonl，说明这是第几轮 scan。
- 只检查剩余 system-prompt compliance badcase。
- 输出新的候选列表给我人工核对，先不要 apply。
- 记录本轮 scan 后停止。
```

### Batch scan a folder and stop for review

```text
$prompt-compliance-optimizer

对话文件夹：<folder path>

任务：
批量生成 badcase review，先不要 apply。

要求：
- 使用 Batch Mode。
- 每个 JSON 文件都要独立 parse、scan、记录 scan round。
- 输出 batch review JSON 和 Markdown。
- 按文件汇总候选 badcase：文件名、编号、turn、error_type、证据、建议修复方式。
- 只判断 system-prompt compliance violation，不做泛泛回答质量优化。
- 处理完停止，等我人工确认。
```

### Batch apply approved cases and conclude

```text
$prompt-compliance-optimizer

我确认批量修复：<全部/全修 或 approval file path>

Batch review：<batch_review.json path>

任务：
批量 apply 已确认 badcase，验证 residual scan，并给 conclusion。

要求：
- 不要修复我没有确认的 badcase。
- 每个输入文件输出一个 updated JSON。
- 每个文件都要记录 apply round 和 residual scan round。
- 必须给 aggregate conclusion 和 per-file conclusion。
- Conclusion 必须包含：总共修了多少、哪些文件修了、Round History event ids、unsupported cases、residual badcase count、下一步。
- 修完后停止，不要自动再扫下一轮，除非我明确要求。
```

## Natural Language Trigger

The skill may also trigger when you clearly ask about:

- Prompt Optimizer Agent
- system-prompt compliance badcases
- Trace List behavior
- Apply / Apply selected
- targeted rerun
- Updated Conversation
- required tool calls
- Conclusion wording
- Batch Mode

## Best Practice

Start the request with:

```text
$prompt-compliance-optimizer
```

Then describe the issue. This keeps the skill opt-in and avoids affecting normal coding tasks.
