# Prompt Optimizer Agent + Skill Migration

## Archive Contents

- `prompt_optimizer_agent/`: the app source code.
- `prompt-compliance-optimizer/`: the Codex skill folder.
- `test_case/`: optional JSON files for smoke testing, if present in the archive.

## Install In Another Codex Environment

1. Unpack the archive.

```bash
tar -xzf prompt_optimizer_agent_with_skill.tar.gz
```

2. Move the skill into Codex skills.

```bash
mkdir -p ~/.codex/skills
cp -R prompt-compliance-optimizer ~/.codex/skills/
```

3. Enter the project and install dependencies.

```bash
cd prompt_optimizer_agent
python3 -m venv .venv39
.venv39/bin/python -m pip install -r requirements.txt
```

4. Validate the app.

```bash
PYTHONPYCACHEPREFIX=/private/tmp/prompt_optimizer_pycache .venv39/bin/python -m py_compile app.py prompt_optimizer_agent/agent_logic.py prompt_optimizer_agent/json_utils.py prompt_optimizer_agent/company_demo_client.py
```

5. Start the UI when needed.

```bash
.venv39/bin/streamlit run app.py
```

## Smoke Test Prompt

```text
$prompt-compliance-optimizer

对话文件：<path-to-json>

任务：
生成一轮 badcase，整理候选列表给我人工核对，先不要 apply。

要求：
- 只判断 system-prompt compliance violation，不做泛泛回答质量优化。
- 每个候选 badcase 给出：编号、turn、error_type、证据、违反的 system prompt 规则、建议修复方式。
- 必须记录或引用本轮 scan 的 Round History / logs/optimization_rounds.jsonl 记录。
- 输出候选列表后停止，等我人工确认。
```

After review, reply:

```text
全修
```

The final answer must include the required Conclusion.
