# 从 GitHub 获取并使用 Prompt Compliance Optimizer Skill

本文介绍如何从 GitHub 获取并安装 `prompt-compliance-optimizer` Codex skill。

仓库地址：

```text
https://github.com/Marsbeeeee/badcase_detect_agent
```

## 1. 准备环境

安装前请确认电脑上已有：

- Git
- Python 3.9 或更高版本
- Codex
- GitHub 仓库访问权限

如果仓库是私有仓库，请先让仓库管理员添加你的 GitHub 账号，并在本机配置 GitHub 登录凭据或 SSH key。

## 2. 克隆完整仓库

这个 skill 会使用仓库中的 `app.py`、`prompt_optimizer_agent/` 和 `tools/`，因此不能只下载 `SKILL.md`，必须保留完整仓库。

使用 HTTPS：

```bash
git clone https://github.com/Marsbeeeee/badcase_detect_agent.git
cd badcase_detect_agent
```

私有仓库也可以使用 SSH：

```bash
git clone git@github.com:Marsbeeeee/badcase_detect_agent.git
cd badcase_detect_agent
```

## 3. 安装 Skill

skill 的源目录是：

```text
skills/prompt-compliance-optimizer/
```

### macOS 或 Linux

在仓库根目录执行：

```bash
mkdir -p ~/.codex/skills/prompt-compliance-optimizer
rsync -a --delete \
  skills/prompt-compliance-optimizer/ \
  ~/.codex/skills/prompt-compliance-optimizer/
```

### Windows PowerShell

在仓库根目录执行：

```powershell
$dest = Join-Path $HOME ".codex\skills\prompt-compliance-optimizer"
New-Item -ItemType Directory -Force $dest | Out-Null
Copy-Item "skills\prompt-compliance-optimizer\*" $dest -Recurse -Force
```

安装完成后，重新打开 Codex，或在 Codex 的下一轮对话中使用该 skill。

## 4. 安装项目依赖

### macOS 或 Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
```

不要把 API key 写进仓库文件。请通过环境变量或未被 Git 跟踪的 `.env` 文件配置。

例如，使用 OpenAI 时：

```bash
export OPENAI_API_KEY="your_api_key"
export PROMPT_OPTIMIZER_MODEL="your_model"
```

Windows PowerShell：

```powershell
$env:OPENAI_API_KEY = "your_api_key"
$env:PROMPT_OPTIMIZER_MODEL = "your_model"
```

## 5. 在 Codex 中调用

建议从仓库根目录启动 Codex，以便 skill 找到应用代码：

```bash
cd badcase_detect_agent
codex
```

在对话中明确调用 skill：

```text
$prompt-compliance-optimizer

检查 <conversation.json> 中是否存在违反 system prompt 的 badcase，
生成候选列表和 scan round id，然后停下来等我审核，不要直接 Apply。
```

批量扫描示例：

```text
$prompt-compliance-optimizer

批量扫描 <数据目录> 中的 conversation JSON，生成 review JSON 和 Markdown 报告，
按文件汇总候选 badcase，然后停下来等我批准。
```

批准指定 badcase 后：

```text
$prompt-compliance-optimizer

批准 review 结果中的第 1 和第 3 个 badcase。
只处理这两个 case，完成 Apply、一次 residual scan 和最终 Conclusion 后停止。
```

## 6. 可选：使用 Codex GitHub Skill Installer

如果本机带有 Codex 的 `skill-installer`，可以直接从 GitHub 安装 skill 目录。

macOS 或 Linux：

```bash
python ~/.codex/skills/.system/skill-installer/scripts/install-skill-from-github.py \
  --repo Marsbeeeee/badcase_detect_agent \
  --path skills/prompt-compliance-optimizer
```

Windows PowerShell：

```powershell
py "$HOME\.codex\skills\.system\skill-installer\scripts\install-skill-from-github.py" `
  --repo Marsbeeeee/badcase_detect_agent `
  --path skills/prompt-compliance-optimizer
```

公开仓库可直接下载。私有仓库需要本机已有 Git 凭据，或者设置 `GITHUB_TOKEN` / `GH_TOKEN`。

注意：GitHub Skill Installer 只安装 skill 目录。由于本 skill 依赖完整项目，仍需按第 2 节克隆仓库，并从仓库根目录运行 Codex。

如果目标 skill 目录已经存在，installer 会停止而不是覆盖。更新时请使用下一节的同步方式。

## 7. 更新 Skill

进入之前克隆的仓库并拉取更新：

```bash
cd /path/to/badcase_detect_agent
git pull
```

macOS 或 Linux 再执行：

```bash
rsync -a --delete \
  skills/prompt-compliance-optimizer/ \
  ~/.codex/skills/prompt-compliance-optimizer/
```

Windows PowerShell 再执行：

```powershell
$dest = Join-Path $HOME ".codex\skills\prompt-compliance-optimizer"
Copy-Item "skills\prompt-compliance-optimizer\*" $dest -Recurse -Force
```

更新后重新打开 Codex，确保新版本被加载。

## 8. 检查是否安装成功

确认下面的文件存在：

macOS 或 Linux：

```bash
test -f ~/.codex/skills/prompt-compliance-optimizer/SKILL.md && echo "installed"
```

Windows PowerShell：

```powershell
Test-Path "$HOME\.codex\skills\prompt-compliance-optimizer\SKILL.md"
```

Windows 输出 `True`，或者 macOS/Linux 输出 `installed`，表示文件已经安装。之后在 Codex 中输入 `$prompt-compliance-optimizer` 测试触发。

## 9. 数据安全要求

不要提交任何真实对话、benchmark 输入、运行日志、模型请求、生成报告、API key 或 token。

仓库已经默认忽略：

```text
data/
logs/
outputs/
.env*
.streamlit/secrets.toml
```

建议把本地运行数据全部放入 `data/`，生成结果放入 `outputs/`，日志放入 `logs/`。push 前执行：

```bash
git status --short
git ls-files data logs outputs
```

第二条命令必须没有输出。如果有输出，说明运行数据已被 Git 跟踪，需要先从 Git 索引移除：

```bash
git rm -r --cached --ignore-unmatch data logs outputs
```

该命令不会删除本地文件。

`.gitignore` 只能阻止未来提交。如果运行数据曾经被 push 到 GitHub，它们仍可能存在于 Git 历史中。此时必须联系仓库管理员清理历史，并轮换日志中可能出现过的 API key、token 或内部凭据。

## 10. 常见问题

### Codex 找不到 skill

确认 `~/.codex/skills/prompt-compliance-optimizer/SKILL.md` 存在，然后重新打开 Codex。

### Skill 找不到应用代码

确认已经克隆完整仓库，并从 `badcase_detect_agent` 仓库根目录启动 Codex。

### GitHub 返回权限错误

确认自己的 GitHub 账号有仓库权限，并检查 HTTPS 凭据、SSH key、`GITHUB_TOKEN` 或 `GH_TOKEN`。

### 更新后仍然使用旧版本

执行 `git pull` 后重新同步 skill 目录，然后重启 Codex。GitHub Skill Installer 不会自动覆盖已有安装。
