# Repository Instructions

## Skill source of truth

- Treat each `skills/<skill-name>/` directory in this repository as the only editable source for that skill.
- Never directly edit an installed copy under `$CODEX_HOME/skills/` or `~/.codex/skills/`.
- For a skill change, edit the repository copy first, validate it, confirm the change appears in `git status`, and only then mirror the repository copy to the installed location.
- Synchronize skills in one direction only: repository to installed copy. Do not copy an installed skill back into the repository unless the user explicitly requests recovery and the diff has been reviewed first.
- Keep every skill implementation file that belongs to the skill, including `SKILL.md`, `agents/`, `assets/`, `references/`, and `scripts/`, in Git.
- Do not add generated conversations, benchmark data, traces, logs, runtime outputs, or analysis/experiment reports to Git. Keep those artifacts under ignored output locations such as `data/`, `logs/`, or `outputs/`, or in the user's explicitly selected report directory.

