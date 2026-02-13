RALPH_WORKER_MODEL="qwen2.5-coder:14b" \
RALPH_WORKER_PROVIDER="ollama" \
RALPH_REVIEWER_MODEL="claude-sonnet-4-20250514" \
RALPH_REVIEWER_PROVIDER="anthropic" \
~/.config/goose/recipes/ralph-loop.sh "Review and update this codebase, add comments, use deadcode and ruff to make ready for further development.  use ~/environments/ductile-discord/ as the venv"

