# Terminal demo for the README. Requires: brew install vhs, ollama pull qwen3:8b,
# and seeded runs (uv run python -m training.seed_runs). Run from the repo root:
#   vhs docs/demo.tape
Output docs/demo.gif

Require uv
Require ollama

Set Shell zsh
Set FontSize 16
Set Width 1300
Set Height 800
Set Padding 20
Set TypingSpeed 40ms
Set Theme "Catppuccin Mocha"

Hide
Type "export RUNDOCTOR_LOG_LEVEL=WARNING && clear"
Enter
Show

Type "uv run rundoctor-chat --model qwen3:8b"
Enter
Sleep 4s

Type "What's wrong with my runs? Diagnose each one."
Enter
# qwen3 thinks before each tool call; allow time for list_runs + 4x diagnose_run.
Sleep 90s

Type "Relaunch run 1 with a learning rate that should work."
Enter
Sleep 40s

Type "exit"
Enter
Sleep 1s
