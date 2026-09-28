#!/bin/bash
TOPIC="${1:-flappy-rl-naman}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Watcher started for scripts/train_all_remote.py (Topic: $TOPIC) at $(date)"

# Send confirmation ping
curl -s \
  -H "Title: Watcher Online 🎯" \
  -H "Priority: 5" \
  -H "Tags: white_check_mark,eyes" \
  -d "Watcher is actively monitoring Python training. You will be alerted the moment all runs finish!" \
  "https://ntfy.sh/$TOPIC"

# Loop while train_all_remote.py is running
while pgrep -f "scripts/train_all_remote.py" > /dev/null; do
    sleep 15
done

echo "Training completed at $(date)! Generating plots and tables..."

cd "$PROJECT_DIR"
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi
python scripts/make_plots.py

SUMMARY_MSG="All 9 training runs (Uniform, PER, Double DQN across seeds 0, 1, 2) have completed!"
if [ -f "$PROJECT_DIR/results/tables/method_comparison.csv" ]; then
    SUMMARY_MSG=$(cat "$PROJECT_DIR/results/tables/method_comparison.csv")
fi

# Send final push notification
curl -s \
  -H "Title: Flappy Bird RL Training Finished! 🏆" \
  -H "Priority: 5" \
  -H "Tags: trophy,party_popper,bird" \
  -d "Training finished on Azure VM! Models and plots generated.

$SUMMARY_MSG" \
  "https://ntfy.sh/$TOPIC"

echo "Notification dispatched to https://ntfy.sh/$TOPIC at $(date)"
