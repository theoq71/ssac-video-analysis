#!/usr/bin/env bash
# The push logic for this project. Started by "Push to GitHub.command".
set -euo pipefail
cd "$(dirname "$0")"

OWNER="theoq71"
REPO="ssac-video-analysis"
NAME="theoq71"
EMAIL="133422938+theoq71@users.noreply.github.com"

echo "Clearing git leftovers from bridge sessions..."
find .git -name "*.lock" -delete 2>/dev/null || true
find .git -name "tmp_obj_*" -delete 2>/dev/null || true

if [ ! -d .git ]; then
  git init -b main >/dev/null
fi

git add -A
if git diff --cached --quiet; then
  echo "Nothing new to commit."
else
  git -c user.name="$NAME" -c user.email="$EMAIL" \
    commit -m "Update $(date '+%Y-%m-%d %H:%M')" >/dev/null
  echo "Committed the latest changes."
fi

echo "Checking nothing private is about to be published..."
if git ls-files | grep -qiE "settings\.json|\.env|password|secret|token"; then
  echo "STOP: a file that looks private is tracked. Not pushing." >&2
  git ls-files | grep -iE "settings\.json|\.env|password|secret|token" >&2
  exit 1
fi
echo "  clean: $(git ls-files | wc -l | tr -d ' ') files, none private"

if git remote get-url origin >/dev/null 2>&1; then
  git push --force -u origin main
elif command -v gh >/dev/null 2>&1; then
  echo "Creating $OWNER/$REPO as a public repo and pushing..."
  gh repo create "$OWNER/$REPO" --public --source=. --remote=origin --push
else
  echo "Connecting to github.com/$OWNER/$REPO and pushing..."
  git remote add origin "https://github.com/$OWNER/$REPO.git"
  git push -u origin main
fi

echo
echo "Done: https://github.com/$OWNER/$REPO"
