#!/usr/bin/env bash
# Double click this file in Finder. It opens Terminal and pushes the project
# to GitHub. A .command file starts in your home folder, so it moves to its
# own folder first.
cd "$(dirname "$0")"

echo "SSAC race analysis"
echo "Pushing to github.com/theoq71/ssac-video-analysis"
echo

if ./push-to-github.sh; then
  echo
  echo "Finished."
else
  echo
  echo "Stopped with an error. The message above says why."
fi

echo
echo "Press any key to close this window."
read -n 1 -s
