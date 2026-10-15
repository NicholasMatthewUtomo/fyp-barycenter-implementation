#!/bin/bash
cd -- "$(dirname -- "$0")" || exit 1
/Users/jiaqiwang/opt/anaconda3/bin/python -u laptop_launcher.py --mode classify "$@"
run_exit_code=$?
if [[ -t 0 ]]; then
    printf '\nPress Return to close this launcher. '
    read -r
fi
exit "$run_exit_code"
