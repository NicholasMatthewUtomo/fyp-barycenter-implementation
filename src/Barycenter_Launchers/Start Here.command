#!/bin/bash
cd "$(dirname "$0")" || exit 1
python3 -u laptop_launcher.py --mode menu "$@"
run_exit_code=$?
if [[ -t 0 ]]; then
    printf '\nPress Return to close this launcher. '
    read -r
fi
exit "$run_exit_code"
