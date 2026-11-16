#!/usr/bin/env bash
# Run both full ECG200 classes independently in the active MOSEK environment.
set -u
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." || exit 1
ecg_python="${PYTHON:-python3}"
mkdir -p results || exit 1
"$ecg_python" -u -B scripts/run_ecg200.py --inspect > results/inspection.json || exit "$?"
ecg_failed=0
for ecg_class in normal abnormal; do
    printf 'Running full ECG200 %s class; log: results/%s.log\n' "$ecg_class" "$ecg_class"
    "$ecg_python" -u -B scripts/run_ecg200.py --class "$ecg_class" \
        --output "results/$ecg_class" > "results/$ecg_class.log" 2>&1
    ecg_status=$?
    printf '{"class":"%s","process_exit_code":%d}\n' "$ecg_class" "$ecg_status" \
        > "results/${ecg_class}_exit_status.json"
    if [ "$ecg_status" -ne 0 ]; then
        ecg_failed=1
        printf 'ECG200 %s failed with exit code %s; see log.\n' "$ecg_class" "$ecg_status" >&2
    fi
done
exit "$ecg_failed"
