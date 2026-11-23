#!/usr/bin/env bash
# Run from the A100 environment with: bash scripts/run_fig9_skin_scan.sh
set -euo pipefail
cd "$(dirname "$0")/.."

device=${FIG9_DEVICE:-cuda}
python_bin=${FIG9_PYTHON:-python}
run_id=${FIG9_RUN_ID:-$(date -u +%Y%m%d_%H%M%S)_fig9_skin_profiles}
if [[ ! "$run_id" =~ ^[[:alnum:]][[:alnum:]_.-]*$ ]]; then
    echo "Invalid FIG9_RUN_ID: $run_id" >&2
    exit 2
fi
run_dir="outputs/$run_id"
archive="outputs/_downloads/$run_id.tar.gz"
mkdir -p outputs/_downloads
if [[ -e "$archive" ]]; then
    echo "Download archive already exists: $archive" >&2
    exit 1
fi
mkdir "$run_dir"

finish() {
    task_rc=$?
    trap - EXIT
    set +e
    status=complete
    if (( task_rc != 0 )); then status=failed; fi
    printf 'status=%s\nexit_code=%d\nfinished_utc=%s\n' "$status" "$task_rc" \
        "$(date -u +%FT%TZ)" >> "$run_dir/run_info.txt"
    tar -czf "$archive.part" -C outputs "$run_id" && mv "$archive.part" "$archive"
    archive_rc=$?
    if (( archive_rc == 0 )); then
        printf 'Run %s. Download: %s/%s\n' "$status" "$PWD" "$archive"
        printf 'After downloading, run from the local project root:\n  mkdir -p outputs\n  tar -xzf "$HOME/Downloads/%s.tar.gz" -C outputs\n' "$run_id"
    else
        printf 'Packaging failed (exit %d). Results: %s/%s\n' "$archive_rc" "$PWD" "$run_dir" >&2
    fi
    if (( task_rc != 0 )); then exit "$task_rc"; fi
    exit "$archive_rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

{
    printf 'run_id=%s\nstarted_utc=%s\nworking_directory=%s\n' "$run_id" "$(date -u +%FT%TZ)" "$PWD"
    printf 'command=FIG9_PYTHON=%q FIG9_DEVICE=%q FIG9_RUN_ID=%q bash scripts/run_fig9_skin_scan.sh\n' \
        "$python_bin" "$device" "$run_id"
    printf 'git_commit='
    git rev-parse HEAD
    printf 'input_config=configs/shape_scan_a100.yaml\ndeformation_reference=configs/fig9/reference_grid.csv\nexisting_skin_points=configs/fig9/existing_skin_points.csv\n'
} > "$run_dir/run_info.txt"

cp configs/fig9/reference_grid.csv "$run_dir/reference_grid.csv"
cp configs/fig9/existing_skin_points.csv "$run_dir/existing_skin_points.csv"
"$python_bin" -u scripts/calculate_fig9_skin_points.py --device "$device" --output "$run_dir" \
    2>&1 | tee "$run_dir/run.log"
"$python_bin" -u scripts/plot_fig9_skin_scan.py --reference "$run_dir/reference_grid.csv" \
    --existing-skin "$run_dir/existing_skin_points.csv" --new-points "$run_dir/new_skin_points.csv" \
    --output "$run_dir" 2>&1 | tee -a "$run_dir/run.log"
