#!/usr/bin/env bash
set -euo pipefail

print_usage() {
    echo "Usage: ${0} [-h] [experiment_name]"
    echo
    echo "Run TradeRL analytics."
    echo
    echo "Positional argument:"
    echo "  experiment_name        The name of the experiment to run analytics for."
    echo
    echo "Optional arguments:"
    echo "  -h, --help             Show this help message and exit."
}

EXPERIMENT_NAME=

main() {
    parse_args "$@"
    check_uv_install

    echo "Running TradeRL Analytics For Experiment: ${EXPERIMENT_NAME}..."
    uv run scripts/run_analytics.py --experiment-name "${EXPERIMENT_NAME}"
}

parse_args() {
    local i=0

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help|help)
                print_usage
                exit 0
                ;;
            *)
                if [[ $i -eq 0 ]]; then
                    EXPERIMENT_NAME="$1"
                    i=1
                fi
                shift
                ;;
        esac
    done

    [[ -z "${EXPERIMENT_NAME}" ]] && printf "Requires EXPERIMENT_NAME\n\n" >&2 && print_usage >&2 && exit 1
    return 0
}

check_uv_install() {
    echo -n "Checking uv install..."
    if command -v uv >/dev/null; then
        echo "Ok."
    else
        printf "\nuv not installed! Try issuing 'pip install uv' and re-running\n\n" >&2 && exit 1
    fi
}

main "$@"
