#!/usr/bin/env bash
set -euo pipefail

print_usage() {
    echo "Usage: ${0} [-h] [config_file]"
    echo
    echo "Run TradeRL with a specific configuration."
    echo
    echo "Positional argument:"
    echo "  config_file            The path of the config file to use (default='config/baselines/random.yaml)."
    echo
    echo "Optional arguments:"
    echo "  -h, --help             Show this help message and exit."
}

CONFIG_FILE="./config/baselines/random.yaml"

main() {
    parse_args "$@"
    check_uv_install

    echo "Running TradeRL with config: ${CONFIG_FILE}..."
    uv run trade_rl/main.py --config-file "${CONFIG_FILE}"
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
                    CONFIG_FILE="$1"
                    i=1
                fi
                shift
                ;;
        esac
    done

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
