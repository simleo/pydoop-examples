[ -n "${PYDEEP_COMMON:-}" ] && return || readonly PYDEEP_COMMON=1

use_two_classes () {
    find "${1}" -mindepth 1 -maxdepth 1 -type d | tail -n +3 | xargs rm -rf
}

downsample () {
    local d=$1
    shopt -q -o xtrace && local reset_x=1
    set +x
    find "${d}" -mindepth 2 -maxdepth 2 -type f | while read f; \
	do [ $(( RANDOM % 20 )) -gt 1 ] && rm -f "${f}";
    done
    [ -n ${reset_x} ] && set -x
}

utc_timestamp_ns () {
    date +%s%N
}

export -f use_two_classes downsample utc_timestamp_ns

export BNECKS_DIR="${BNECKS_DIR:-bottlenecks}"
export CMP_INPUT_DIR="${CMP_INPUT_DIR:-cmp_input}"
export CMP_NUM_MAPS="${CMP_NUM_MAPS-2}"
export CMP_OUTPUT_DIR="${CMP_OUTPUT_DIR:-cmp_results}"
export GENBNECKS_NUM_MAPS="${GENBNECKS_NUM_MAPS-4}"
export KERAS_INPUT_DIR="${KERAS_INPUT_DIR-keras_input}"
export KERAS_NUM_MAPS="${KERAS_NUM_MAPS-1}"
export KERAS_OUTPUT_DIR="${KERAS_OUTPUT_DIR-keras_output}"
export LIBHDFS_OPTS="${LIBHDFS_OPTS:--Xmx512m}"
export NUM_STEPS="${NUM_STEPS:-400}"
export PLOTS_DIR="${PLOTS_DIR:-training_plots}"
export PYTHON="${PYTHON:-python3}"
export RETRAIN_NUM_MAPS="${RETRAIN_NUM_MAPS-4}"
export TEST_NUM_MAPS="${TEST_NUM_MAPS-2}"
export TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR:-test_results}"
export TRAINED_MODELS_DIR="${TRAINED_MODELS_DIR:-trained_models}"

if [ -n "${DEBUG:-}" ]; then
    LOG_LEVEL="DEBUG"
else
    LOG_LEVEL="INFO"
fi
export LOG_LEVEL
