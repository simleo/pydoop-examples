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

export -f use_two_classes downsample
