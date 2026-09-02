#!/usr/bin/env bash

set -eo pipefail

if [[ ! -d "${RDV_WORKTREE}" ]]; then
    echo "CATLASS worktree does not exist: ${RDV_WORKTREE}" >&2
    exit 1
fi

if [[ -f /workspace/Ascend/cann/set_env.sh ]]; then
    cann_env=/workspace/Ascend/cann/set_env.sh
elif [[ -f /usr/local/Ascend/cann/set_env.sh ]]; then
    cann_env=/usr/local/Ascend/cann/set_env.sh
else
    echo "CANN environment script was not found in either fallback path" >&2
    exit 1
fi

echo "Activating CANN environment from ${cann_env}"
# shellcheck source=/dev/null
source "${cann_env}"

if [[ -f /workspace/miniforge3/etc/profile.d/conda.sh ]]; then
    conda_env_script=/workspace/miniforge3/etc/profile.d/conda.sh
elif [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
    conda_env_script="${HOME}/miniforge3/etc/profile.d/conda.sh"
else
    echo "Conda was not found in either fallback path" >&2
    exit 1
fi

echo "Activating Conda from ${conda_env_script}"
# shellcheck source=/dev/null
source "${conda_env_script}"
conda activate catlass-ci-py311-torch290

cd "${RDV_WORKTREE}"
case "${CATLASS_TEST_SUITE}" in
    dsl)
        export CATLASS_DSL_PREBUILT_ASCENDNPU_IR=/workspace/AscendNPU-IR
        bash tests/run_dsl_test.sh
        ;;
    all)
        bash tests/run_all_test.sh 3510
        ;;
    *)
        echo "suite must be dsl or all" >&2
        exit 1
        ;;
esac
