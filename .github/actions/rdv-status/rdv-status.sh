#!/usr/bin/env bash

set -euo pipefail

readonly REPOSITORY_API="https://api.gitcode.com/api/v5/repos/cann/catlass"
readonly STATUS_LABELS=(RDV-SUCC RDV-RUNNING RDV-FAIL)

if [[ ! "${PR_NUMBER}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PR_NUMBER must be a positive integer" >&2
    exit 1
fi
if [[ -z "${GITCODE_TOKEN}" ]]; then
    echo "GITCODE_TOKEN is required to update pull request status" >&2
    exit 1
fi
if [[ -z "${RDV_RESULT_URL}" ]]; then
    echo "result-url is required" >&2
    exit 1
fi

readonly LABELS_URL="${REPOSITORY_API}/pulls/${PR_NUMBER}/labels"
readonly COMMENTS_URL="${REPOSITORY_API}/pulls/${PR_NUMBER}/comments"

gitcode_api() {
    local method="$1"
    local url="$2"
    shift 2
    curl --fail --silent --show-error --location \
        --request "${method}" \
        --header "PRIVATE-TOKEN: ${GITCODE_TOKEN}" \
        "${url}" "$@"
}

remove_status_labels() {
    local current_labels label
    current_labels="$(gitcode_api GET "${LABELS_URL}")"
    for label in "${STATUS_LABELS[@]}"; do
        if jq -e --arg label "${label}" \
            'any(.[]; .name == $label)' <<<"${current_labels}" >/dev/null; then
            gitcode_api DELETE "${LABELS_URL}/${label}" >/dev/null
        fi
    done
}

add_label() {
    local label="$1"
    gitcode_api POST "${LABELS_URL}" \
        --header "Content-Type: application/json" \
        --data "[\"${label}\"]" >/dev/null
}

case "${RDV_PHASE}" in
    start)
        remove_status_labels
        add_label RDV-RUNNING

        comment_body="🚀 RDV Started: ${RDV_RESULT_URL}"
        comment_json="$(jq -cn --arg body "${comment_body}" '{body: $body}')"
        comment_response="$(gitcode_api POST "${COMMENTS_URL}" \
            --header "Content-Type: application/json" \
            --data "${comment_json}")"
        comment_id="$(jq -er '.id' <<<"${comment_response}")"
        echo "comment-id=${comment_id}" >>"${GITHUB_OUTPUT}"
        echo "Set PR #${PR_NUMBER} to RDV-RUNNING and published comment ${comment_id}"
        ;;
    finish)
        if [[ "${RDV_RESULT}" == "success" ]]; then
            result_label=RDV-SUCC
            result_text="✅ RDV Succeeded"
        else
            result_label=RDV-FAIL
            result_text="❌ RDV Failed"
        fi

        remove_status_labels
        add_label "${result_label}"

        if [[ ! "${RDV_COMMENT_ID}" =~ ^[0-9A-Za-z_-]+$ ]]; then
            echo "A valid comment-id from the start phase is required" >&2
            exit 1
        fi
        comment_body="${result_text}: ${RDV_RESULT_URL}"
        comment_json="$(jq -cn --arg body "${comment_body}" '{body: $body}')"
        gitcode_api PATCH "${REPOSITORY_API}/pulls/comments/${RDV_COMMENT_ID}" \
            --header "Content-Type: application/json" \
            --data "${comment_json}" >/dev/null
        echo "Set PR #${PR_NUMBER} label to ${result_label} and updated comment ${RDV_COMMENT_ID}"
        ;;
    *)
        echo "phase must be start or finish" >&2
        exit 1
        ;;
esac
