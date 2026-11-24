#!/usr/bin/env bash
# require-tests-passed.sh checks verbose `go test -v` output. It fails unless
# every named test has a PASS line. A skipped test or a test that the run
# filter missed has no PASS line, so it fails too.
#
# Usage: require-tests-passed.sh <go-test-v-output> <TestName>...
set -euo pipefail

output="$1"
shift

missing=0
for name in "$@"; do
  if ! grep -E -q -- "--- PASS: ${name} \(" "${output}"; then
    printf 'test %s did not pass\n' "${name}" >&2
    missing=1
  fi
done
exit "${missing}"
