#!/usr/bin/env bash
# Stream Slurm job state transitions, one line per transition. Exits when all
# named jobs reach a terminal state. Intended for the orchestrator's Monitor tool.
set -u

JOBS=("$@")
if [ "${#JOBS[@]}" -eq 0 ]; then
  echo "usage: watch_jobs.sh JOBID..." >&2
  exit 2
fi

declare -A LAST=()
for j in "${JOBS[@]}"; do LAST[$j]=""; done

ITER=0
MAX_ITER=120   # 30s * 120 = 60 min cap

while [ "$ITER" -lt "$MAX_ITER" ]; do
  done_all=1
  for j in "${JOBS[@]}"; do
    state=$(sacct -j "$j" --noheader --parsable2 -o JobID,State 2>/dev/null | awk -F'|' -v j="$j" '$1==j{print $2; exit}')
    [ -z "$state" ] && state="UNKNOWN"
    if [ "$state" != "${LAST[$j]}" ]; then
      printf '[%s] job %s -> %s\n' "$(date -u +%H:%M:%SZ)" "$j" "$state"
      LAST[$j]="$state"
    fi
    case "$state" in
      COMPLETED|FAILED|CANCELLED|TIMEOUT|NODE_FAIL|BOOT_FAIL|DEADLINE|PREEMPTED|OUT_OF_MEMORY) : ;;
      *) done_all=0 ;;
    esac
  done
  if [ "$done_all" -eq 1 ]; then
    printf '[%s] all jobs terminal -- watch ending\n' "$(date -u +%H:%M:%SZ)"
    exit 0
  fi
  ITER=$((ITER+1))
  sleep 30
done
printf '[%s] watch_jobs.sh hit MAX_ITER cap (60 min) -- exiting\n' "$(date -u +%H:%M:%SZ)"
exit 0
