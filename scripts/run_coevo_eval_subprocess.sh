#!/bin/bash
# Evaluate co-evolved defense using subprocess-per-attack with OS-level timeout.
# Each attack runs in its own Python process. If it hangs, `timeout` kills it.
set -euo pipefail

CHECKPOINT="${1:-experiments/results/exp4/run_20260216_113117}"
GOALS="${2:-10}"
SEED="${3:-42}"
ATTACKS="${4:-pair,tap,mapelites}"
TIMEOUT_S="${5:-300}"  # 5 minutes per attack
OUTPUT_DIR="experiments/results/exp4"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="${OUTPUT_DIR}/coevo_eval_${SEED}_${TIMESTAMP}.json"
TMPDIR=$(mktemp -d)

cd "$(dirname "$0")/.."
source ../venv311/bin/activate

echo "=== Co-Evolution Subprocess Evaluation ==="
echo "Checkpoint: ${CHECKPOINT}"
echo "Goals: ${GOALS}, Seed: ${SEED}"
echo "Attacks: ${ATTACKS}, Timeout: ${TIMEOUT_S}s"
echo "Temp dir: ${TMPDIR}"
echo ""

# Load test goals from Python
python3 -c "
import csv, random, json, sys
from huggingface_hub import hf_hub_download
p = hf_hub_download('JailbreakBench/JBB-Behaviors', 'data/harmful-behaviors.csv', repo_type='dataset')
harmful = []
with open(p, newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        harmful.append({'goal': row['Goal'], 'category': row.get('Category', 'unknown')})
rng = random.Random(${SEED})
by_cat = {}
for g in harmful:
    by_cat.setdefault(g['category'], []).append(g)
train, val, test = [], [], []
for cat in sorted(by_cat):
    items = list(by_cat[cat])
    rng.shuffle(items)
    n = len(items)
    nt, nv = max(1, int(n*0.6)), max(1, int(n*0.2))
    train.extend(items[:nt])
    val.extend(items[nt:nt+nv])
    test.extend(items[nt+nv:])
json.dump(test[:${GOALS}], sys.stdout)
" > "${TMPDIR}/goals.json"

N_GOALS=$(python3 -c "import json; print(len(json.load(open('${TMPDIR}/goals.json'))))")
echo "Loaded ${N_GOALS} test goals"
echo ""

# Results accumulator
echo '{"results": {}}' > "${TMPDIR}/all_results.json"

IFS=',' read -ra ATTACK_LIST <<< "${ATTACKS}"
T0=$(date +%s)

for ATTACK in "${ATTACK_LIST[@]}"; do
    ATTACK_UP=$(echo "$ATTACK" | tr '[:lower:]' '[:upper:]')
    echo "--- Running ${ATTACK_UP} on ${N_GOALS} goals ---"
    SUCCESSES=0
    TOTAL=0

    for i in $(seq 0 $((N_GOALS - 1))); do
        GOAL=$(python3 -c "import json; print(json.load(open('${TMPDIR}/goals.json'))[$i]['goal'])")
        GOAL_SHORT=$(echo "$GOAL" | head -c 60)
        RESULT_TMP="${TMPDIR}/${ATTACK}_${i}.json"

        echo -n "  [${ATTACK_UP}] [$((i+1))/${N_GOALS}] ${GOAL_SHORT}... "

        # Run attack in subprocess with OS timeout
        if timeout "${TIMEOUT_S}" python -u scripts/run_single_attack.py \
            --checkpoint "${CHECKPOINT}" \
            --goal "${GOAL}" \
            --attack "${ATTACK}" \
            --output "${RESULT_TMP}" 2>/dev/null; then
            # Attack completed within timeout
            if [ -f "${RESULT_TMP}" ]; then
                SUCCESS=$(python3 -c "import json; print(json.load(open('${RESULT_TMP}'))['success'])")
                SR=$(python3 -c "import json; print(f\"{json.load(open('${RESULT_TMP}'))['strongreject_score']:.2f}\")")
                WALL=$(python3 -c "import json; print(f\"{json.load(open('${RESULT_TMP}'))['wall_time_s']:.0f}\")")
                if [ "$SUCCESS" = "True" ]; then
                    echo "BYPASS (SR=${SR}, ${WALL}s)"
                    SUCCESSES=$((SUCCESSES + 1))
                else
                    echo "BLOCKED (SR=${SR}, ${WALL}s)"
                fi
            else
                echo "NO OUTPUT"
                echo '{"success": false, "raw_success": false, "strongreject_score": 0.0, "wall_time_s": 0, "error": "no_output"}' > "${RESULT_TMP}"
            fi
        else
            # Timed out or failed
            echo "TIMEOUT (${TIMEOUT_S}s)"
            echo "{\"success\": false, \"raw_success\": false, \"strongreject_score\": 0.0, \"wall_time_s\": ${TIMEOUT_S}, \"timed_out\": true, \"goal\": \"$(echo "$GOAL" | head -c 200)\"}" > "${RESULT_TMP}"
        fi

        TOTAL=$((TOTAL + 1))
    done

    ASR=$(python3 -c "print(f'{${SUCCESSES}/${N_GOALS}*100:.1f}')")
    echo "  ${ATTACK_UP} ASR: ${ASR}% (${SUCCESSES}/${N_GOALS})"
    echo ""
done

# Combine all results
echo "--- Combining results ---"
python3 -c "
import json, glob, os, time
results = {}
tmpdir = '${TMPDIR}'
attacks = '${ATTACKS}'.split(',')
for attack in attacks:
    atk_results = []
    for i in range(${N_GOALS}):
        f = os.path.join(tmpdir, f'{attack}_{i}.json')
        if os.path.exists(f):
            atk_results.append(json.load(open(f)))
        else:
            atk_results.append({'success': False, 'strongreject_score': 0.0, 'wall_time_s': 0, 'error': 'missing'})
    asr = sum(1 for r in atk_results if r.get('success')) / max(1, len(atk_results))
    results[attack] = {'asr': asr, 'results': atk_results}

summary = {
    'exp_name': 'exp4_coevo_eval_subprocess',
    'seed': ${SEED},
    'timestamp': '${TIMESTAMP}',
    'checkpoint': '${CHECKPOINT}',
    'n_goals': ${N_GOALS},
    'attack_types': attacks,
    'wall_time_s': time.time() - ${T0},
    'timeout_per_attack_s': ${TIMEOUT_S},
    'coevolved': results,
}
json.dump(summary, open('${RESULT_FILE}', 'w'), indent=2, default=str)
print(f'Results saved: ${RESULT_FILE}')

print()
print('=' * 60)
print('SUMMARY')
print('=' * 60)
print(f\"{'Attack':<12} {'ASR':>12}\")
print('-' * 28)
for attack in attacks:
    asr = results[attack]['asr']
    print(f'{attack:<12} {asr*100:>11.1f}%')
"

# Cleanup
rm -rf "${TMPDIR}"
echo ""
echo "Total wall time: $(( $(date +%s) - T0 ))s"
