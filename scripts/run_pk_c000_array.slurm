#!/bin/bash
#SBATCH -J pk_c000
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-499%40
#SBATCH -o /pscratch/sd/v/vtorresg/hod-astra-box500/c000/logs/slurm_pk_%A_%a.out
#SBATCH -e /pscratch/sd/v/vtorresg/hod-astra-box500/c000/logs/slurm_pk_%A_%a.err

set -eo pipefail

cd /global/homes/v/vtorresg/fisher-astra

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u

PY=python
ROOT=/pscratch/sd/v/vtorresg/hod-astra-box500
COSMO=c000

mkdir -p "$ROOT/$COSMO/logs"

HOD=$($PY - <<'PY'
import csv, os
idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
with open('params/params_hods_c000_ph000_seed0.csv', newline='') as f:
    rows = list(csv.DictReader(f))
print(rows[idx]['hod'])
PY
)

HOD_NUM=${HOD#hod}
PK="$ROOT/power_spectra/$HOD/$COSMO/pk_all_zone_${COSMO}_ph000_seed0_hod${HOD_NUM}.csv"

if [[ -s "$PK" ]]; then
    echo "[skip] P(k) already exists for ${COSMO}_${HOD}"
    exit 0
fi

echo "[run] P(k) cosmo=$COSMO hod=$HOD"
"$PY" src/calc_pk.py \
  --data-root "$ROOT" \
  --cosmo "$COSMO" \
  --hod "$HOD" \
  --field all \
  --grid 256 \
  --kmax 0.5 \
  --no-plot \
  --skip-existing
