#!/bin/bash
# Parameter sweep as a SLURM job array: one run per wall contact angle,
# each writing its own run file. SLURM sends SIGTERM before the wall-time
# limit; droplet-relax then finishes the iteration and writes a checkpoint,
# so a run that did not finish can be continued with
#     droplet-relax restart angle_<theta>.h5
#
#     sbatch examples/slurm_contact_angle_sweep.sh
#
#SBATCH --job-name=droplet-sweep
#SBATCH --array=0-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --signal=TERM@300

ANGLES=(60 75 90 105 120)
THETA=${ANGLES[$SLURM_ARRAY_TASK_ID]}

# one thread per run: the job array provides the parallelism
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

srun droplet-relax run examples/default.toml -o "angle_${THETA}.h5" --quiet \
    --set "solver.contact_angle_deg=${THETA}"
