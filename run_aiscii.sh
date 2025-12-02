#!/bin/bash
#
#SBATCH -A aiscii                   # allocation/account
#SBATCH -p aiscii                   # partition/queue
#SBATCH --job-name=run_model        # job name (appears in squeue)
#SBATCH --cpus-per-task=16          # CPU cores
#SBATCH --mem=128G                  # RAM
#SBATCH --gres=gpu:1 
#SBATCH --time=96:00:00             # wall‑clock limit (adjust)
#SBATCH --output=%x-%j.out          # stdout+stderr → run_model-<jobid>.out
#SBATCH --error=%x-%j.err           # stderr → run_model-<jobid>.err

# Store the job ID and node name
JOB_ID=$SLURM_JOB_ID
NODE_NAME=$SLURM_NODELIST

echo "Job $JOB_ID allocated to node: $NODE_NAME"


# SSH to the allocated node and run the commands there
ssh $NODE_NAME << 'EOF'

source /home/txr269/csds447/project/CSDS-447-Responsible-AI-Engineering-Project/447_env/bin/activate
module load Python/3.12.3-GCCcore-13.3.0

# Load your environment
cd /home/txr269/csds447/project/CSDS-447-Responsible-AI-Engineering-Project/
python generate_and_compare.py --num-images 1
EOF

# Don't cancel the job - let it finish naturally