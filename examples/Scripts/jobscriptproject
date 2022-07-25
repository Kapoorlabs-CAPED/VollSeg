#!/bin/bash
#!/bin/bash 
#SBATCH --nodes=1             # Number of nodes 
##SBATCH --partition=gpu_p3

#SBATCH --gres=gpu:1         # Allocate 4 GPUs per node
#SBATCH -C v100-32g

#SBATCH --job-name=oneat                # Jobname 
#SBATCH --cpus-per-task=48
#SBATCH --output=oneat.o%j            # Output file 
#SBATCH --error=oneat.o%j            # Error file 
#SBATCH --time=20:00:00       # Expected runtime HH:MM:SS (max 100h)
module purge # purging modules inherited by default

module load tensorflow-gpu/py3/2.7.0
module load anaconda-py3/2020.11
#conda init bash # deactivating environments inherited by default
conda deactivate
conda activate naparienv
set -x # activating echo of

srun python -u TrainProjectionModel.py
