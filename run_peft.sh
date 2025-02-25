#!/bin/bash
#SBATCH --job-name=First_test_dj17292n                # Job name
#SBATCH --output=output.log               # Standard output and error log
#SBATCH --error=error.log                 # Separate error log
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks (processes)
#SBATCH --cpus-per-task=1                 # Number of CPU cores per task
#SBATCH --mem=16G                          # Memory per node
#SBATCH --partition=compute               # Partition name
#SBATCH --mail-user=dj17292n@pace.edu # Email notifications
#SBATCH --mail-type=ALL                    # Notify on job start, end, fail

# Load modules (if required)
module load python/3.9

# Create and activate a virtual environment (optional but recommended)
python -m venv my_venv
source my_venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required packages from requirements.txt
pip install -r requirements.txt

# Run your application
python peft-research.py

# Deactivate the virtual environment after execution
deactivate
