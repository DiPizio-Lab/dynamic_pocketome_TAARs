#!/bin/bash

#SBATCH --job-name=meta_analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --partition=standard
#SBATCH --cpus-per-task=46
#SBATCH --nodelist=TUZELSB-MMLAB-ALANINE
#SBATCH --time=0-23:50:00
#SBATCH --output=meta_analysis_%j.log  

# Activate Conda
source /opt/anaconda3/bin/activate
eval "$(conda shell.bash hook)"
conda activate md_env
echo "conda activated"

# prj_ls=$(find apo_structures holo_structures -type d -path "*/*/[0-9]/pockets_dens")
# echo "$prj_ls"
# Run the script with arguments
# python comparative_study.py $prj_ls

python comp_with_sel.py --across-genes >across_genes.log 2>&1


conda deactivate
echo "conda deactivated"
