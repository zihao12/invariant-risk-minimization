#!/bin/bash
#SBATCH --mail-user=wangzh@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/wangzh/git/invariant-risk-minimization/output/%j.%N.stdout
#SBATCH --error=/home/wangzh/git/invariant-risk-minimization/output/%j.%N.stderr
#SBATCH --chdir=/home/wangzh/git/invariant-risk-minimization
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --account=pi-vveitch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --job-name=lab-gpu

export PATH="/home/wangzh/miniconda3/bin:$PATH"

i=${1}
method=${2}
maxiter=${3}
out_result=${4}
out_model=${5}
out_print=${6}

echo "python -u main.py --i $i --method $method --maxiter $maxiter --out_result $out_result --out_model $out_model > $out_print"
# python main.py --i $i --method $method --maxiter $maxiter --out_result $out_result --out_model $out_model
python -u main.py --i $i --method $method --maxiter $maxiter --out_result $out_result --out_model $out_model > $out_print
