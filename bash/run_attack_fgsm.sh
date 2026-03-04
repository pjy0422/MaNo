#!/bin/bash

# ==========================================
# [SBATCH 옵션]
# ==========================================
#SBATCH --job-name=MANO_fgsm
#SBATCH --partition=RTX3090
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --exclude=n68
#
# MaNo FGSM Attack (1 step): 3 architectures x 3 datasets, parallelized across 3 GPUs.
#
# Usage:
#   bash bash/run_attack_fgsm.sh

DATA_ROOT=./datasets
RESULTS_DIR="${RESULTS_DIR:-results_attack}"

ATTACK_MODE="fgsm"
ATTACK_EPS="${ATTACK_EPS:-0.031372549}"    # 8/255
ATTACK_ALPHA="${ATTACK_ALPHA:-0.007843137}" # 2/255
ATTACK_STEPS=1
ATTACK_GAMMA="${ATTACK_GAMMA:-0.05}"
ATTACK_TAU="${ATTACK_TAU:-0.01}"

# wandb
WANDB_PROJECT="${WANDB_PROJECT:-MaNo-Attack}"
WANDB_GROUP="${WANDB_GROUP:-attack_comparison-fgsm}"
WANDB_TAGS="${WANDB_TAGS:-attack,fgsm}"

ATTACK_ARGS="--attack_mode ${ATTACK_MODE} --attack_eps ${ATTACK_EPS} --attack_alpha ${ATTACK_ALPHA} --attack_steps ${ATTACK_STEPS} --attack_gamma ${ATTACK_GAMMA} --attack_tau ${ATTACK_TAU}"

COMMON="--lr 0.001 --batch_size 128 --seed 1 --norm_type 4 --results_dir ${RESULTS_DIR} --wandb_project ${WANDB_PROJECT} --wandb_group ${WANDB_GROUP} --wandb_tags ${WANDB_TAGS} ${ATTACK_ARGS}"

echo "============================================================"
echo "MaNo Attack Comparison (GPU 0,1,2 parallel)"
echo "Mode: ${ATTACK_MODE}"
echo "  eps=${ATTACK_EPS}"
echo "W&B: project=${WANDB_PROJECT}, group=${WANDB_GROUP}"
echo "============================================================"

# GPU 0: resnet18
(
    echo "[GPU 0] resnet18 started"
    for DATA in cifar10 cifar100 tinyimagenet; do
        if [ "$DATA" = "cifar10" ]; then
            DPATH=${DATA_ROOT}/cifar10; CPATH=${DATA_ROOT}/CIFAR-10-C
        elif [ "$DATA" = "cifar100" ]; then
            DPATH=${DATA_ROOT}/cifar100; CPATH=${DATA_ROOT}/CIFAR-100-C
        else
            DPATH=${DATA_ROOT}/tinyimagenet; CPATH=${DATA_ROOT}/TinyImageNet-C
        fi
        echo "[GPU 0] resnet18 / ${DATA}"
        python main_attack.py --arch resnet18 --severity -1 --dataname ${DATA} \
            --gpu 0 --cifar_data_path ${DPATH} --cifar_corruption_path ${CPATH} ${COMMON}
    done
    echo "[GPU 0] resnet18 done"
) &
pid0=$!

# GPU 1: resnet50
(
    echo "[GPU 1] resnet50 started"
    for DATA in cifar10 cifar100 tinyimagenet; do
        if [ "$DATA" = "cifar10" ]; then
            DPATH=${DATA_ROOT}/cifar10; CPATH=${DATA_ROOT}/CIFAR-10-C
        elif [ "$DATA" = "cifar100" ]; then
            DPATH=${DATA_ROOT}/cifar100; CPATH=${DATA_ROOT}/CIFAR-100-C
        else
            DPATH=${DATA_ROOT}/tinyimagenet; CPATH=${DATA_ROOT}/TinyImageNet-C
        fi
        echo "[GPU 1] resnet50 / ${DATA}"
        python main_attack.py --arch resnet50 --severity -1 --dataname ${DATA} \
            --gpu 1 --cifar_data_path ${DPATH} --cifar_corruption_path ${CPATH} ${COMMON}
    done
    echo "[GPU 1] resnet50 done"
) &
pid1=$!

# GPU 2: wrn_50_2
(
    echo "[GPU 2] wrn_50_2 started"
    for DATA in cifar10 cifar100 tinyimagenet; do
        if [ "$DATA" = "cifar10" ]; then
            DPATH=${DATA_ROOT}/cifar10; CPATH=${DATA_ROOT}/CIFAR-10-C
        elif [ "$DATA" = "cifar100" ]; then
            DPATH=${DATA_ROOT}/cifar100; CPATH=${DATA_ROOT}/CIFAR-100-C
        else
            DPATH=${DATA_ROOT}/tinyimagenet; CPATH=${DATA_ROOT}/TinyImageNet-C
        fi
        echo "[GPU 2] wrn_50_2 / ${DATA}"
        python main_attack.py --arch wrn_50_2 --severity -1 --dataname ${DATA} \
            --gpu 2 --cifar_data_path ${DPATH} --cifar_corruption_path ${CPATH} ${COMMON}
    done
    echo "[GPU 2] wrn_50_2 done"
) &
pid2=$!

echo "Launched: GPU0=resnet18(${pid0}) GPU1=resnet50(${pid1}) GPU2=wrn_50_2(${pid2})"
wait ${pid0} ${pid1} ${pid2}

echo ""
echo "============================================================"
echo "All attack comparisons complete. Results in ${RESULTS_DIR}/"
echo "============================================================"

ATK_SUFFIX="fgsm_eps$(printf '%.4f' ${ATTACK_EPS})"

echo "Generating 3x3 grid scatter plot..."
python combine_grid.py --results_dir "${RESULTS_DIR}" --attack_suffix "${ATK_SUFFIX}"
echo "Done."
