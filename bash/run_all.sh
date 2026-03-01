#!/usr/bin/env bash

gpu=$1
DATA_ROOT=./datasets

# ===== Step 1: Pre-training =====
echo "===== Step 1: Pre-training ====="
for ARCH in resnet18 resnet50 wrn_50_2
do
    echo "[CIFAR-10] Training ${ARCH}..."
    python init_base_model.py --arch ${ARCH} --train_epoch 20 --train_data_name cifar10 --lr 0.001 --batch_size 128 --seed 123 --gpu ${gpu} --cifar_data_path ${DATA_ROOT}/cifar10 --cifar_corruption_path ${DATA_ROOT}/CIFAR-10-C

    echo "[CIFAR-100] Training ${ARCH}..."
    python init_base_model.py --arch ${ARCH} --train_epoch 50 --train_data_name cifar100 --lr 0.001 --batch_size 128 --seed 123 --gpu ${gpu} --cifar_data_path ${DATA_ROOT}/cifar100 --cifar_corruption_path ${DATA_ROOT}/CIFAR-100-C

    echo "[TinyImageNet] Training ${ARCH}..."
    python init_base_model.py --arch ${ARCH} --train_epoch 50 --train_data_name tinyimagenet --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path ${DATA_ROOT}/tinyimagenet --cifar_corruption_path ${DATA_ROOT}/TinyImageNet-C
done

# ===== Step 2: MaNo Evaluation =====
echo "===== Step 2: MaNo Evaluation ====="
for ARCH in resnet18 resnet50 wrn_50_2
do
    echo "[CIFAR-10] Evaluating ${ARCH}..."
    python main.py --alg mano --arch ${ARCH} --severity -1 --dataname cifar10 --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path ${DATA_ROOT}/cifar10 --cifar_corruption_path ${DATA_ROOT}/CIFAR-10-C --norm_type 4

    echo "[CIFAR-100] Evaluating ${ARCH}..."
    python main.py --alg mano --arch ${ARCH} --severity -1 --dataname cifar100 --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path ${DATA_ROOT}/cifar100 --cifar_corruption_path ${DATA_ROOT}/CIFAR-100-C --norm_type 4

    echo "[TinyImageNet] Evaluating ${ARCH}..."
    python main.py --alg mano --arch ${ARCH} --severity -1 --dataname tinyimagenet --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path ${DATA_ROOT}/tinyimagenet --cifar_corruption_path ${DATA_ROOT}/TinyImageNet-C --norm_type 4
done
