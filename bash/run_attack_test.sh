#!/usr/bin/env bash
# Quick test: resnet18 + cifar10 + PGD 4-step
python main_attack.py \
    --arch resnet18 --severity -1 --dataname cifar10 \
    --lr 0.001 --batch_size 128 --seed 1 --gpu 0 \
    --cifar_data_path ./datasets/cifar10 \
    --cifar_corruption_path ./datasets/CIFAR-10-C \
    --norm_type 4 --results_dir results_attack_test \
    --attack_mode pgd --attack_steps 4
