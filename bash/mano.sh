#!/usr/bin/env bash

gpu=$1
DATA_ROOT=./datasets

# wandb / results config (leave WANDB_PROJECT empty to disable wandb)
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_GROUP="${WANDB_GROUP:-}"
RESULTS_DIR="${RESULTS_DIR:-results}"

WANDB_ARGS=""
if [ -n "${WANDB_PROJECT}" ]; then
    WANDB_ARGS="--wandb_project ${WANDB_PROJECT}"
    if [ -n "${WANDB_GROUP}" ]; then
        WANDB_ARGS="${WANDB_ARGS} --wandb_group ${WANDB_GROUP}"
    fi
fi

for ARCH in resnet18 resnet50 wrn_50_2
do
    python main.py --alg mano --arch ${ARCH} --severity -1 --dataname cifar10 --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path ${DATA_ROOT}/cifar10 --cifar_corruption_path ${DATA_ROOT}/CIFAR-10-C --norm_type 4 --results_dir ${RESULTS_DIR} --wandb_tags cifar10,${ARCH} ${WANDB_ARGS}
    #python main.py --alg mano --arch ${ARCH} --severity -1 --dataname cifar100 --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path ${DATA_ROOT}/cifar100 --cifar_corruption_path ${DATA_ROOT}/CIFAR-100-C --norm_type 4 --results_dir ${RESULTS_DIR} --wandb_tags cifar100,${ARCH} ${WANDB_ARGS}
    #python main.py --alg mano --arch ${ARCH} --severity -1 --dataname tinyimagenet --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path ${DATA_ROOT}/tinyimagenet --cifar_corruption_path ${DATA_ROOT}/TinyImageNet-C --norm_type 4 --results_dir ${RESULTS_DIR} --wandb_tags tinyimagenet,${ARCH} ${WANDB_ARGS}
    # python main.py --alg mano --arch ${ARCH} --severity -1 --dataname imagenet --lr 0.001 --num_classes 1000 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path path_of_train_data --cifar_corruption_path path_of_corrupted_data --norm_type 4 --results_dir ${RESULTS_DIR} --wandb_tags imagenet,${ARCH} ${WANDB_ARGS}
    # python main.py --alg mano --arch ${ARCH} --dataname pacs --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path path_of_train_data --cifar_corruption_path path_of_corrupted_data --norm_type 4 --results_dir ${RESULTS_DIR} --wandb_tags pacs,${ARCH} ${WANDB_ARGS}
    # python main.py --alg mano --arch ${ARCH} --dataname office_home --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path path_of_train_data --cifar_corruption_path path_of_corrupted_data --norm_type 4 --results_dir ${RESULTS_DIR} --wandb_tags office_home,${ARCH} ${WANDB_ARGS}
    # python main.py --alg mano --arch ${ARCH} --severity -1 --dataname domainnet --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path path_of_train_data --cifar_corruption_path path_of_corrupted_data --norm_type 4 --results_dir ${RESULTS_DIR} --wandb_tags domainnet,${ARCH} ${WANDB_ARGS}
    # python main.py --alg mano --arch ${ARCH} --severity -1 --dataname wilds_rr1 --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path path_of_train_data --cifar_corruption_path path_of_corrupted_data --norm_type 4 --results_dir ${RESULTS_DIR} --wandb_tags wilds_rr1,${ARCH} ${WANDB_ARGS}
    # python main.py --alg mano --arch ${ARCH} --severity -1 --dataname entity13 --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path path_of_train_data --cifar_corruption_path path_of_corrupted_data --norm_type 4 --results_dir ${RESULTS_DIR} --wandb_tags entity13,${ARCH} ${WANDB_ARGS}
    # python main.py --alg mano --arch ${ARCH} --severity -1 --dataname entity30 --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path path_of_train_data --cifar_corruption_path path_of_corrupted_data --norm_type 4 --results_dir ${RESULTS_DIR} --wandb_tags entity30,${ARCH} ${WANDB_ARGS}
    # python main.py --alg mano --arch ${ARCH} --severity -1 --dataname living17 --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path path_of_train_data --cifar_corruption_path path_of_corrupted_data --norm_type 4 --results_dir ${RESULTS_DIR} --wandb_tags living17,${ARCH} ${WANDB_ARGS}
    # python main.py --alg mano --arch ${ARCH} --severity -1 --dataname nonliving26 --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path path_of_train_data --cifar_corruption_path path_of_corrupted_data --norm_type 4 --results_dir ${RESULTS_DIR} --wandb_tags nonliving26,${ARCH} ${WANDB_ARGS}
done
