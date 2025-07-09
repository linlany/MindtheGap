#!bin/bash



python epoch.py \
    --config-path configs/class \
    --config-name cifar100_10-10.yaml \
    dataset_root="path" \
    class_order="class_orders/cifar100.yaml"

python main.py \
    --config-path configs/class \
    --config-name cifar100_10-10.yaml \
    dataset_root="path" \
    class_order="class_orders/cifar100.yaml"


