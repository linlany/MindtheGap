#!bin/bash


python epoch.py \
    --config-path configs/class \
    --config-name imagenet100_10-10.yaml \
    dataset_root="path" \
    class_order="class_orders/imagenet100.yaml"

python main.py \
    --config-path configs/class \
    --config-name imagenet100_10-10.yaml \
    dataset_root="path" \
    class_order="class_orders/imagenet100.yaml"