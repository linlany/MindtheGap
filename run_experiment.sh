#!bin/bash




python main.py \
    --config-path configs/class \
    --config-name imagenet_r_20-20.yaml \
    dataset_root="path" \
    class_order="class_orders/imagenet_R.yaml"



python epoch.py \
    --config-path configs/class \
    --config-name imagenet_r_20-20.yaml \
    dataset_root="path" \
    class_order="class_orders/imagenet_R.yaml"


