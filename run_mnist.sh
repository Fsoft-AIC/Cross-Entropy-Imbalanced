# Bigger mnist dataset
CUDA_VISIBLE_DEVICES=0 python train_1st_order_unbalanced.py --model MLP \
            --no-bias --dataset mnist --depth_relu 6 --depth_linear 1 \
            --width 1024 --seed 1 --sep_decay --batch_size 256 \
            --feature_decay_rate 0.00001 --loss CrossEntropy --lr 0.0001 \
            --weight_decay 0.0001 --optimizer Adam --epochs 4000 --patience 2000

CUDA_VISIBLE_DEVICES=0 python validate_NC_unbalanced.py --model MLP \
            --no-bias --dataset mnist --depth_relu 6 --depth_linear 1 \
            --width 1024 --seed 1 --sep_decay --batch_size 256 \
            --feature_decay_rate 0.00001 --loss CrossEntropy --lr 0.0001 \
            --weight_decay 0.0001 --optimizer Adam --epochs 4000 --patience 2000

CUDA_VISIBLE_DEVICES=0 python train_1st_order_unbalanced.py --model ResNet18 \
            --no-bias --dataset mnist --depth_relu 6 --depth_linear 1 \
            --width 1024 --seed 1 --sep_decay --batch_size 256 \
            --feature_decay_rate 0.00001 --loss CrossEntropy --lr 0.0001 \
            --weight_decay 0.0001 --optimizer Adam --epochs 4000 --patience 2000

CUDA_VISIBLE_DEVICES=0 python validate_NC_unbalanced.py --model ResNet18 \
            --no-bias --dataset mnist --depth_relu 6 --depth_linear 1 \
            --width 1024 --seed 1 --sep_decay --batch_size 256 \
            --feature_decay_rate 0.00001 --loss CrossEntropy --lr 0.0001 \
            --weight_decay 0.0001 --optimizer Adam --epochs 4000 --patience 2000

CUDA_VISIBLE_DEVICES=0 python train_1st_order_unbalanced.py --model VGG11 \
            --no-bias --dataset mnist --depth_relu 6 --depth_linear 1 \
            --width 1024 --seed 1 --sep_decay --batch_size 256 \
            --feature_decay_rate 0.00001 --loss CrossEntropy --lr 0.0001 \
            --weight_decay 0.0001 --optimizer Adam --epochs 4000 --patience 2000

CUDA_VISIBLE_DEVICES=0 python validate_NC_unbalanced.py --model VGG11 \
            --no-bias --dataset mnist --depth_relu 6 --depth_linear 1 \
            --width 1024 --seed 1 --sep_decay --batch_size 256 \
            --feature_decay_rate 0.00001 --loss CrossEntropy --lr 0.0001 \
            --weight_decay 0.0001 --optimizer Adam --epochs 4000 --patience 2000

CUDA_VISIBLE_DEVICES=0 python train_1st_order_unbalanced.py --model densenet_cifar \
            --no-bias --dataset mnist --depth_relu 6 --depth_linear 1 \
            --width 1024 --seed 1 --sep_decay --batch_size 256 \
            --feature_decay_rate 0.00001 --loss CrossEntropy --lr 0.0001 \
            --weight_decay 0.0001 --optimizer Adam --epochs 4000 --patience 2000

CUDA_VISIBLE_DEVICES=0 python validate_NC_unbalanced.py --model densenet_cifar \
            --no-bias --dataset mnist --depth_relu 6 --depth_linear 1 \
            --width 1024 --seed 1 --sep_decay --batch_size 256 \
            --feature_decay_rate 0.00001 --loss CrossEntropy --lr 0.0001 \
            --weight_decay 0.0001 --optimizer Adam --epochs 4000 --patience 2000