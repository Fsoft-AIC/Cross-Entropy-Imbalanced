# Neural Collapse for Cross-entropy Class-Imbalanced Learning with Unconstrained ReLU Features Model

This is the code for the [paper](https://arxiv.org/abs/2401.02058) "Neural Collapse for Cross-entropy Class-Imbalanced Learning with Unconstrained ReLU Features Model".

International Conference on Machine Learning (ICML), 2024

## Experiments
***MLP experiment***
```
CUDA_VISIBLE_DEVICES=0 python train_1st_order_unbalanced.py --model MLP \
            --no-bias --dataset cifar10 --depth_relu 6 --depth_linear 1 \
            --width 1024 --seed 1 --sep_decay --batch_size 256 \
            --feature_decay_rate 0.00001 --loss CrossEntropy --lr 0.0001 \
            --weight_decay 0.0001 --optimizer Adam --epochs 4000 --patience 2000

CUDA_VISIBLE_DEVICES=0 python validate_NC_unbalanced.py --model MLP \
            --no-bias --dataset cifar10 --depth_relu 6 --depth_linear 1 \
            --width 1024 --seed 1 --sep_decay --batch_size 256 \
            --feature_decay_rate 0.00001 --loss CrossEntropy --lr 0.0001 \
            --weight_decay 0.0001 --optimizer Adam --epochs 4000 --patience 2000
```
***ResNet18 experiment***
```
CUDA_VISIBLE_DEVICES=0 python train_1st_order_unbalanced.py --model ResNet18 \
            --no-bias --dataset cifar10 --depth_relu 6 --depth_linear 1 \
            --width 1024 --seed 1 --sep_decay --batch_size 32 \
            --feature_decay_rate 0.00001 --loss CrossEntropy --lr 0.0001 \
            --weight_decay 0.0001 --optimizer Adam --epochs 4000 --patience 2000

CUDA_VISIBLE_DEVICES=0 python train_1st_order_unbalanced.py --model ResNet18 \
            --no-bias --dataset cifar10 --depth_relu 6 --depth_linear 1 \
            --width 1024 --seed 1 --sep_decay --batch_size 32 \
            --feature_decay_rate 0.00001 --loss CrossEntropy --lr 0.0001 \
            --weight_decay 0.0001 --optimizer Adam --epochs 4000 --patience 2000
```
## Citation and reference 
For technical details and full experimental results, please check [our paper](https://arxiv.org/abs/2401.02058).
```
@article{dang2024neuralcollapsecrossentropyclassimbalanced,
  title={Neural Collapse for Cross-entropy Class-Imbalanced Learning with Unconstrained ReLU Feature Model,
  author={Hien Dang and Tho Tran and Tan Nguyen and Nhat Ho},
  journal={arXiv preprint arXiv:2401.02058},
  year={2024}
}
