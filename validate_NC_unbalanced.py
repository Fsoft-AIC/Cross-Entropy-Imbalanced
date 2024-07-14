import sys
import pickle

import torch
import scipy.linalg as scilin

import models
from utils import *
from args_unbalance import parse_eval_args
from datasets_unbalance import make_dataset
import time
import math
import numpy as np
import os
from sympy import Symbol, solve, S
import math
import wandb


CIFAR10_TRAIN_SAMPLES = (1000, 1000, 2000, 2000, 3000, 3000, 4000, 4000, 5000, 5000)
CIFAR100_TRAIN_SAMPLES = tuple([*[100]*10, *[100]*10, *[200]*10, *[200]*10, *[300]*10, 
                                *[300]*10, *[400]*10, *[400]*10, *[500]*10, *[500]*10])
MNIST_TRAIN_SAMPLES = (100, 100, 200, 200, 300, 300, 400, 400, 500, 500)
FashionMNIST_TRAIN_SAMPLES = (100, 100, 200, 200, 300, 300, 400, 400, 500, 500)
SVHN_TRAIN_SAMPLES = (100, 100, 200, 200, 300, 300, 400, 400, 500, 500)


def compute_M_array(NUM_CLASS_SAMPLES, lambda_H, lambda_W, device):
    print("test")
    K = len(NUM_CLASS_SAMPLES)
    N = sum(NUM_CLASS_SAMPLES)
    M_array = torch.zeros(K, device=device)
    for class_indx in range(K):
        n = NUM_CLASS_SAMPLES[class_indx]
        inside_log = (K-1) * (n**(1/2) / 
                              (N * ((lambda_H/2 * lambda_W/2 ) ** (1/2)) * ((K-1)/K * lambda_W/lambda_H)**(1/2)) - 1)
        if inside_log > 0:
            m = math.log(inside_log)
            if m > 0:
                M_array[class_indx] = m
            else:
                M_array[class_indx] = 0
        else:
            M_array[class_indx] = 0
    return M_array

def compute_NC2_H(mu_c_dict, lambda_H, lambda_W, M_array, NUM_CLASS_SAMPLES, load_path):
    K = len(mu_c_dict)
    H = torch.empty(mu_c_dict[0].shape[0], K, device = M_array.device)
    diag_array = torch.zeros(K, device=M_array.device)
    for i in range(K):
        H[:, i] = mu_c_dict[i]
        diag_array[i] = (((K-1)/K * lambda_W/lambda_H * 1/NUM_CLASS_SAMPLES[i])**(1/2) * M_array[i])
    ReLU_H = torch.relu(H)    
    HTH = torch.mm(H.T, H)
    ReLU_HTH = torch.mm(ReLU_H.T, ReLU_H)
    HTH_np = HTH.detach().cpu().numpy()
    ReLU_HTH_np = ReLU_HTH.detach().cpu().numpy()
    np.savetxt(os.path.join(load_path, 'HTH.csv'), HTH_np, delimiter=',')
    np.savetxt(os.path.join(load_path, 'ReLU_HTH.csv'), ReLU_HTH_np, delimiter=',')
    ReLU_HTH /= torch.norm(ReLU_HTH, p='fro')
    sub = torch.diag_embed(diag_array)
    sub /= torch.norm(sub, p='fro')
    metric = torch.norm(ReLU_HTH - sub, p='fro')
    return metric.detach().cpu().numpy().item()

def compute_NC2_W(W, mu_c_dict, NUM_CLASS_SAMPLES):
    K = len(mu_c_dict)
    # H_pre_relu = torch.empty(mu_c_pre_relu_dict[0].shape[0], K, device = W.device)
    H = torch.empty(mu_c_dict[0].shape[0], K, device = W.device)
    sub = torch.zeros([K, mu_c_dict[0].shape[0]], device = W.device)
    for i in range(K):
        H[:, i] = mu_c_dict[i]
    NUM_CLASS_SAMPLES_tensor_sqrt = torch.Tensor(NUM_CLASS_SAMPLES).to("cuda") ** (1/2)
    sum_features = torch.einsum('c, dc -> d', NUM_CLASS_SAMPLES_tensor_sqrt, H)
    for i in range(K):
        sub[i, :] = K * NUM_CLASS_SAMPLES_tensor_sqrt[i] * H[:, i] - sum_features
    sub /= torch.norm(sub, p='fro')
    W /= torch.norm(W, p="fro")
    res = torch.norm(W - sub, p='fro')
    return res.detach().cpu().numpy().item()

def compute_NC2_WWT(W, M_array, NUM_CLASS_SAMPLES):
    WTW = torch.mm(W, W.T)
    K = WTW.shape[0]
    sub = torch.zeros(WTW.shape, device = W.device)
    # Fill in diagonal
    for i in range(K):
        sub[i, i] = (K-1) ** 2 * (NUM_CLASS_SAMPLES[i]) ** (1/2) * M_array[i]
        for m in range(K):
            if m != i:
                sub[i, i] += NUM_CLASS_SAMPLES[m] ** (1/2) * M_array[m]
    # Fill in off-diagonal
    for i in range(K):
        for j in range(K):
            if j != i:
                sub[i, j] = - (K-1) * (NUM_CLASS_SAMPLES[i]) ** (1/2) * M_array[i] \
                            - (K-1) * (NUM_CLASS_SAMPLES[j]) ** (1/2) * M_array[j]
                for m in range(K):
                    if m != i and m != j:
                        sub[i, j] += NUM_CLASS_SAMPLES[m] ** (1/2) * M_array[m]
    # Return metric
    sub /= torch.norm(sub, p='fro')
    WTW /= torch.norm(WTW, p="fro")
    res = torch.norm(WTW - sub, p='fro')
    return res.detach().cpu().numpy().item()

def compute_NC3_WH(W, mu_c_dict, M_array, load_path):
    K = len(mu_c_dict)
    # H_pre_relu = torch.empty(mu_c_pre_relu_dict[0].shape[0], K, device = W.device)
    H = torch.empty(mu_c_dict[0].shape[0], K, device = W.device)
    sub = torch.zeros([K, K], device = M_array.device)
    for i in range(K):
        H[:, i] = mu_c_dict[i]
        sub[:, i] = -1/K * M_array[i]
        sub[i, i] = (K-1)/K * M_array[i]
    # H_pre_relu_np = H_pre_relu.detach().cpu().numpy()
    # np.savetxt(os.path.join(load_path, 'H.csv'), H_pre_relu_np, delimiter=',')    
    #Note that H is after ReLU
    W_np = W.detach().cpu().numpy()
    H_np = H.detach().cpu().numpy()
    np.savetxt(os.path.join(load_path, 'W.csv'), W_np, delimiter=',')
    np.savetxt(os.path.join(load_path, 'ReLU_H.csv'), H_np, delimiter=',')
    WTW = torch.mm(W, W.T)
    WTW_np = WTW.detach().cpu().numpy()
    np.savetxt(os.path.join(load_path, 'WTW.csv'), WTW_np, delimiter=',')
    WH = torch.mm(W, H.cuda())
    WH_np = WH.detach().cpu().numpy()
    np.savetxt(os.path.join(load_path, 'WH.csv'), WH_np, delimiter=',')
    WH /= torch.norm(WH, p='fro')
    sub /= torch.norm(sub, p='fro')
    res = torch.norm(WH - sub, p='fro')
    return res.detach().cpu().numpy().item()


class FCFeatures:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in):
        self.outputs.append(module_in)

    def clear(self):
        self.outputs = []


def compute_info(args, model, fc_features, dataloader, isTrain, NUM_CLASS_SAMPLES):
    if isTrain is True:
        mu_G_list = [0] * (len(model.fc))
        pairs = [(i, 0) for i in range(len(NUM_CLASS_SAMPLES))]
        mu_c_dict_list = [dict(pairs) for i in range(len(model.fc))]
        mu_c_dict_list_count = [dict(pairs) for i in range(len(model.fc))]

    top1 = AverageMeter()
    top5 = AverageMeter()
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):

        inputs, targets = inputs.to(args.device), targets.to(args.device)

        with torch.no_grad():
            outputs = model(inputs)
        
        prec1, prec5 = compute_accuracy(outputs[0].data, targets.data, topk=(1, 5))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        if isTrain is True:
            features_list = []
            for i in range(len(model.fc)):
                features_list.append(fc_features[i].outputs[0][0])
                fc_features[i].clear()
                mu_G_list[i] += torch.sum(features_list[i], dim=0)
            for i in range(len(model.fc)):
                for y in range(len(NUM_CLASS_SAMPLES)):
                    indexes = (targets == y).nonzero(as_tuple=True)[0]
                    if indexes.nelement()==0:
                        mu_c_dict_list[i][y] += 0
                    else:
                        mu_c_dict_list[i][y] += features_list[i][indexes, :].sum(dim=0)
                        mu_c_dict_list_count[i][y] += indexes.shape[0]
    if isTrain:
        for i in range(len(model.fc)):
            mu_G_list[i] /= sum(NUM_CLASS_SAMPLES)
            for j in range(len(NUM_CLASS_SAMPLES)):
                mu_c_dict_list[i][j] /= NUM_CLASS_SAMPLES[j]
    else:
        pass
    
    if isTrain:
        return mu_G_list, mu_c_dict_list, top1.avg, top5.avg
    else:
        return top1.avg, top5.avg


def compute_Sigma_W(args, model, fc_features, mu_c_dict, dataloader, NUM_CLASS_SAMPLES, isTrain):

    Sigma_W = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):

        inputs, targets = inputs.to(args.device), targets.to(args.device)

        with torch.no_grad():
            outputs = model(inputs)

        features = fc_features.outputs[0][0]
        fc_features.clear()
        for y in range(len(mu_c_dict)):
            indexes = (targets == y).nonzero(as_tuple=True)[0]
            if indexes.nelement()==0:
                pass
            else:
                Sigma_W += ((features[indexes, :] - mu_c_dict[y]).unsqueeze(2) @ (features[indexes, :] - mu_c_dict[y]).unsqueeze(1)).sum(0)
    
    if isTrain:
        Sigma_W /= sum(NUM_CLASS_SAMPLES)
    else:
        Sigma_W /= sum(NUM_CLASS_SAMPLES)

    return Sigma_W.cpu().numpy()


def compute_Sigma_B(mu_c_dict, mu_G):
    Sigma_B = 0
    K = len(mu_c_dict)
    for i in range(K):
        Sigma_B += (mu_c_dict[i] - mu_G).unsqueeze(1) @ (mu_c_dict[i] - mu_G).unsqueeze(0)

    Sigma_B /= K

    return Sigma_B.cpu().numpy()


def main():
    args = parse_eval_args()
    name = "unbalanced_" + args.dataset + "-" + args.model \
           + "-" + args.loss + "-" + args.optimizer \
           + "-width_" + str(args.width) \
           + "-depth_relu_" + str(args.depth_relu) \
           + "-depth_linear_" + str(args.depth_linear) \
           + "-bias_" + str(args.bias) + "-" + "lr_" + str(args.lr) \
           + "-sep_decay_" + str(args.sep_decay) \
           + "-weight_decay_" + str(args.weight_decay) \
           + "-" + "epochs_" + str(args.epochs) + "-" + "seed_" + str(args.seed)
    load_path = os.path.join("model_weights", name)
    name = "eval-" + name
    
    if args.wandb == "False":
        wandb.init(project=f'Imbalanced_learning', entity='fpt-team', config={}, name=name, mode="disabled")
    else:
        wandb.init(project=f'Imbalanced_learning', entity='fpt-team', config={}, name=name, mode="online")
    wandb.config.update(args)
    
    wandb.save("validate_NC_unbalanced.py")
    wandb.save("models/*.py")

    device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    args.device = device
    args.batch_size = 64
    set_seed(manualSeed = args.seed)

    if args.dataset == "cifar10":
        NUM_CLASS_TRAIN_SAMPLES = CIFAR10_TRAIN_SAMPLES
    elif args.dataset == "cifar100":
        NUM_CLASS_TRAIN_SAMPLES = CIFAR100_TRAIN_SAMPLES
    elif args.dataset == "mnist":
        NUM_CLASS_TRAIN_SAMPLES = MNIST_TRAIN_SAMPLES
    elif args.dataset == "fashionmnist":
        NUM_CLASS_TRAIN_SAMPLES = FashionMNIST_TRAIN_SAMPLES
    elif args.dataset == "svhn":
        NUM_CLASS_TRAIN_SAMPLES = SVHN_TRAIN_SAMPLES

    trainloader, testloader, num_classes = make_dataset(args.dataset, args.data_dir,
                                                   NUM_CLASS_TRAIN_SAMPLES, args.batch_size,
                                                   args.sample_size)
    if args.model == "MLP":
        model = models.__dict__[args.model](hidden = args.width, depth_relu = args.depth_relu, depth_linear = args.depth_linear, fc_bias=args.bias, num_classes=num_classes, batchnorm=False).to(device)
    else:
        model = models.__dict__[args.model](hidden=args.width, depth_linear=args.depth_linear,
                                            num_classes=num_classes, fc_bias=args.bias).to(device)

    fc_features = [FCFeatures() for i in range(len(model.fc))]

    for i in reversed(range(len(model.fc))):
        model.fc[i].register_forward_pre_hook(fc_features[len(model.fc) - 1 - i])


    M_array = compute_M_array(NUM_CLASS_SAMPLES=NUM_CLASS_TRAIN_SAMPLES, lambda_H=args.feature_decay_rate,
                              lambda_W=args.weight_decay, device=args.device)

    for epoch in range(args.epochs):
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            pass
        else:
            continue
        not_available = True
        while not_available is True:
            not_available = False
            try:
                model.load_state_dict(torch.load(load_path + "/" + 'epoch_' + str(epoch + 1).zfill(3) + '.pth'))
                not_available = False
            except Exception as e:
                not_available = True
                print("Waiting for load when model available")
                print(load_path + "/" + 'epoch_' + str(epoch + 1).zfill(3) + '.pth')
                time.sleep(60)

        model.eval()
        start = time.time()
        W_list = []
        W_temp = model.fc[-1].weight.clone()
        W_list.append(W_temp)
        for j in list(reversed(range(0, len(model.fc)-1))):
            if isinstance(model.fc[j], nn.Linear):
                W_temp = W_temp @ model.fc[j].weight
                W_list.append(W_temp)
        if args.bias is True:
            b = model.fc[-1].bias

        mu_G_train_dict, mu_c_dict_train_dict, \
            train_acc1, train_acc5 = compute_info(args, model, fc_features, trainloader, 
                                                  isTrain=True, NUM_CLASS_SAMPLES=NUM_CLASS_TRAIN_SAMPLES)
        wandb.log({"train_acc1": train_acc1, "train_acc5": train_acc5})
        test_acc1, test_acc5 = compute_info(args, model, fc_features, testloader, 
                                            isTrain=False, NUM_CLASS_SAMPLES=None)
        wandb.log({"test_acc1": test_acc1, "test_acc5": test_acc5})

        Sigma_W = compute_Sigma_W(args, model, fc_features[-1], mu_c_dict_train_dict[-1], trainloader, 
                                  isTrain=True, NUM_CLASS_SAMPLES=NUM_CLASS_TRAIN_SAMPLES)
        Sigma_B = compute_Sigma_B(mu_c_dict_train_dict[-1], mu_G_train_dict[-1])

        NC1_collapse_metric = np.trace(Sigma_W @ scilin.pinv(Sigma_B)) / len(mu_c_dict_train_dict[-1])

        NC2_W = compute_NC2_W(W_list[-1].clone(), mu_c_dict_train_dict[-1], NUM_CLASS_TRAIN_SAMPLES)
        NC2_WWT = compute_NC2_WWT(W_list[-1].clone(), M_array, NUM_CLASS_TRAIN_SAMPLES)
        NC2_H = compute_NC2_H(mu_c_dict_train_dict[-1], args.feature_decay_rate, args.weight_decay, M_array, NUM_CLASS_TRAIN_SAMPLES, load_path )
        NC3_WH = compute_NC3_WH(W_list[-1].clone(), mu_c_dict_train_dict[-1], M_array, load_path)
        wandb.log({"NC1": NC1_collapse_metric, "NC2_W": NC2_W, "NC2_WWT": NC2_WWT, "NC2_H": NC2_H, "NC3_WH": NC3_WH})
        print(time.time()-start)


if __name__ == "__main__":
    main()
    wandb.finish()