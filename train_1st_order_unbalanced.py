import sys

import torch

import models
from utils import *
from args_unbalance import parse_train_args
from datasets_unbalance import make_dataset
from tqdm import tqdm
import wandb
import time

CIFAR10_TRAIN_SAMPLES = (1000, 1000, 2000, 2000, 3000, 3000, 4000, 4000, 5000, 5000)
CIFAR100_TRAIN_SAMPLES = tuple([*[100]*10, *[100]*10, *[200]*10, *[200]*10, *[300]*10, 
                                *[300]*10, *[400]*10, *[400]*10, *[500]*10, *[500]*10])
MNIST_TRAIN_SAMPLES = (100, 100, 200, 200, 300, 300, 400, 400, 500, 500)
FashionMNIST_TRAIN_SAMPLES = (100, 100, 200, 200, 300, 300, 400, 400, 500, 500)
SVHN_TRAIN_SAMPLES = (100, 100, 200, 200, 300, 300, 400, 400, 500, 500)


def loss_compute(args, model, criterion, outputs, targets):
    if args.loss == 'CrossEntropy':
        loss = criterion(outputs[0], targets)
    elif args.loss == 'MSE':
        loss = criterion(outputs[0], nn.functional.one_hot(
            targets, num_classes=10).type(torch.FloatTensor).to(args.device))

    # Now decide whether to add weight decay on last weights and last features
    if args.sep_decay:
        # Find features and weights
        features = outputs[1]
        W = []
        B = []
        for fc_layer in model.fc:
            if isinstance(fc_layer, nn.Linear):
                W.append(fc_layer.weight)
                if fc_layer.bias is not None:
                    B.append(fc_layer.bias)
        lamb = args.weight_decay / 2
        lamb_feature = args.feature_decay_rate / 2
        loss = loss/2
        loss += lamb * sum([torch.sum(w ** 2) for w in W])
        loss += lamb_feature * torch.sum(features ** 2)

    return loss


def trainer(args, model, trainloader, epoch_id, criterion, optimizer, scheduler, logfile):

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    print_and_save('\nTraining Epoch: [%d | %d] LR: %f' % (
        epoch_id + 1, args.epochs, scheduler.get_last_lr()[-1]), logfile)
    # start = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # print("Data loading time is:")
        # print(time.time() - start)
        # start = time.time()
        inputs, targets = inputs.to(args.device), targets.to(args.device)

        model.train()
        outputs = model(inputs)

        if args.sep_decay:
            loss = loss_compute(args, model, criterion, outputs, targets)
        else:
            if args.loss == 'CrossEntropy':
                loss = criterion(outputs[0], targets)
            elif args.loss == 'MSE':
                loss = criterion(outputs[0], nn.functional.one_hot(
                    targets).type(torch.FloatTensor).to(args.device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        model.eval()
        outputs = model(inputs)
        prec1, prec5 = compute_accuracy(
            outputs[0].detach().data, targets.detach().data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        # print("Training time is:")
        # print(time.time()-start)
        # start = time.time()
        if batch_idx % 100 == 0:
            print_and_save('[epoch: %d] (%d/%d) | Loss: %.4f | top1: %.4f | top5: %.4f ' %
                           (epoch_id + 1, batch_idx + 1, len(trainloader), losses.avg, top1.avg, top5.avg), logfile)
    wandb.log({"loss": losses.avg})

    scheduler.step()


def train(args, model, trainloader):

    criterion = make_criterion(args)
    optimizer = make_optimizer(args, model)
    scheduler = make_scheduler(args, optimizer)

    logfile = open('%s/train_log.txt' % (args.save_path), 'w')

    print_and_save('# of model parameters: ' +
                   str(count_network_parameters(model)), logfile)
    print_and_save(
        '--------------------- Training -------------------------------', logfile)
    for epoch_id in tqdm(range(args.epochs)):

        trainer(args, model, trainloader, epoch_id,
                criterion, optimizer, scheduler, logfile)
        if epoch_id % 10 == 0 or epoch_id == args.epochs - 1:
            torch.save(model.state_dict(), args.save_path +
                       "/epoch_" + str(epoch_id + 1).zfill(3) + ".pth")

    logfile.close()


def main():
    args = parse_train_args()
    name = "unbalanced_" + args.dataset + "-" + args.model \
           + "-" + args.loss + "-" + args.optimizer \
           + "-width_" + str(args.width) \
           + "-depth_relu_" + str(args.depth_relu) \
           + "-depth_linear_" + str(args.depth_linear) \
           + "-bias_" + str(args.bias) + "-" + "lr_" + str(args.lr) \
           + "-data_augmentation_" + str(args.data_augmentation) \
           + "-" + "epochs_" + str(args.epochs) + "-" + "seed_" + str(args.seed) \

    if args.wandb == "False":
        wandb.init(project=f'Imbalanced_learning', entity='fpt-team', config={}, name=name, mode="disabled")
    else:
        wandb.init(project=f'Imbalanced_learning', entity='fpt-team', config={}, name=name, mode="online")
    wandb.config.update(args)
    
    wandb.save("train_1st_order_unbalanced.py")
    wandb.save("models/*.py")

    if args.optimizer == 'LBFGS':
        sys.exit('Support for training with 1st order methods!')

    device = torch.device("cuda:"+str(args.gpu_id)
                          if torch.cuda.is_available() else "cpu")
    args.device = device
    set_seed(manualSeed=args.seed)

    if args.dataset == "cifar10":
        trainloader, _, num_classes = make_dataset(args.dataset, args.data_dir,
                                                   CIFAR10_TRAIN_SAMPLES, args.batch_size,
                                                   args.sample_size)
    elif args.dataset == "cifar100":
        trainloader, _, num_classes = make_dataset(args.dataset, args.data_dir,
                                                   CIFAR100_TRAIN_SAMPLES, args.batch_size,
                                                   args.sample_size)
    elif args.dataset == "mnist":
        trainloader, _, num_classes = make_dataset(args.dataset, args.data_dir,
                                                   MNIST_TRAIN_SAMPLES, args.batch_size,
                                                   args.sample_size)
    elif args.dataset == "fashionmnist":
        trainloader, _, num_classes = make_dataset(args.dataset, args.data_dir,
                                                   FashionMNIST_TRAIN_SAMPLES, args.batch_size,
                                                   args.sample_size)
    elif args.dataset == "svhn":
        trainloader, _, num_classes = make_dataset(args.dataset, args.data_dir,
                                                   SVHN_TRAIN_SAMPLES, args.batch_size,
                                                   args.sample_size)

    if args.model == "MLP":
        model = models.__dict__[args.model](hidden=args.width, depth_relu=args.depth_relu,
                                            depth_linear=args.depth_linear, fc_bias=args.bias, num_classes=num_classes, batchnorm=False).to(device)

    else:
        model = models.__dict__[args.model](hidden=args.width, depth_linear=args.depth_linear,
                                            num_classes=num_classes, fc_bias=args.bias).to(device)

    train(args, model, trainloader)


if __name__ == "__main__":
    main()
    wandb.finish()