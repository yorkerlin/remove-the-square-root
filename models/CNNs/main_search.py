import argparse
import os
import socket
import sys
import time
import wandb
from torch.nn import Conv2d, Linear
from contextlib import suppress
from functools import partial

import math
import torch
import torch.nn as nn
import torch.optim as optim
from scaler_timm import NativeScaler
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from myoptim import MyRmsProp

from torch.optim.lr_scheduler import MultiStepLR, ConstantLR
from tqdm import tqdm
from utils.data_utils import get_dataloader
from utils.network_utils import get_network

def make_criterion(loss):
    if loss == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss().cuda()
    elif loss == 'MSE':
        criterion = nn.MSELoss().cuda()
    return criterion



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# fetch args
parser = argparse.ArgumentParser()


parser.add_argument("--network", default="vgg16_bn", type=str)
parser.add_argument("--dataset", default="cifar10", type=str)
parser.add_argument('--loss', default='CrossEntropy', type=str,
                    help='loss type')

parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--resume", "-r", action="store_true")
parser.add_argument("--load_path", default="", type=str)
parser.add_argument("--log_dir", default="runs/pretrain", type=str)

parser.add_argument("--beta2", default=0.5, type=float)
parser.add_argument("--optimizer", default="adamw", type=str)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--epoch", default=100, type=int)
parser.add_argument("--milestone", default=None, type=str)
parser.add_argument("--learning_rate", default=0.01, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--damping", default=5e-3, type=float)
parser.add_argument("--weight_decay", default=3e-3, type=float)
parser.add_argument("--run_id", default=1, type=int)
parser.add_argument("--amp", default=False, type=bool)
parser.add_argument("--lr_cov", default=1e-2, type=float)
parser.add_argument("--prefix", default=None, type=str)

parser.add_argument(
    "--using_constant_lr",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--using_dummy_init",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--using_dummy_scaling",
    action="store_true",
    default=False,
)
args = parser.parse_args()

# init model
nc = {
    "tinyimagenet": 200,
    "imagenet100": 100,
    "cifar10": 10,
    "cifar100": 100,
    "imagewoof": 10,
}


num_classes = nc[args.dataset]
net = get_network(args.network, num_classes=num_classes)
net = net.to(args.device)


# init dataloader
trainloader, testloader = get_dataloader(
    dataset=args.dataset, train_batch_size=args.batch_size, test_batch_size=256
)

# init optimizer and lr scheduler
optim_name = args.optimizer.lower()
tag = optim_name
data_name = args.dataset
model_name = args.network
print(optim_name)
print(count_parameters(net))
print(args.loss)


if optim_name.startswith("sgd"):
    args.amp = False
    if optim_name.find("^") > 0:
        if optim_name.split("^")[1] == "amp":
            print("enable amp")
            args.amp = True
    optimizer = optim.SGD(
        net.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
elif optim_name.startswith("adamw"):
    args.amp = False
    if optim_name.find("^") > 0:
        if optim_name.split("^")[1] == "amp":
            print("enable amp")
            args.amp = True
    optimizer = optim.AdamW(
        net.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=args.damping,
        betas=(args.momentum, 1.0 - args.lr_cov),
    )

elif optim_name.startswith("rfrmsprop"):
    args.amp = False
    if optim_name.find("^") > 0:
        if optim_name.split("^")[1] == "amp":
            print("enable amp")
            args.amp = True

    amp_dtype = torch.float32
    if args.amp:
        amp_dtype = torch.bfloat16

    optimizer = MyRmsProp(
        net.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        eps=args.damping,
        weight_decay=args.weight_decay,
        alpha= 1.0 - args.lr_cov,
        batch_averaged=True,
        cast_dtype=amp_dtype,
        batch_size=args.batch_size,
        dummy_init = args.using_dummy_init,
        dummy_scaling = args.using_dummy_scaling,
    )

else:
    raise NotImplementedError

loss_scaler = None
if args.amp:
    amp_dtype = torch.bfloat16
    amp_autocast = partial(torch.autocast, device_type=args.device, dtype=amp_dtype)
    print("amp", amp_dtype)
    loss_scaler = NativeScaler()
else:
    print("no amp")
    amp_autocast = suppress  # do nothing


if args.using_constant_lr:
    print('using constant lr')
    lr_scheduler = ConstantLR(optimizer, factor=1.0, total_iters=0)
else:
    if args.milestone is None:
        lr_scheduler = MultiStepLR(
            optimizer,
            milestones=[int(args.epoch * 0.5), int(args.epoch * 0.75)],
            gamma=0.1,
        )
    else:
        milestone = [int(_) for _ in args.milestone.split(",")]
        lr_scheduler = MultiStepLR(
            optimizer,
            milestones=milestone, gamma=0.1
        )


# init criterion
criterion = make_criterion(args.loss)

start_epoch = 0
if args.resume:
    print("==> Resuming from checkpoint..")
    assert os.path.isfile(args.load_path), "Error: no checkpoint directory found!"
    checkpoint = torch.load(args.load_path)
    net.load_state_dict(checkpoint["net"])
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]
    print("==> Loaded checkpoint at epoch: %d, acc: %.2f%%" % (start_epoch, best_acc))

log_dir = os.path.join(
    args.log_dir,
    args.dataset,
    args.network,
    args.optimizer,
    "lr%.3f_wd%.4f_damping%.4f" % (args.learning_rate, args.weight_decay, args.damping),
)
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)


def train(epoch):
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    desc = "[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)" % (
        tag,
        lr_scheduler.get_last_lr()[0],
        0,
        0,
        correct,
        total,
    )

    lr_scheduler.step()

    prog_bar = tqdm(
        enumerate(trainloader), total=len(trainloader), desc=desc, leave=True
    )

    batch_time = 0.0
    M = 30
    k = math.sqrt(15)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.to(args.device), targets.to(args.device)

        if args.loss == 'MSE':
               mask = nn.functional.one_hot(targets, num_classes=num_classes).type(torch.FloatTensor).to('cuda')
               rms_targets = mask * (M*k)
               scaling_mask = mask * (k-1) + 1

        end = time.time()
        optimizer.zero_grad()

        with amp_autocast():
            outputs = net(inputs)
            if args.loss == 'CrossEntropy':
               loss = criterion(outputs, targets)
            elif args.loss == 'MSE':
               loss = criterion(outputs * scaling_mask, rms_targets)

        train_loss += loss.item()

        if loss_scaler is not None:
            loss_scaler(
                loss,
                optimizer,
            )
        else:
            optimizer.acc_stats = True
            loss.backward()
            optimizer.acc_stats = False
            optimizer.step()

        torch.cuda.current_stream().synchronize()
        batch_time += time.time() - end

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if optim_name.startswith("sngd"):
            desc = "[%s][%s][LR=%s][%s][%f] Loss: %.3f | Acc: %.3f%% (%d/%d)" % (
                data_name,
                tag,
                lr_scheduler.get_last_lr()[0],
                model_name,
                batch_time,
                train_loss / (batch_idx + 1),
                100.0 * correct / total,
                correct,
                total,
            )

        else:
            desc = "[%s][%s][LR=%s][%s][%f] Loss: %.3f | Acc: %.3f%% (%d/%d)" % (
                data_name,
                tag,
                lr_scheduler.get_last_lr()[0],
                model_name,
                batch_time,
                train_loss / (batch_idx + 1),
                100.0 * correct / total,
                correct,
                total,
            )

        prog_bar.set_description(desc, refresh=True)

    curr_lr = lr_scheduler.get_last_lr()[0]
    train_loss = train_loss / len(trainloader)
    train_acc = 100.0 * correct / total

    return batch_time, curr_lr, train_loss, train_acc


def test(epoch, info):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    desc = "[%s]Loss: %.3f | Acc: %.3f%% (%d/%d)" % (
        tag,
        test_loss / (0 + 1),
        0,
        correct,
        total,
    )

    M = 30
    k = math.sqrt(15)
    prog_bar = tqdm(enumerate(testloader), total=len(testloader), desc=desc, leave=True)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in prog_bar:
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            if args.loss == 'MSE':
                   # we use a loss rescaling for MSE losses
                   # https://arxiv.org/abs/2006.07322
                   mask = nn.functional.one_hot(targets, num_classes=num_classes).type(torch.FloatTensor).to('cuda')
                   rms_targets = mask * (M*k)
                   scaling_mask = mask * (k-1) + 1

            with amp_autocast():
                outputs = net(inputs)
                if args.loss == 'CrossEntropy':
                   loss = criterion(outputs, targets)
                elif args.loss == 'MSE':
                   loss = criterion(outputs * scaling_mask, rms_targets)


            test_loss += loss.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            desc = "[%s]Loss: %.3f | Acc: %.3f%% (%d/%d)" % (
                tag,
                test_loss / (batch_idx + 1),
                100.0 * correct / total,
                correct,
                total,
            )
            prog_bar.set_description(desc, refresh=True)

    # Save checkpoint.
    test_loss = test_loss / len(testloader)
    test_acc = 100.0 * correct / total

    info.setdefault(epoch, test_acc)

    return test_loss, test_acc


def main():
    opt_run_name = optim_name

    run_name = "%s-%s-%s" % (opt_run_name, args.network, socket.gethostname())

    # if args.loss == 'CrossEntropy':
        # wandb.init(project="%s-better-exp" % (args.network), name=run_name, config=args)
    # else:
        # wandb.init(project="%s-%s-better-exp" % (args.network,args.loss), name=run_name, config=args)

    print(optim_name, args.learning_rate, args.beta2, args.momentum)
    info = {}
    time_info = {}
    cur_time = 0.0
    for epoch in range(start_epoch, args.epoch):
        batch_time, curr_lr, train_loss, train_acc = train(epoch)
        cur_time += batch_time
        time_info.setdefault(epoch, cur_time)
        test_loss, test_acc = test(epoch, info)

        log_data = {
            "curr_lr": curr_lr,
            "train_loss": train_loss,
            "train_acc1": train_acc,
            "acc1": test_acc,
            "test_loss": test_loss,
            "batch_time": batch_time,
        }
        # wandb.log(log_data)

if __name__ == "__main__":
    main()

