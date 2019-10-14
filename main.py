'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import random
import numpy as np


from models import *
from optimizers import *
from utils import *



checkpoint_path = "./checkpoint/"
logs_path = "./logs/"
if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)
if not os.path.exists(logs_path):
    os.mkdir(logs_path)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)

# random seed
random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch_size', default=128, type=int, metavar='N', help='train batchsize')
args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}



# main
def main():
    fold = 'Lookahead0908'   # for experiments

    start_epoch = 0         # start from epoch 0 or last checkpoint epoch
    best_acc = 0            # best test accuracy
    best_loss = np.inf      # best test loss

    # 1 Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 2 Model
    print('==> Building model..')
    # net = VGG('VGG19')
    net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    #net = EfficientNetB0()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # model complexity
    print('    Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))


    log_path = os.path.join(logs_path, fold + os.sep)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        filepath = os.path.join(checkpoint_path, fold + os.sep + 'checkpoint.pth.tar')
        checkpoint = torch.load(filepath)
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['best_acc']
        best_loss = checkpoint['best_loss']
        start_epoch = checkpoint['epoch']
        logger = Logger(os.path.join(log_path, 'log.txt'), title=title, resume=True)
    else:
        # log
        logger = Logger(os.path.join(log_path, 'log.txt'))
        logger.set_names(['Learning Rate', 'Train Loss', 'Test Loss', 'Train Acc.', 'Test Acc.'])

    # 3 Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)        # SGD
    #optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)  #Adam
    #optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)   #Amsgrad
    #optimizer = AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay = 0.1)    # AdamW
    #optimizer = RAdam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0)        # RAdam
    base_opt = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))                     # Any optimizer
    optimizer = Lookahead(base_opt, k=5, alpha=0.5)                                             # Lookahead
    
    


    # 4 train and test
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train(trainloader, net, criterion, optimizer, epoch)
        test_loss, test_acc = test(testloader, net, criterion, epoch)


        # append log file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        best_loss = min(test_loss, best_loss)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'best_loss': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=checkpoint_path, fold=fold)


    logger.close()
    logger.plot()
    savefig(os.path.join(log_path, 'log.eps'))

    print('Best accuracy:')
    print(best_acc)

# Training
def train(trainloader, net, criterion, optimizer, epoch):
    print('\nEpoch: %d' % epoch)

    net.train()
    train_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        # lastest
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        train_loss.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        #top5.update(prec5.item(), inputs.size(0))
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss.avg, top1.avg, correct, total))

    return (train_loss.avg, top1.avg)


def test(testloader, net, criterion, epoch):
    #global best_acc

    net.eval()
    test_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            # latest
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            test_loss.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            #top5.update(prec5.item(), inputs.size(0))
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss.avg, top1.avg, correct, total))

    return (test_loss.avg, top1.avg)
   

if __name__ == "__main__":
    main()
