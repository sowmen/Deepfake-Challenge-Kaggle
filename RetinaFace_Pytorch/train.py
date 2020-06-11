import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import device, num_classes, train_label_file, valid_label_file, rgb_mean, print_freq, num_workers, grad_clip
from retinaface.data import WiderFaceDetection, detection_collate, preproc, cfg_mnet
from retinaface.layers.functions.prior_box import PriorBox
from retinaface.layers.modules import MultiBoxLoss
from retinaface.models.retinaface import RetinaFace
from utils import parse_args, save_checkpoint, AverageMeter, get_logger, clip_gradient

warnings.simplefilter(action='ignore', category=FutureWarning)


def train_net(args):
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_loss = float('inf')
    writer = SummaryWriter()
    epochs_since_improvement = 0

    cfg = cfg_mnet
    img_dim = cfg['image_size']
    batch_size = cfg['batch_size']

    # Initialize / load checkpoint
    if checkpoint is None:
        net = RetinaFace(cfg=cfg)
        print("Printing net...")
        print(net)
        # net = nn.DataParallel(net)

        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        net = checkpoint['net']
        optimizer = checkpoint['optimizer']

    logger = get_logger()

    # Move to GPU, if available
    net = net.to(device)

    cudnn.benchmark = True

    # Loss function
    criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

    priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.cuda()

    # Custom dataloaders
    train_dataset = WiderFaceDetection(train_label_file, preproc(img_dim, rgb_mean))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers,
                                               collate_fn=detection_collate)
    valid_dataset = WiderFaceDetection(valid_label_file, preproc(img_dim, rgb_mean))
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                               collate_fn=detection_collate)

    scheduler = MultiStepLR(optimizer, milestones=[190, 220], gamma=0.1)

    # Epochs
    for epoch in range(start_epoch, args.end_epoch):
        # One epoch's training
        train_loss = train(train_loader=train_loader,
                           net=net,
                           criterion=criterion,
                           optimizer=optimizer,
                           cfg=cfg,
                           priors=priors,
                           epoch=epoch,
                           logger=logger)

        writer.add_scalar('model/train_loss', train_loss, epoch)

        lr = optimizer.param_groups[0]['lr']
        print('\nLearning rate: {}'.format(lr))
        writer.add_scalar('model/learning_rate', lr, epoch)

        # One epoch's validation
        # val_loss = valid(valid_loader=valid_loader,
        #                  net=net,
        #                  criterion=criterion,
        #                  cfg=cfg,
        #                  priors=priors,
        #                  logger=logger)
        # writer.add_scalar('model/valid_loss', val_loss, epoch)

        scheduler.step(epoch)

        # Check if there was an improvement
        # is_best = val_loss < best_loss
        # best_acc = min(val_loss, best_loss)
        # if not is_best:
        #     epochs_since_improvement += 1
        #     print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        # else:
        #     epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, net, optimizer, train_loss, True)


def train(train_loader, net, criterion, optimizer, cfg, priors, epoch, logger):
    net.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()

    # Batches
    for i, (images, targets) in enumerate(train_loader):
        # Move to GPU, if available
        images = images.to(device)
        targets = [anno.cuda() for anno in targets]

        # forward
        out = net(images)

        # Back prop.
        optimizer.zero_grad()
        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        if loss_l.item() == float('inf'):
            continue
        # print('loc_weight={}, loss_l={}, loss_c={}, loss_landm={}'.format(cfg['loc_weight'], loss_l.item(), loss_c.item(), loss_landm.item()))
        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
        loss.backward()
        optimizer.step()

        # Clip gradients
        clip_gradient(optimizer, grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())

        # Print status
        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader), loss=losses))

    return losses.avg


def valid(valid_loader, net, criterion, cfg, priors, logger):
    net.eval()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()

    # Batches
    for (images, targets) in tqdm(valid_loader):
        # Move to GPU, if available
        images = images.to(device)
        targets = [anno.cuda() for anno in targets]

        # forward
        with torch.no_grad():
            out = net(images)

        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm

        # Keep track of metrics
        losses.update(loss.item())

        # Print status
        status = 'Validation\t Loss {loss.avg:.5f}\n'.format(loss=losses)
        logger.info(status)

    return losses.avg


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
