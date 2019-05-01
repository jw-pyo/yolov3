import argparse
import time

import test  # Import test.py to get mAP after each epoch
from models import load_darknet_weights
from multitask_models import *

from utils.datasets import *
from utils.utils import *


def train(
        cfg,
        data_cfg,
        img_size=416,
        resume=False,
        epochs=100,
        batch_size=16,
        accumulated_batches=1,
        multi_scale=False,
        freeze_backbone=True,
        var=0,
        weight_path="weights/rainy",
        result="result.txt",
        ckpt=10
):
    weights = weight_path
    latest = weights + 'latest.pt'
    best = weights + 'best.pt'
    device = torch_utils.select_device()

    if multi_scale:  # pass maximum multi_scale size
        img_size = 608
    else:
        torch.backends.cudnn.benchmark = True  # unsuitable for multiscale

    # Configure run
    train_path = parse_data_cfg(data_cfg)['train']

    # Initialize model
    model = Darknet(cfg, img_size)

    # Get dataloader
    dataloader = LoadImagesAndLabels(train_path, batch_size, img_size, multi_scale=multi_scale, augment=True)

    lr0 = 0.001
    cutoff = 10  # backbone reaches to cutoff layer
    start_epoch = 0
    best_loss = float('inf')
    if resume:
        checkpoint = torch.load(latest, map_location='cpu')

        # Load weights to resume from
        model.load_state_dict(checkpoint['model'])

        # if torch.cuda.device_count() > 1:
        #   model = nn.DataParallel(model)
        model.to(device).train()

        # Transfer learning (train only YOLO layers)
        # for i, (name, p) in enumerate(model.named_parameters()):
        #     p.requires_grad = True if (p.shape[0] == 255) else False

        # Set optimizer
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=lr0, momentum=.9)

        start_epoch = checkpoint['epoch'] + 1
        if checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_loss = checkpoint['best_loss']

        del checkpoint  # current, saved

    else:
        # Initialize model with backbone (optional)
        if cfg.endswith('yolov3.cfg'):
            load_darknet_weights(model, weights + 'darknet53.conv.74')
            cutoff = 75
        elif cfg.endswith('yolov3-tiny.cfg'):
            load_darknet_weights(model, weights + 'yolov3-tiny.conv.15')
            cutoff = 15
        elif cfg.startswith('cfg/bdd100k'):
            #transfer learning
            print("Apply transfer learning for bdd100k cfg")
            tmp_model = Darknet('cfg/yolov3.cfg', img_size)
            load_darknet_weights(tmp_model, "weights/yolov3.weights")
            pretrained_dict = tmp_model.state_dict()

            for k, v in model.state_dict().items():
                if v.shape != pretrained_dict[k].shape:
                    pretrained_dict[k] = torch.empty(v.shape)
                    #TODO: conv, batch
                    if k.split(".")[2].startswith("conv"):
                        nn.init.normal_(pretrained_dict[k], 0.0, 0.03)
                    elif k.split(".")[2].startswith("batch_norm") and k.split(".")[3] == "weight":
                        nn.init.normal_(pretrained_dict[k], 1.0, 0.03)
                    elif k.split(".")[2].startswith("batch_norm") and k.split(".")[3] == "bias":
                        nn.init.constant_(pretrained_dict[k], 0.0)
                    else:
                        nn.init_normal_(pretrained_dict[k], torch.mean(v), torch.std(v))

                    print(k, v.shape)
            model.load_state_dict(pretrained_dict) 
            del tmp_model
        #freeze_layer
        cutoff = 10
        model.freeze_layers(cutoff)
        model.to(device).train()

        # Set optimizer
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=lr0, momentum=.9)

    # Set scheduler
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[54, 61], gamma=0.1)

    t0 = time.time()
    model_info(model)
    n_burnin = min(round(dataloader.nB / 5), 1000)  # number of burn-in batches
    for epoch in range(1, epochs+1):
        epoch += start_epoch

        print(('%8s%12s' + '%10s' * 7) % (
            'Epoch', 'Batch', 'xy', 'wh', 'conf', 'cls', 'total', 'nTargets', 'time'))

        # Update scheduler (automatic)
        # scheduler.step()

        # Update scheduler (manual)  at 0, 54, 61 epochs to 1e-3, 1e-4, 1e-5
        if epoch > 50:
            lr = lr0 / 10
        else:
            lr = lr0
        for g in optimizer.param_groups:
            g['lr'] = lr

        # Freeze darknet53.conv.74 for first epoch
        if freeze_backbone:
            for i, (name, p) in enumerate(model.named_parameters()):
                if int(name.split('.')[1]) < cutoff:  # if layer < 75
                    p.requires_grad = False if (epoch == 0) else True

        ui = -1
        rloss = defaultdict(float)  # running loss
        optimizer.zero_grad()
        for i, (imgs, targets, _, _, var) in enumerate(dataloader):
            if sum([len(x) for x in targets]) < 1:  # if no targets continue
                continue

            # SGD burn-in
            if (epoch == 0) & (i <= n_burnin):
                lr = lr0 * (i / n_burnin) ** 4
                for g in optimizer.param_groups:
                    g['lr'] = lr

            # Compute loss, compute gradient, update parameters
            loss = model(imgs.to(device), targets, var=0)
            loss.backward()

            # accumulate gradient for x batches before optimizing
            if ((i + 1) % accumulated_batches == 0) or (i == len(dataloader) - 1):
                optimizer.step()
                optimizer.zero_grad()

            # Running epoch-means of tracked metrics
            ui += 1
            for key, val in model.losses.items():
                rloss[key] = (rloss[key] * ui + val) / (ui + 1)

            s = ('%8s%12s' + '%10.3g' * 7) % (
                '%g/%g' % (epoch, epochs + start_epoch),
                '%g/%g' % (i, len(dataloader) - 1),
                rloss['xy'], rloss['wh'], rloss['conf'],
                rloss['cls'], rloss['loss'],
                model.losses['nT'], time.time() - t0)
            t0 = time.time()
            print(s)
        # Update best loss
        loss_per_target = rloss['loss'] / rloss['nT']
        if loss_per_target < best_loss:
            best_loss = loss_per_target

        # Save latest checkpoint
        checkpoint = {'epoch': epoch,
                      'best_loss': best_loss,
                      'model': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, latest)

        # Save best checkpoint
        if best_loss == loss_per_target:
            os.system('cp ' + latest + ' ' + best)

        # Save backup weights every 5 epochs (optional)
        if (epoch > 0) & (epoch % ckpt == 0):
            os.system('cp ' + latest + ' ' + weights + 'backup{}.pt'.format(epoch))

        # Calculate mAP
        with torch.no_grad():
            mAP, R, P = test.test(cfg, data_cfg, weights=latest, batch_size=batch_size, img_size=img_size)

        # Write epoch results
        with open(result, 'a') as file:
            file.write(s + '%11.3g' * 3 % (mAP, P, R) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=15, help='size of each image batch')
    parser.add_argument('--accumulated-batches', type=int, default=1, help='number of batches before optimizer step')
    parser.add_argument('--cfg', type=str, default='cfg/bdd100k/bdd100k.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='cfg/bdd100k/bdd100k_rainy.data', help='coco.data file path')
    parser.add_argument('--multi-scale', action='store_true', help='random image sizes per batch 320 - 608')
    parser.add_argument('--img-size', type=int, default=32 * 13, help='pixels')
    parser.add_argument('--resume', type=bool, default=False, help='resume training flag')
    parser.add_argument('--var', type=float, default=0, help='test variable')
    parser.add_argument('--weight_path', type=str, default="weights/rainy/", help="weight path")
    parser.add_argument('--result', type=str, default="result/rainy/rainy.txt", help="result txt file")
    parser.add_argument('--ckpt', type=int, default=10, help="save the weight by this value")
    opt = parser.parse_args()
    print(opt, end='\n\n')

    init_seeds()

    train(
        opt.cfg,
        opt.data_cfg,
        img_size=opt.img_size,
        resume=opt.resume,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
        accumulated_batches=opt.accumulated_batches,
        multi_scale=opt.multi_scale,
        var=opt.var,
        weight_path=opt.weight_path,
        result=opt.result,
        ckpt=opt.ckpt
    )
