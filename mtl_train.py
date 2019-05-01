import argparse
import time
import ast
import mtl_test as test # Import test.py to get mAP after each epoch
#from models import *
from multitask_models import *

from utils.datasets import *
from utils.utils import *


def train(
        cfg,
        shared_cfg,
        diff_cfg,
        data_cfg,
        img_size=416,
        resume=False,
        epochs=100,
        batch_size=16,
        accumulated_batches=1,
        multi_scale=False,
        freeze_backbone=False,
        cond=0,
        weight_path="weights/rainy",
        result="result.txt",
        ckpt=10,
        transfer_learning=False
):
    weights = weight_path
    latest = weights + 'latest.pt'
    temp_latest = weights + 'temp_latest.pt'
    best = weights + 'best.pt'
    yolov3 = "weights/yolov3.weights"
    devices = torch_utils.multiple_devices()
    num_branches = 3

    if multi_scale:  # pass maximum multi_scale size
        img_size = 608
    else:
        torch.backends.cudnn.benchmark = True  # unsuitable for multiscale

    # Configure run
    train_path = parse_data_cfg(data_cfg)['train']

    # Initialize model
    model = MultiDarknet(shared_cfg, diff_cfg, num_branches, img_size)
    darknet_model = Darknet(cfg, img_size)

    # Get dataloader
    dataloader = LoadImagesAndLabels(train_path, batch_size, img_size, multi_scale=multi_scale, augment=True, multi_domain=True, classify_index=[2522, 4730])

    lr0 = 0.001
    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0
    best_loss = float('inf')
    if resume:

        #checkpoint = model.load_weights(weights)
        checkpoint = torch.load(latest, map_location='cpu') #jwpyo

        # Load weights to resume from
        model.load_state_dict(checkpoint['model']) #jwpyo

        # if torch.cuda.device_count() > 1:
        #   model = nn.DataParallel(model)
        model.to(devices[0]).train()
        
        # Set optimizer
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=lr0, momentum=.9)
        start_epoch = checkpoint['epoch'] + 1
        if checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_loss = checkpoint['best_loss']

        del checkpoint  # current, saved

    else:
        if transfer_learning:
            # apply the transfer learning up to cutoff layer
            
            model_dict = model.state_dict()
            
            # load the original yolov3 weights to pretrained_dict_
            darknet_model.load_weights(yolov3)
            pretrained_dict = darknet_model.state_dict()
            pretrained_dict_ = copy.copy(pretrained_dict)
            
            # change the layer's name corresponding with multi-domain's layer
            # and copy the diff branch to others.
            for i in range(num_branches):
                for k, v in pretrained_dict.items():
                    index = int(k.split(".")[1])
                    if index > 74:
                        if i == 0:
                            del pretrained_dict_[k] # let go first domain branch
                        k_arr = k.split(".")
                        k_arr[1] = str(int(k_arr[1]) + 32*i)
                        k_arr2_arr = k_arr[2].split("_")
                        k_arr2_arr[-1] = str(int(k_arr[1]) - (75+32*i))
                        k_arr2 = "_".join(k_arr2_arr)
                        k_arr[2] = k_arr2
                        k = ".".join(k_arr)
                        pretrained_dict_[k] = v
            
            # change the pretrained_dict's layer whose shape is different from multi-domain network. Then initialize Gaussian.
            for k, v in model_dict.items():
                if v.shape != pretrained_dict_[k].shape:
                    pretrained_dict_[k] = torch.empty(v.shape)
                    #TODO: conv, batch
                    if k.split(".")[2].startswith("conv"):
                        nn.init.normal_(pretrained_dict_[k], 0.0, 0.03)
                    elif k.split(".")[2].startswith("batch_norm") and k.split(".")[3] == "weight":
                        nn.init.normal_(pretrained_dict_[k], 1.0, 0.03)
                    elif k.split(".")[2].startswith("batch_norm") and k.split(".")[3] == "bias":
                        nn.init.constant_(pretrained_dict_[k], 0.0)
                    else:
                        nn.init_normal_(pretrained_dict_[k], torch.mean(v), torch.std(v))

                    print(k, v.shape)

            model_dict.update(pretrained_dict_)
            model.load_state_dict(model_dict)
            # freeze layer 
            model.freeze_layer(cutoff)
        
        elif shared_cfg.startswith('bdd100k'):
            load_darknet_weights(model, "weights/yolov3.weights")
            cutoff = 75
        else:
            model.apply(weights_init_normal)
            cutoff = 75
        model.to(devices[0]).train()
        
        # Set optimizer
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=lr0, momentum=.9)

    # Set scheduler
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[54, 61], gamma=0.1)

    t0 = time.time()
    model_info(model)

    n_burnin = min(round(dataloader.nB / 5 + 1), 1000)  # number of burn-in batches
    start_coff = 75
    model_morph_flag = False
    for epoch in range(1, epochs):
        epoch += start_epoch

        print(('%8s%12s' + '%10s' * 7) % (
            'Epoch', 'Batch', 'xy', 'wh', 'conf', 'cls', 'totalLoss', 'nTargets', 'time'))

        # Update scheduler (automatic)
        # scheduler.step()

        # Update scheduler (manual)  at 0, 54, 61 epochs to 1e-3, 1e-4, 1e-5
        if epoch > 50:
            lr = lr0 / 10
        else:
            lr = lr0
        for g in optimizer.param_groups:
            g['lr'] = lr

        ui = -1
        rloss = defaultdict(float)  # running loss
        optimizer.zero_grad()
        
        #dataloader iteration 
        for i, (imgs, targets, _, _, cond) in enumerate(dataloader):
            #print("dataloader iteration, {}".format(i)) 
            if sum([len(x) for x in targets]) < 1:  # if no targets continue
                continue

            # SGD burn-in
            if (epoch == 0) & (i <= n_burnin):
                lr = lr0 * (i / n_burnin) ** 4
                for g in optimizer.param_groups:
                    g['lr'] = lr

            # Compute loss, compute gradient, update parameters
            imgs_0 = imgs.to(devices[0])
            loss = model(imgs_0, targets, cond=cond)
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
        if rloss['loss'] < best_loss:
            best_loss = rloss['loss']

        # Save latest checkpoint
        checkpoint = {'epoch': epoch,
                      'best_loss': best_loss,
                      'model': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, latest)

        # Save best checkpoint
        if best_loss == rloss['loss']:
            os.system('cp ' + latest + ' ' + best)
        
        # Save backup weights every ckpt epochs (optional)
        if (epoch > 0) & (epoch % ckpt == 0):
            os.system('cp ' + latest + ' ' + weights + 'backup{}.pt'.format(epoch))

        # Calculate mAP
        with torch.no_grad():
            mAP, R, P = test.test(shared_cfg, diff_cfg, num_branches, data_cfg, weights=latest, batch_size=batch_size, img_size=img_size)
        # Write epoch results
        with open(result, 'a') as file:
            file.write(s + '%11.3g' * 3 % (mAP, P, R) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=15, help='size of each image batch')
    parser.add_argument('--accumulated-batches', type=int, default=1, help='number of batches before optimizer step')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--shared-cfg', type=str, default='cfg/multidarknet/shared.cfg', help='cfg file path')
    parser.add_argument('--diff-cfg', type=str, default='cfg/multidarknet/diff1.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='cfg/bdd100k/bdd100k_rainy.data', help='coco.data file path')
    parser.add_argument('--multi-scale', action='store_true', help='random image sizes per batch 320 - 608')
    parser.add_argument('--img-size', type=int, default=32 * 13, help='pixels')
    parser.add_argument('--resume', type=bool, default=False, help='resume training flag')
    parser.add_argument('--cond', type=float, default=0, help='test variable')
    parser.add_argument('--weight_path', type=str, default="weights/test/", help="weight path")
    parser.add_argument('--result', type=str, default="result/test.txt", help="result txt file")
    parser.add_argument('--ckpt', type=int, default=5, help="save the weight by this value")
    parser.add_argument('--transfer-learning', type=bool, default=False, help="transfer learning")
    opt = parser.parse_args()
    print(opt, end='\n\n')

    init_seeds()

    train(
        opt.cfg,
        opt.shared_cfg,
        opt.diff_cfg,
        opt.data_cfg,
        img_size=opt.img_size,
        resume=opt.resume,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
        accumulated_batches=opt.accumulated_batches,
        multi_scale=opt.multi_scale,
        cond=opt.cond,
        weight_path=opt.weight_path,
        result=opt.result,
        ckpt=opt.ckpt,
        transfer_learning=opt.transfer_learning
    )
