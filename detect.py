import argparse
import shutil
import time
import ast
from pathlib import Path
from sys import platform

from multitask_models import *
#from models import *
from utils.datasets import *
from utils.utils import *


def detect(
        cfg,
        shared_cfg,
        diff_cfgs,
        weights,
        images,
        class_name,
        output='output',  # output folder
        img_size=416,
        conf_thres=0.3,
        nms_thres=0.45,
        save_txt=False,
        save_images=True,
        webcam=False,
        multi_domain=False,
        cond=0
):
    device = torch_utils.select_device()
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder

    # Initialize model
    if multi_domain:
        model = MultiDarknet(shared_cfg, ast.literal_eval(diff_cfgs), img_size)
    else:
        model = Darknet(cfg, img_size)
    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        if weights.endswith('yolov3.pt') and not os.path.exists(weights):
            if (platform == 'darwin') or (platform == 'linux'):
                os.system('wget https://storage.googleapis.com/ultralytics/yolov3.pt -O ' + weights)
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)
    #load_darknet_weights(model, weights)
    model.to(device).eval()

    # Set Dataloader
    if webcam:
        save_images = False
        dataloader = LoadWebcam(img_size=img_size)
    else:
        dataloader = LoadImages(images, img_size=img_size)

    # Get classes and colors
    classes = load_classes(parse_data_cfg(class_name)['names'])
    colors = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(len(classes))]

    for i, (path, img, im0) in enumerate(dataloader):
        t = time.time()
        if webcam:
            print('webcam frame %g: ' % (i + 1), end='')
        else:
            print('image %g/%g %s: ' % (i + 1, len(dataloader), path), end='')
        save_path = str(Path(output) / Path(path).name)

        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        if ONNX_EXPORT:
            torch.onnx.export(model, img, 'weights/model.onnx', verbose=True)
            return
        
        if multi_domain:
            pred = model(img, None, cond)
        else:
            pred = model(img)
        
        pred = pred[pred[:, :, 4] > conf_thres]  # remove boxes < threshold

        if len(pred) > 0:
            # Run NMS on predictions
            detections = non_max_suppression(pred.unsqueeze(0), conf_thres, nms_thres)[0]

            # Rescale boxes from 416 to true image size
            scale_coords(img_size, detections[:, :4], im0.shape).round()

            # Print results to screen
            unique_classes = detections[:, -1].cpu().unique()
            for c in unique_classes:
                n = (detections[:, -1].cpu() == c).sum()
                print('%g %ss' % (n, classes[int(c)]), end=', ')

            # Draw bounding boxes and labels of detections
            for x1, y1, x2, y2, conf, cls_conf, cls in detections:
                if save_txt:  # Write to file
                    with open(save_path + '.txt', 'a') as file:
                        file.write('%g %g %g %g %g %g\n' %
                                   (x1, y1, x2, y2, cls, cls_conf * conf))

                # Add bbox to the image
                label = '%s %.2f' % (classes[int(cls)], conf)
                plot_one_box([x1, y1, x2, y2], im0, label=label, color=colors[int(cls)])
                #plot_one_box([x1, y1, x2, y2], im0, color=colors[int(cls)])

        dt = time.time() - t
        print('Done. (%.3fs)' % dt)

        if save_images:  # Save generated image with detections
            cv2.imwrite(save_path, im0)

        if webcam:  # Show live webcam
            cv2.imshow(weights, im0)

    if save_images and (platform == 'darwin'):  # linux/macos
        os.system('open ' + output + ' ' + save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/bdd100k/bdd100k.cfg', help='cfg file path')
    parser.add_argument('--shared-cfg', type=str, default='cfg/multidarknet/shared.cfg', help='cfg file path')
    parser.add_argument('--diff-cfgs', type=str, default="['cfg/multidarknet/diff1.cfg','cfg/multidarknet/diff2.cfg','cfg/multidarknet/diff3.cfg']", help='cfg file path')
    parser.add_argument('--weights', type=str, default='weights/rainy/multidomain/best.pt', help='path to weights file')
    parser.add_argument('--images', type=str, default='val_videos/rainy_night/b2064e61-2beadd45', help='path to images')
    parser.add_argument('--class_name', type=str, default='cfg/bdd100k/bdd100k.data', help='path to configure file')
    parser.add_argument('--img-size', type=int, default=32 * 13, help='size of each image dimension')
    parser.add_argument('--conf-thres', type=float, default=0.6, help='object confidence threshold. NMS remove the bbox lower than conf-thres')
    parser.add_argument('--nms-thres', type=float, default=0.4, help='iou threshold for non-maximum suppression. NMS remove the bbox higher than nms-thres')
    parser.add_argument('--multi-domain', type=bool, default=False, help='True if you use multi darknet')
    parser.add_argument('--cond', type=int, default=0, help='Detect condition if you use multi-domain')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(
            opt.cfg,
            opt.shared_cfg,
            opt.diff_cfgs,
            opt.weights,
            opt.images,
            opt.class_name,
            img_size=opt.img_size,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres,
            multi_domain=opt.multi_domain,
            cond=opt.cond
        )
