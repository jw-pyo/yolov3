import torch


def init_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(force_cpu=False):
    if force_cpu:
        cuda = False
        device = torch.device('cpu')
    else:
        cuda = torch.cuda.is_available()
        device = torch.device('cuda:0' if cuda else 'cpu')
        print(device)
        if torch.cuda.device_count() > 1:
            print('Found %g GPUs' % torch.cuda.device_count())
            print('WARNING Multi-GPU Issue: https://github.com/ultralytics/yolov3/issues/21')
            torch.cuda.set_device(0)  # OPTIONAL: Set your GPU if multiple available
        #    # print('Using ', torch.cuda.device_count(), ' GPUs')

    print('Using %s %s\n' % (device.type, torch.cuda.get_device_properties(0) if cuda else ''))
    return device

def multiple_devices(force_cpu=False):
    if force_cpu:
        cuda = False
        devices = torch.device('cpu')
    else:
        cuda = torch.cuda.is_available()
        if cuda:
            devices = [torch.device('cuda:{}'.format(i)) for i in range(torch.cuda.device_count())]
            for i, device in enumerate(devices):
                print(device)
                print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,3), 'GB')
                print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,3), 'GB')
    return devices

def check_model_epoch(weight_path=None):
    if weight_path is None:
        print("Weight path is not set")
        return

    checkpoint = torch.load(weight_path, map_location='cpu')
    print(checkpoint['epoch'])

def training_time(result_path=None):
    if result_path is None:
        print("Result path is not set")
        return
    with open(result_path, "r") as f:
        total_time = 0
        index = 0
        for line in f:
            batch_time, batch_num = float(line.split()[-4]), int(line.split()[1].split("/")[1])
            total_time += batch_time * batch_num
            index += 1
        print("Average Epoch training time: {} min".format(total_time/(index*60)))


            

if __name__ == "__main__":
    #select_device()
    #multiple_devices()
    #check_model_epoch("/lab_shared/datasets/self_driving/berkeley_deepdrive/bdd100k/weights/rainy/best.pt")
    training_time("/home/jwpyo/priv_prj/yolov3/result/rainy/singledomain/train_general/rainy_freeze_30.txt")
    training_time("/home/jwpyo/priv_prj/yolov3/result/rainy/multidomain/rainy_md_10.txt")
