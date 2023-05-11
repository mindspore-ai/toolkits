"""PyTorch training"""
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as trans
from resnet_convert.resnet_pytorch.resnet import resnet50
import mindspore as ms
#from troubleshooter.migrator.diff_handler import WeightMigrator
import troubleshooter as ts
import pprint
import numpy as np

def train_epoch(epoch, model, loss_fun, device, data_loader, optimizer):
    """Single train one epoch"""
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = loss_fun(output, target.to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(data_loader),
                100. * batch_idx / len(data_loader), loss.item()))


def test_epoch(model, device, data_loader):
    """Single evaluation once"""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data.to(device))
            test_loss += F.nll_loss(output, target.to(device), reduction='sum').item() # sum up batch loss
            pred = output.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.to(device)).sum().item()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.0f}%\n'.format(
        test_loss, 100. * correct / len(data_loader.dataset)))

if __name__ == '__main__':
    use_cuda = False#torch.cuda.is_available()
    device_pt = torch.device("cuda" if use_cuda else "cpu")

    train_transform = trans.Compose([
        trans.RandomCrop(32, padding=4),
        trans.RandomHorizontalFlip(0.5),
        trans.Resize(224),
        trans.ToTensor(),
        trans.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    ])
    test_transform = trans.Compose([
        trans.Resize(224),
        trans.RandomHorizontalFlip(0.5),
        trans.ToTensor(),
        trans.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    ])
    train_set = torchvision.datasets.CIFAR10(root='/mnt/d/06_project/MNIST_Data_py', train=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    test_set = torchvision.datasets.CIFAR10(root='/mnt/d/06_project/MNIST_Data_py', train=False, transform=test_transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
    # 2. define forward network
    net = resnet50(num_classes=10).cuda() if use_cuda else resnet50(num_classes=10)
    pth_path="/mnt/d/06_project/m/resnet_pytorch_res/resnet_pytroch_res/resnet.pth"
    wm = ts.WeightMigrator(pt_model=net, pth_file_path=pth_path, ckpt_save_path='./convert_resnet.ckpt')

    wm.convert(print_conv_info=True)
    # ms_path = "/mnt/d/06_project/m/docs-r1.9/docs-r1.9/docs/mindspore/source_zh_cn/migration_guide/code/resnet_convert/resnet_ms/resnet.ckpt"
    #wm.compare_ckpt(ckpt_path=ms_path, print_result=1)

    pt_para_dict = torch.load(pth_path, map_location='cpu')
    ms_para_dict = ms.load_checkpoint("./convert_resnet.ckpt")

    # 比较转换后的pt与ms的参数值
    w_map = wm.get_weight_map(print_map=True)
    for key, value in w_map[0].items():
        pt_value = pt_para_dict.get(key)
        ms_value = ms_para_dict.get(value)
        orig_value=pt_value.numpy()
        target_value=ms_value.asnumpy()
        result = np.allclose(orig_value, target_value)
        print(result)
