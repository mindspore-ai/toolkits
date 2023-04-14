"""PyTorch training"""
import torch
import mindspore
import troubleshooter as ts


class msNet(mindspore.nn.Cell):
    def __init__(self):
        super(msNet, self).__init__()
        self.conv1d = mindspore.nn.Conv1d(256, 256, kernel_size=1, has_bias=True)
    def construct(self, A):
        return self.conv1d(A)

class torchNet(torch.nn.Module):
    def __init__(self):
        super(torchNet, self).__init__()
        self.conv1d = torch.nn.Conv1d(256, 256, kernel_size=1)
    def forward(self, A):
        return self.conv1d(A)


if __name__ == '__main__':

    #ms_net = msNet()
    #A = ms.Tensor(np.ones((1, 256, 32), np.float32))
    #res = ms_net(A)
    #print("----------------MINDSPORE conv1d-----------")
    #for param in ms_net.trainable_params():
    #    print(param)

    torch_net=torchNet()
    #A=torch.Tensor(np.ones((1,256,32), np.float32))
    #res=torch_net(A)
    torch.save(torch_net.state_dict(), "torch_net.pth")

    pth_path = "./torch_net.pth"
    wm = ts.weight_migrator(pt_model=torch_net, pth_file_path=pth_path, ckpt_save_path='./convert_resnet.ckpt')
    #w_map = wm.get_weight_map(print_map=True)
    # test_map = {'bn1.bias': 'bn1.beta',}
    wm.convert(print_conv_info=True)
    #wm.compare_ckpt(ckpt_path=ms_path, print_result=1)