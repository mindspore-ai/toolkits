import yaml
import torch
import troubleshooter as  ts
from models.yolo import  Model

hpy='/mnt/d/06_project/toolkits/troubleshooter/examples/migrator/yolov7-main/yolov7-main/data/hyp.scratch.p5.yaml'
cfg_path='/mnt/d/06_project/toolkits/troubleshooter/examples/migrator/yolov7-main/yolov7-main/cfg/baseline/yolov3.yaml'
weights = '/mnt/d/06_project/toolkits/troubleshooter/examples/migrator/yolov7-main/yolov7-tiny.pt'

with open(hpy) as f:
    hyp = yaml.load(f,Loader=yaml.SafeLoader)

model = Model(cfg_path, ch=3,nc=1,anchors=hyp.get('anchors')).to("cpu")

model_1 = torch.load(weights,map_location='cpu')


wm = ts.WeightMigrator(pt_model=model,pth_file_path=weights,ckpt_save_path='./abc.ckpt')
wm.convert()
