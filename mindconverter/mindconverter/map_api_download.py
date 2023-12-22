import requests
import re
import json
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
#获取算子字典
def get_ops_dict(version='master', pt_filter=None, ms_filter=None):
    url = f"https://gitee.com/mindspore/docs/raw/{version}/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_api_mapping.md"
    response = requests.get(url)
    content = response.content.decode("utf-8")

    if not pt_filter:
        pt_filter = lambda x: x
    if not ms_filter:
        ms_filter = lambda x: x
    table_pattern = re.compile(
        r'\|.*?API.*API.*说明.*\|\n\|.*?-+.*?\|.*?-+.*?\|.*?-+.*?\|\n((\|.*?\|.*?\|.*?\|\n?)*)'
    )
    item_pattern = re.compile(
        r'\|.*?\[([\w\.]+)\]\(.*?\).*?\|.*?(None|\[[\w\.]+\]\(.*?\)).*?\|\s*?(.*?)\s*?\|'
    )
    table_content = re.findall(table_pattern, content)
    api_dict = {}
    for t in table_content:
        item_content = re.findall(item_pattern, str(t))
        # table_dict = {pt_filter(k):ms_filter(v) for k, v, c in item_content if c in ['一致', '功能一致，参数名不同']}
        table_dict = {pt_filter(k): ms_filter(v.split(']')[0][1:]) for k, v, c in item_content if v != 'None'}
        api_dict.update(table_dict)
    return api_dict

def get_api_type(type="torch"):
    api_dict = {}
    ret = get_ops_dict()
    for k,v in ret.items():
        if(v.split(".")[1] not in ["ops", "nn", "Tensor", "experimental"]):
            continue
        cur_type = k.split(".")
        if(type == "torch"):
            if(cur_type[0] == "torch" and cur_type[1] != "nn" and cur_type[1] != "Tensor" and cur_type[1] != "distributions" and cur_type[1] != "distributed"and cur_type[1] != "optim"):
                api_dict.update({k:v})
        elif(type == "nn"):
            if(cur_type[1] == "nn" and cur_type[2] != "functional"):
                api_dict.update({k[6:]:v})
        elif(type == "nn.functional"):
            if(cur_type[1] == "nn" and cur_type[2] == "functional"):
                api_dict.update({k:v})
        elif(type == "tensor"):
            if(cur_type[0] == "torch" and cur_type[1] == "Tensor"):
                api_dict.update({k[12:]:v[16:]})
        elif(type == "optim"):
            if(cur_type[0] == "torch" and cur_type[1] == "optim"):
                api_dict.update({k[6:]:v})
    return api_dict

#匹配算子的输入参数
def get_ops_input(ms_version='master', pt_version='stable', ms_ops=None, pt_ops=None, type="torch"):
    ret = get_api_type(type)
    ops_mapping = {}
    IS_REQUIRED="REQUIRED"
    for k,v in ret.items():
        if(type == "tensor"):
            if(v[0] != "."):
                continue

        ops_api = {
            "ms_api": [],
            "pt_api": [],
            "ms2pt_mapping": {},
            "gen_explicit_map": None
        }
        #Mindpsore API
        ops_name = v
        if(type == "tensor"):
            ops_name_tensor = ops_name
            ops_name = "mindspore.Tensor" + ops_name_tensor
            ops_api["ms_api"].append(ops_name_tensor)
        else:
            if(ops_name[10:22] == "experimental"):
                ops_api["ms_api"].append(ops_name[23:])
            else:
                ops_api["ms_api"].append(ops_name[10:])
            
        if(type == "torch" or type == "nn.functional" or ops_name == "ops.clip_by_norm" or ops_name == "ops.clip_by_value"):
            url = f"https://www.mindspore.cn/docs/zh-CN/{ms_version}/api_python/ops/{ops_name}.html"
        elif(type == "nn"):
            url = f"https://www.mindspore.cn/docs/zh-CN/{ms_version}/api_python/nn/{ops_name}.html"
        elif(type == "tensor"):
            url = f"https://www.mindspore.cn/docs/zh-CN/{ms_version}/api_python/mindspore/Tensor/{ops_name}.html"
        elif(type == "optim"):
            if(ops_name[10:12] == "nn"):
                url = f"https://www.mindspore.cn/docs/zh-CN/{ms_version}/api_python/nn/{ops_name}.html"
            else:
                url = f"https://www.mindspore.cn/docs/zh-CN/{ms_version}/api_python/experimental/optim/{ops_name}.html"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()
        pattern = ops_name +"\((.*)\)"
        match = re.findall(pattern, text)
        parameters = []
        for i in range(len(match)):
            if(match[i] != ''):
                parameters = match[i].split(", ")
                break
        if(parameters == []):
            print("【警告】在mindspore文档搜索函数参数失败或者对应文档中函数参数为空,函数名为"+ops_name)
        # try:
        #     parameters = match[0].split(", ")
        # except:
        #     print("【错误】在mindspore文档搜索函数参数失败,函数名为"+ops_name)
        # print(f"函数 {ops_name} 的参数列表为：{parameters}")
        # print("MS_PARAM")
        # print(parameters)
        param_dict_ms = {}
        for param in parameters:
            if(param == "*"):
                continue
            if(param.find("=") != -1):
                default_param = param.split("=")
                if(default_param[1] == "False"):
                    param_dict_ms.update({default_param[0]:False})
                elif(default_param[1] == "True"):
                    param_dict_ms.update({default_param[0]:True})
                elif(default_param[1] == "None"):
                    param_dict_ms.update({default_param[0]:None})
                else:
                    try:
                        param_dict_ms.update({default_param[0]:float(default_param[1])})
                    except:
                        param_dict_ms.update({default_param[0]:default_param[1]})
            else:
                param_dict_ms.update({param:IS_REQUIRED})
        ops_api["ms_api"].append(param_dict_ms)
        #Pytorch API
        ops_name = k
        if(type == "nn.functional"):
            ops_name_F = "F." + ops_name[20:]
            ops_api["pt_api"].append(ops_name_F)
        elif(type == "nn"):
            ops_name_nn = ops_name
            ops_name = "torch." + ops_name_nn
            ops_api["pt_api"].append(ops_name_nn)
        elif(type == "tensor"):
            ops_name_tensor = ops_name
            ops_name = "torch.Tensor" + ops_name_tensor
            ops_api["pt_api"].append(ops_name_tensor)
        elif(type == "optim"):
            ops_name_optim = ops_name
            ops_name = "torch." + ops_name_optim
            ops_api["pt_api"].append(ops_name_optim)
        else:
            ops_api["pt_api"].append(ops_name)

        url = f"https://pytorch.org/docs/{pt_version}/generated/{ops_name}.html"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()
        if(type == "tensor"):
            pattern = ops_name_tensor[1:] + "\((.*)\)"
        else:
            pattern = ops_name + "\((.*)\)"
        match = re.findall(pattern, text)
        parameters = []
        for i in range(len(match)):
            if(match[i] != ''):
                parameters = match[i].split(", ")
                break
        if(parameters == []):
            print("【警告】在torch文档搜索函数参数失败或者对应文档中函数参数为空,函数名为"+ops_name)
        # try:
        #     parameters = match[0].split(", ")
        # except:
        #     print("【错误】在torch文档搜索函数参数失败,函数名为"+ops_name)
        # print(f"函数 {ops_name} 的参数列表为：{parameters}")
        # print("TORCH_PARAM")
        # print(parameters)
        param_dict_pt = {}
        for param in parameters:
            if(param == "*"):
                continue
            if(param.find("=") != -1):
                default_param = param.split("=")
                if(default_param[1] == "False"):
                    param_dict_pt.update({default_param[0]:False})
                elif(default_param[1] == "True"):
                    param_dict_pt.update({default_param[0]:True})
                elif(default_param[1] == "None"):
                    param_dict_pt.update({default_param[0]:None})
                else:
                    try:
                        param_dict_pt.update({default_param[0]:float(default_param[1])})
                    except:
                        param_dict_pt.update({default_param[0]:default_param[1]})
            else:
                param_dict_pt.update({param:IS_REQUIRED})
        ops_api["pt_api"].append(param_dict_pt)
        #生成参数对应表
        print("正在完成MindSpore算子 %s 到Pytorch算子 %s 的参数解析"% (v,k))
        ms_param_list = list(param_dict_ms.keys())
        pt_param_list = list(param_dict_pt.keys())
        ms_param_num = len(ms_param_list)
        pt_param_num = len(pt_param_list)
        for i in range(ms_param_num):
            flag = False
            if(ms_param_list[i] == "learning_rate"):
                ops_api["ms2pt_mapping"].update({"learning_rate":"lr"})
                continue
            for j in range(pt_param_num):
                if(ms_param_list[i] == pt_param_list[j]):
                    ops_api["ms2pt_mapping"].update({ms_param_list[i]:pt_param_list[j]})
                    flag = True
            if(flag == False and i < pt_param_num):
                if(ms_param_list[i][0:1] == "x" or ms_param_list[i][0:5] == "input" or ms_param_list[i][0:6] == "*input" or ms_param_list[i][0:4] == "axis" or ms_param_list[i][0:9] == "multiples"):
                    ops_api["ms2pt_mapping"].update({ms_param_list[i]:pt_param_list[i]})
                elif(fuzz.partial_ratio(ms_param_list[i],pt_param_list[i]) > 50):
                    ops_api["ms2pt_mapping"].update({ms_param_list[i]:pt_param_list[i]})

        if(type == "nn.functional"):
            ops_mapping[ops_name_F] = ops_api
        elif(type == "nn"):
            ops_mapping[ops_name_nn] = ops_api
        elif(type == "tensor"):
            ops_mapping[ops_name_tensor] = ops_api
        elif(type == "optim"):
            ops_mapping[ops_name_optim] = ops_api
        else:
            ops_mapping[ops_name] = ops_api


    return ret, ops_mapping
#API映射表的更新
def update_ops_mapping():
    pass

if __name__ == "__main__":
    #torch
    torch_list, torch_dot_mappings = get_ops_input(type="torch")
    torch_dot_list = list(torch_list.keys())
    with open('./testdir/ops/torch_dot_list.json', 'w', encoding='utf-8') as f:
        json.dump(torch_dot_list, f, indent=2)
    with open('./testdir/mappings/torch_dot_mappings.json', 'w', encoding='utf-8') as f:
        json.dump(torch_dot_mappings, f, indent=2)

    #torch.nn
    torch_nn_list, nn_mappings = get_ops_input(type="nn")
    nn_list = list(torch_nn_list.keys())
    with open('./testdir/ops/nn_list.json', 'w', encoding='utf-8') as f:
        json.dump(nn_list, f, indent=2)
    with open('./testdir/mappings/nn_mappings.json', 'w', encoding='utf-8') as f:
        json.dump(nn_mappings, f, indent=2)

    #torch.nn.functional
    nn_f_list, f_mappings = get_ops_input(type="nn.functional")
    f_list = list(nn_f_list.keys())
    with open('./testdir/ops/f_list.json', 'w', encoding='utf-8') as f:
        json.dump(f_list, f, indent=2)
    with open('./testdir/mappings/f_mappings.json', 'w', encoding='utf-8') as f:
        json.dump(f_mappings, f, indent=2)
    
    #torch.tensor
    tensor_list, tensor_dot_mappings = get_ops_input(type="tensor")
    tensor_dot_list = list(tensor_list.keys())
    with open('./testdir/ops/tensor_dot_list.json', 'w', encoding='utf-8') as f:
        json.dump(tensor_dot_list, f, indent=2)
    with open('./testdir/mappings/tensor_dot_mappings.json', 'w', encoding='utf-8') as f:
        json.dump(tensor_dot_mappings, f, indent=2)
    
    #torch.optim
    torch_optim_list, torch_optim_mappings = get_ops_input(type="optim")
    torch_optim_list = list(torch_optim_list.keys())
    with open('./testdir/ops/torch_optim_list.json', 'w', encoding='utf-8') as f:
        json.dump(torch_optim_list, f, indent=2)
    with open('./testdir/mappings/torch_optim_mappings.json', 'w', encoding='utf-8') as f:
        json.dump(torch_optim_mappings, f, indent=2)
