from ops_match import *

pt_list = OPSList("pytorch")
ms_list = OPSList("mindspore")

model = 'mobilenet'
pt_list.Construct(f"demo_net/{model}_pt.pkl")
GetUniIO()(pt_list.ops_list)
ms_list.Construct(f"demo_net/{model}_ms.pkl")

print('--------------------------')
apis_map = FlowMatch()(pt_list, ms_list, 1)
print_apis_map(apis_map)