import requests
import re
import os


class GetPTAPIDict:
    def __init__(self) -> None:
        self._api_maps = {}

    def __call__(self, version='master', pt_filter=None, ms_filter=None):
        content = self._api_maps.get(version)
        if content is None:
            try:
                url = f"https://gitee.com/mindspore/docs/raw/{version}/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_api_mapping.md"
                response = requests.get(url, timeout=5)
                content = response.content.decode("utf-8")
            except Exception as e:
                print('The mapping file download failed, use local.')
                with open(
                    os.path.join(os.path.dirname(__file__), 'pytorch_api_mapping.md'),
                    'r',
                    encoding='UTF-8',
                ) as f:
                    content = f.read()
            self._api_maps[version] = content

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
            # 忽略表格的「说明」
            # table_dict = {pt_filter(k):ms_filter(v) for k, v, c in item_content if c in ['一致', '功能一致，参数名不同']}
            table_dict = {
                pt_filter(k): ms_filter(v.split(']')[0][1:])
                for k, v, c in item_content
                if v != 'None'
            }
            api_dict.update(table_dict)

        return api_dict


get_pt_api_dict = GetPTAPIDict()

if __name__ == "__main__":
    import yaml

    ret = get_pt_api_dict()

    ms_ops = {}
    pt_ops = {}

    for i in ret.keys():
        ops = i.split('.')[-1]
        ops_type = '.'.join(i.split('.')[:-1])
        if ops_type in pt_ops:
            pt_ops[ops_type].append(ops)
        else:
            pt_ops[ops_type] = [ops]

    for i in ret.values():
        ops = i.split('.')[-1]
        ops_type = '.'.join(i.split('.')[:-1])
        if ops_type in ms_ops:
            ms_ops[ops_type].append(ops)
        else:
            ms_ops[ops_type] = [ops]
    # print(ms_ops)
    with open('./ms.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(ms_ops, stream=f, allow_unicode=True)
    with open('./pt.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(pt_ops, stream=f, allow_unicode=True)
