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
                print('The mapping file download failed.')
                return None
            self._api_maps[version] = content

        if not pt_filter:
            pt_filter = lambda x: x
        if not ms_filter:
            ms_filter = lambda x: x
        table_pattern = re.compile(
            r'\|.*?API.*API.*说明.*\|\n\|.*?-+.*?\|.*?-+.*?\|.*?-+.*?\|\n((\|.*?\|.*?\|.*?\|\s*\n?)*)'
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
