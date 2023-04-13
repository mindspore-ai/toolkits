import os
import re
# print(os.path.dirname("/mnt/d/06_project/trouble-shooter/tests/tracker/test_lenet.py"))

path = "/root/miniconda3/envs/miaoym/lib/python3.7/inspect.py"
root_path = "/aa/dd"


def compare_key_words(err_msg, key_word):
    """Find out whether the key log information is in the exception or not."""
    return key_word and re.search(key_word, err_msg)

#print(not path.find("site-packages/mindspore"))
#print(not path.find(root_path))

#if not path.find("site-packages/mindspore") and not path.find(root_path):
#    print("True")
#else:
#    print("False")
"""
def test_kwargs(first, *args, **kwargs):
   print('Required argument: ', first)
   print(type(kwargs))
   for v in args:
      print ('Optional argument (args): ', v)

   if kwargs.get('k3'):
      kwargs.pop('k1')
   for k, v in kwargs.items():
      print ('Optional argument %s (kwargs): %s' % (k, v))

test_kwargs(1, 2, 3, 4, k1=5, k2=6)
"""


def test_re():

    error_msg = """For 'MirrorPad', all elements of paddings must be >= 0.
    ----------------------------------------------------
    - The Traceback of Net Construct Code:
    ----------------------------------------------------
    The function call stack (See file '/mnt/d/06_project/trouble-shooter/tests/working/rank_0/om/analyze_fail.dat' for more details. Get instructions about `analyze_fail.dat` at https://www.mindspore.cn/search?inputValue=analyze_fail.dat):
    # 0 In file /root/miniconda3/envs/miaoym1_9/lib/python3.7/site-packages/mindspore/nn/layer/basic.py:931
            if self.mode == "CONSTANT":
    # 1 In file /root/miniconda3/envs/miaoym1_9/lib/python3.7/site-packages/mindspore/nn/layer/basic.py:934
                x = self.pad(x, self.paddings)
                    ^
    
    ----------------------------------------------------
    - C++ Call Stack: (For framework developers)
    ----------------------------------------------------
    mindspore/core/ops/mirror_pad.cc:84 InferShape"""
    key_words="all elements of .*paddings.* must be >= 0"

    if compare_key_words(error_msg, key_words):
        print(error_msg)


def test_split():
    error_message = """For 'MirrorPad', all elements of paddings must be >= 0.
    gfgfgfgfg
    The function call stack (See file '/mnt/d/06_project/trouble-shooter/tests/working/rank_0/om/analyze_fail.dat' for more details. Get instructions about `analyze_fail.dat` at https://www.mindspore.cn/search?inputValue=analyze_fail.dat):
    # 0 In file /root/miniconda3/envs/miaoym1_9/lib/python3.7/site-packages/mindspore/nn/layer/basic.py:931
            if self.mode == "CONSTANT":
    # 1 In file /root/miniconda3/envs/miaoym1_9/lib/python3.7/site-packages/mindspore/nn/layer/basic.py:934
                x = self.pad(x, self.paddings)
                    ^

    mindspore/core/ops/mirror_pad.cc:84 InferShape"""
    #key_words = "all elements of .*paddings.* must be >= 0"
    msg_list = error_message.split('----------------------------------------------------')
    #print(rs)
    format_msg={}
    current_key = None

    for msg in msg_list:
        if msg_list.index(msg) == 0:
            format_msg["error_message"] = msg
            continue

        msg = msg.strip().strip(os.linesep)
        if msg.startswith("- "):
            current_key = msg[2:]
            continue
        if current_key:
            format_msg[current_key] = msg

    print(format_msg)


#test_re()
test_split()