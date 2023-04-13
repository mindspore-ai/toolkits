import troubleshooter as ts


@ts.proposal()
def main():
    error = """
    Traceback (most recent call last):
      File "/mnt/d/06_project/trouble-shooter/examples/proposal_demo_1.py", line 25, in <module>
        out = net(input_x, input_a, input_b)
      File "/root/envs/lib/python3.7/site-packages/mindspore/nn/cell.py", line 596, in __call__
        out = self.compile_and_run(*args)
      File "/root/envs/lib/python3.7/site-packages/mindspore/nn/cell.py", line 985, in compile_and_run
        self.compile(*inputs)
      File "/root/envs/lib/python3.7/site-packages/mindspore/nn/cell.py", line 957, in compile
        jit_config_dict=self._jit_config_dict)
      File "/root/envs/lib/python3.7/site-packages/mindspore/common/api.py", line 1131, in compile
        result = self._graph_executor.compile(obj, args_list, phase, self._use_vm_mode())
    ValueError: Cannot join the return values of different branches, perhaps you need to make them equal.
    Shape Join Failed: shape1 = (2, 3, 4, 5), shape2 = ().
    For more details, please refer to https://www.mindspore.cn/search?inputValue=Shape%20Join%20Failed
    ......    
    ----------------------------------------------------
    - C++ Call Stack: (For framework developers)
    ----------------------------------------------------
    mindspore/ccsrc/pipeline/jit/static_analysis/static_analysis.cc:850 ProcessEvalResults
           """
    raise ValueError(error)


if __name__ == '__main__':
    main()
