# SPDX-License-Identifier: Apache-2.0
# 
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
setup script
"""
import os
import shutil
import stat

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
from setuptools.command.egg_info import egg_info

version = '1.0.19'

cur_dir = os.path.dirname(os.path.realpath(__file__))
pkg_dir = os.path.join(cur_dir, 'build')


def clean():
    """clean"""
    def readonly_handler(func, path, execinfo):
        os.chmod(path, stat.S_IWRITE)
        func(path)
    if os.path.exists(os.path.join(cur_dir, 'build')):
        shutil.rmtree(os.path.join(cur_dir, 'build'), onerror=readonly_handler)
    if os.path.exists(os.path.join(cur_dir, 'troubleshooter.egg-info')):
        shutil.rmtree(os.path.join(cur_dir, 'troubleshooter.egg-info'), onerror=readonly_handler)


def write_version(file):
    """write version"""
    file.write("__version__ = '{}'\n".format(version))


def build_depends():
    """generate python file"""
    version_file = os.path.join(cur_dir, 'troubleshooter/', 'version.py')
    with open(version_file, 'w') as f:
        write_version(f)


clean()
build_depends()


def update_permissions(path):
    # Refer to
    # https://gitee.com/mindspore/xai/blob/v1.4.0/setup.py
    """
    Update permissions.

    Args:
        path (str): Target directory path.
    """
    for dirpath, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            dir_fullpath = os.path.join(dirpath, dirname)
            os.chmod(dir_fullpath, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC | stat.S_IRGRP | stat.S_IXGRP)
        for filename in filenames:
            file_fullpath = os.path.join(dirpath, filename)
            os.chmod(file_fullpath, stat.S_IREAD)


class EggInfo(egg_info):
    """Egg info."""
    def run(self):
        super().run()
        egg_info_dir = os.path.join(cur_dir, 'troubleshooter.egg-info')
        update_permissions(egg_info_dir)


class BuildPy(build_py):
    """BuildPy."""
    def run(self):
        super().run()
        troubleshooter_dir = os.path.join(pkg_dir, 'lib', 'troubleshooter')
        update_permissions(troubleshooter_dir)


setup(
    name='troubleshooter',
    version=version,
    author='The MindSpore Authors',
    author_email='contact@mindspore.cn',
    url='https://www.mindspore.cn/',
    project_urls={
        'Sources': 'https://gitee.com/mindspore/toolkits/tree/master/troubleshooter',
        'Issue Tracker': 'https://gitee.com/mindspore/toolkits/issues',
    },
    description='A troubleshooting toolbox for MindSpore.',
    license='Apache 2.0',
    packages=find_packages(),
    include_package_data=True,
    cmdclass={
        'egg_info': EggInfo,
        'build_py': BuildPy,
    },
    install_requires=[
        'wheel',
        'tdqm',
        'numpy',
        'pyyaml',
        'pandas',
        'matplotlib',
        'PySnooper >= 1.1.0',
        'prettytable >= 3.0.0, < 3.8.0',
        'protobuf >= 3.13.0, <= 3.20.1',
    ],
    classifiers=[
        'License :: OSI Approved :: Apache Software License'
    ]
)
print(find_packages())
