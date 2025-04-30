import os
import pyvis
from PyInstaller import __main__ as pyi_main

pyvis_path = os.path.dirname(pyvis.__file__)
pyvis_templates_path = os.path.join(pyvis_path, "templates")

pyi_main.run([
    'app.py',
    '--name=ExecutingOrderPreCheck',
    '--onefile',
    '--console',
    '--distpath', '.',  # exe输出到当前目录
    '--add-data', 'static;static',
    '--add-data', 'templates;templates',
    '--add-data', f'{pyvis_templates_path};pyvis/templates',
    '--hidden-import=jinja2', 
    '--hidden-import=pyvis.network',
    '--hidden-import=logging.handlers'
])
