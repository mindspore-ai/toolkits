import PyInstaller.__main__
import ctypes
ctypes.windll.user32.SetProcessDPIAware()

PyInstaller.__main__.run([
    'gui.py',
    '--onefile',
    '--windowed',
    '--name=DeepInsight',
    '--exclude-module=QtWebEngine*',
    '--exclude-module=QtMultimedia',
    '--exclude-module=QtWebChannel',
    '--exclude-module=QtNetwork',
    '--exclude-module=QtPositioning',
    '--exclude-module=QtQuick',
    '--exclude-module=QtSensors',
    '--noupx',
])

