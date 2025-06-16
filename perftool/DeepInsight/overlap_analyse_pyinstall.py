import PyInstaller.__main__

PyInstaller.__main__.run([
    'overlap_analyse_gui.py',
    '--onefile',
    '--windowed',
    '--name=ProfileAnalyzer',
    '--exclude-module=QtWebEngineWidgets',
    '--exclude-module=QtWebEngineCore',
    '--exclude-module=QtWebChannel',
    '--exclude-module=QtNetwork',
])
