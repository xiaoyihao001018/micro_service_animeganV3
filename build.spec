# -*- mode: python ; coding: utf-8 -*-
import os
import onnxruntime as ort

block_cipher = None

# 获取 onnxruntime 路径
ort_path = os.path.dirname(ort.__file__)

# 获取模型文件的绝对路径
model_path = os.path.abspath('AnimeGANv3_Hayao_36.onnx')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型文件不存在: {model_path}")
print(f"模型文件路径: {model_path}")  # 打印路径进行确认

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[
        # 添加 onnxruntime 必需的 DLL 文件
        (os.path.join(ort_path, 'capi', 'onnxruntime_providers_shared.dll'), '.'),
        (os.path.join(ort_path, 'capi', 'onnxruntime_providers_cuda.dll'), '.'),
    ],
    datas=[
        (model_path, '.'),  # 使用绝对路径
        (os.path.join(ort_path, 'capi'), 'onnxruntime/capi'),  # onnxruntime 相关文件
    ],
    hiddenimports=[
        'onnxruntime',
        'onnxruntime.capi._pybind_state',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'onnxruntime-gpu',  # 排除 GPU 相关
        'cudnn',
        'cudart',
        'cuda',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='AnimeGANv3_Server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
