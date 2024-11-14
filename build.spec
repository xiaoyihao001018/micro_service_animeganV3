# -*- mode: python ; coding: utf-8 -*-
import os
import glob
import onnxruntime as ort

block_cipher = None

# 获取 onnxruntime 路径
ort_path = os.path.dirname(ort.__file__)

# 获取所有 .onnx 模型文件
model_files = glob.glob('*.onnx')
if not model_files:
    raise FileNotFoundError("未找到任何 .onnx 模型文件")

# 打印找到的模型文件
print(f"找到以下模型文件:")
for model in model_files:
    print(f"- {model}")

# 构建 datas 列表,包含所有模型文件
model_datas = [(os.path.abspath(model), '.') for model in model_files]

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[
        # 添加 onnxruntime 必需的 DLL 文件
        (os.path.join(ort_path, 'capi', 'onnxruntime_providers_shared.dll'), '.'),
    ],
    datas=[
        *model_datas,  # 添加所有模型文件
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
