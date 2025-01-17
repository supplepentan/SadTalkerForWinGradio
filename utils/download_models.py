import os
import urllib.request

# ディレクトリの作成
os.makedirs("./checkpoints", exist_ok=True)

# ダウンロードするファイルのリスト
files_to_download = {
    "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar": "./checkpoints/mapping_00109-model.pth.tar",
    "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar": "./checkpoints/mapping_00229-model.pth.tar",
    "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors": "./checkpoints/SadTalker_V0.0.2_256.safetensors",
    "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_512.safetensors": "./checkpoints/SadTalker_V0.0.2_512.safetensors",
    "https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth": "./checkpoints/alignment_WFLW_4HG.pth",
    "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth": "./checkpoints/detection_Resnet50_Final.pth",
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth": "./checkpoints/GFPGANv1.4.pth",
    "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth": "./checkpoints/parsing_parsenet.pth",
}

# ファイルのダウンロードと保存
for url, file_path in files_to_download.items():
    if not os.path.exists(file_path):
        urllib.request.urlretrieve(url, file_path)
        print(f"Downloaded: {file_path}")
    else:
        print(f"File already exists: {file_path}")
