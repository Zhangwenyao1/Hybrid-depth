import os
import glob

for file_path in glob.glob('/code/CFMDE-main/manydepth/manydepth/exps/**/adam.pth', recursive=True):
    try:
        os.remove(file_path)
        print(f"Delete: {file_path}")
    except Exception as e:
        print(f"Fail: {file_path}, Reason:{e}")