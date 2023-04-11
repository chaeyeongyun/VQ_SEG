import zipfile
from glob import glob
import os
from tqdm import tqdm

def unzip(zipfile_path, extract_to):
    with zipfile.ZipFile(zipfile_path, 'r') as zip:
        zip.extractall(extract_to)

def unzip_all(path_head, extract_to):
    files = glob(path_head+'*.zip')
    for file in tqdm(files):
        unzip(file, extract_to)

if __name__ == '__main__':
    root = '/content/downloads'
    name_list = ["ckpoints-20230411T021348Z"]
    for name in name_list:
        path = os.path.join(root, name)
        print(path)
        unzip_all(path, root)