class BinaryHeap():
    def __init__(self):
        self.items = [None]
    
    def __len__(self):
        return  len(self.items) - 1
    
    def insert(self, k):
        self.items.append(k)
        self._percolate_up()   
    
    def _percolate_up(self):
        i = len(self)
        parent = i//2
        while parent >= 0:
            if self.items[i] < self.items[parent]:
                self.items[parent], self.items[i] = self.items[i], self.items[parent]
            i = parent
            parent = i // 2
    
    def _percolate_down(self, idx):
        

# from glob import glob
# import os.path as osp
# import os
# from collections import defaultdict
# import shutil
# from tqdm import tqdm
# folder = "/content/downloads/cut_car"
# save_folder = "/content/downloads/cut_car_folds"
# if not osp.isdir(save_folder): os.mkdir(save_folder)
# imgs = glob(osp.join(folder, "*.png"))
# d = defaultdict(list)
# for img in tqdm(imgs):
#     filename = osp.split(img)[-1]
#     cls = osp.splitext(filename)[0].split('_')[-1]
#     if not osp.isdir(osp.join(save_folder, f'cls_{cls}')):
#         os.mkdir(osp.join(save_folder, f'cls_{cls}'))
#     shutil.copyfile(img, osp.join(osp.join(save_folder, f'cls_{cls}'), filename))
    


        
    