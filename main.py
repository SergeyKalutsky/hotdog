# 256 X 256
import os
import cv2
import shutil
from tqdm import tqdm


def rename_folder():
    basedir = 'hotdog/train/not-hotdog'
    files = os.listdir(basedir)
    for file in files:
        os.rename(f'{basedir}/{file}', f'{basedir}/not_hotdog_{file}')


basedir = 'hotdog/train_resized'
for file in tqdm(os.listdir(basedir)):
    if 'not_hotdog' in file:
        shutil.move(f'{basedir}/{file}', f'hotdog/banana/{file}')
    else:
        shutil.move(f'{basedir}/{file}', f'hotdog/hotdog/{file}')
    # img = cv2.imread(f'{basedir}/{file}', cv2.IMREAD_COLOR)
    # # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # resized = cv2.resize(img, (256, 256))
    # cv2.imwrite(f'hotdog/train_resized/{file}', resized)
