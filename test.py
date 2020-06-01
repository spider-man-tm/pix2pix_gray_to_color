import argparse
import os
from PIL import Image

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config.config import TestConfig
from src.dataset import *
from src.models import *
from src.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', type=str, required=True, help='this name directory created.')
args = parser.parse_args()
DHIRECTORY_NAME = args.name

config = TestConfig()
LOAD_MODEL_EPOCH = config.load_model_epochs
DEBUG = config.debug
LOAD_G_MODEL_SCORE = config.load_g_model_score
BATCH_SIZE = config.batch_size
SEED = config.seed
DEVICE = config.device
SIZE = config.size

img_dir = 'data/img_test'
model_g_dir = f'checkpoint/{DHIRECTORY_NAME}/G'
pred_dir = f'output/test_pred/{DHIRECTORY_NAME}'


seed_everything(SEED)
test = np.array(sorted(os.listdir(img_dir)))

if DEBUG:
    test = test[:min(32, len(test))]

test_dataset = TestDataset(test, img_dir, size=SIZE, transform=augmentation(mode='test'))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

path_conf(pred_dir)

G = Generator()

print('G load: ', f'{model_g_dir}/model_psnr_{LOAD_G_MODEL_SCORE}_epoch{str(LOAD_MODEL_EPOCH).zfill(3)}.pth')
checkpoint = torch.load(
    f'{model_g_dir}/model_psnr_{LOAD_G_MODEL_SCORE}_epoch{str(LOAD_MODEL_EPOCH).zfill(3)}.pth', map_location=DEVICE)
G.load_state_dict(checkpoint['model'])

G.to(DEVICE)


def main():
    torch.backends.cudnn.benchmark = True
    num_test_imgs = len(test_loader.dataset)

    print('-----------------------')
    print('Test start.')
    print('-----------------------\n')

    G.eval()
    for gray, img_id in test_loader:
        gray = gray.to(DEVICE)
        fake_crcb = G(gray)
        fake_ycrcb = torch.cat([gray, fake_crcb], dim=1)
        fake_rgb = deprocess_generator(fake_ycrcb, device=DEVICE)
        
        for fake, fname in zip(fake_rgb, img_id):
            fake = fake.detach().cpu().numpy()
            fake = np.where(fake>1.0, 1.0, fake)
            fake = np.where(fake<0.0, 0.0, fake)
            fake *= 255.
            fake = np.squeeze(fake).transpose((1, 2, 0)).astype(np.uint8)
            img_org = cv2.imread(os.path.join(img_dir, fname+'.jpg'))
            img_org, org_size = picture_resize(img_org, size=SIZE)

            fake = set_aspect(fake, org_size, size=SIZE)
            img_org = set_aspect(img_org, org_size, size=SIZE)

            img = np.hstack([fake, img_org])
            pilImg = Image.fromarray(np.uint8(img))
            pilImg.save(os.path.join(pred_dir, fname+'.png'))

    print(f'Total {num_test_imgs} images done.')


if __name__ == "__main__":
    main()