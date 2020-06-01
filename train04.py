import argparse
import os
import time
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from config.config import Config
from src.dataset import *
from src.loss import *
from src.models import *
from src.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', type=str, required=True, help='this name directory created.')
args = parser.parse_args()
DHIRECTORY_NAME = args.name

config = Config()
LOAD_MODEL_EPOCH = config.load_model_epochs
DEBUG = config.debug
N_EPOCHS = config.n_epochs
LOAD_G_MODEL_SCORE = config.load_g_model_score
LOAD_D_MODEL_SCORE = config.load_d_model_score
MODEL_NO = config.model_no
BATCH_SIZE = config.batch_size
N_SPLIT = config.n_split
MAX_LR = config.max_lr
MIN_LR = config.min_lr
LAMBDA1 = config.lambda1
LAMBDA2 = config.lambda2
SEED = config.seed
DATALOADER_SEED = config.dataloader_seed
DEVICE = config.device
SIZE = config.size

img_dir = 'data/img_train'
model_g_dir = f'checkpoint/{DHIRECTORY_NAME}/G'
model_d_dir = f'checkpoint/{DHIRECTORY_NAME}/D'
log_dir = f'output/log/{DHIRECTORY_NAME}'
figure_dir = f'output/figure/{DHIRECTORY_NAME}'
pred_dir = f'output/pred/{DHIRECTORY_NAME}'


seed_everything(SEED)
kf = KFold(n_splits=N_SPLIT, shuffle=True, random_state=SEED)
img_ids = np.array(sorted(os.listdir(img_dir)))
tr_ix, va_ix = list(kf.split(img_ids, img_ids))[MODEL_NO]

train, valid = img_ids[tr_ix], img_ids[va_ix]

np.random.seed(DATALOADER_SEED)
train = np.random.permutation(train)
valid = np.random.permutation(valid)
np.random.seed(SEED)

if DEBUG:
    train = train[:32]
    valid = valid[:32]
    N_EPOCHS = 2

train_dataset = TrainDataset(train, img_dir, mode='train', size=SIZE, transform=augmentation(mode='train'))   # change mode
valid_dataset = TrainDataset(valid, img_dir, mode='valid', size=SIZE, transform=augmentation(mode='valid'))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
loaders_dict = {'train': train_loader, 'val': valid_loader}

for p in [model_g_dir, model_d_dir, log_dir, figure_dir, pred_dir]:
    path_conf(p)

if LOAD_MODEL_EPOCH:
    logs = df_to_log_pix2pix(os.path.join(log_dir, 'log.csv'))
else:
    logs = []

G = Generator()
D = Discriminator(norm='instance', pool_kernel_size=[4,2,2,2])
G_optimizer = optim.Adam(G.parameters(), lr=MAX_LR, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=MAX_LR, betas=(0.5, 0.999))

if LOAD_MODEL_EPOCH:
    print('G load: ', f'{model_g_dir}/model_psnr_{LOAD_G_MODEL_SCORE}_epoch{str(LOAD_MODEL_EPOCH).zfill(3)}.pth')
    checkpoint = torch.load(
        f'{model_g_dir}/model_psnr_{LOAD_G_MODEL_SCORE}_epoch{str(LOAD_MODEL_EPOCH).zfill(3)}.pth')
    G.load_state_dict(checkpoint['model'])
    G_optimizer.load_state_dict(checkpoint['optimizer'])
    for state in G_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(DEVICE)

    print('D load: ', f'{model_d_dir}/model_psnr_{LOAD_D_MODEL_SCORE}_epoch{str(LOAD_MODEL_EPOCH).zfill(3)}.pth')
    checkpoint = torch.load(
        f'{model_d_dir}/model_psnr_{LOAD_D_MODEL_SCORE}_epoch{str(LOAD_MODEL_EPOCH).zfill(3)}.pth', map_location=DEVICE)
    D.load_state_dict(checkpoint['model'])
    D_optimizer.load_state_dict(checkpoint['optimizer'])
    for state in D_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(DEVICE)  

G.to(DEVICE);  D.to(DEVICE)
criterion_hinge_g = HingeLoss('gen')
criterion_hinge_d_real = HingeLoss('dis_real')
criterion_hinge_d_fake = HingeLoss('dis_fake')
criterion_l1 = nn.L1Loss()


def main():
    torch.backends.cudnn.benchmark = True
    num_train_imgs = len(loaders_dict['train'].dataset)
    num_val_imgs = len(loaders_dict['val'].dataset)

    for epoch in range(N_EPOCHS):
        epoch += LOAD_MODEL_EPOCH
        start_time = time.time()
        epoch_train_G_adv_loss = 0
        epoch_train_G_l1_loss, epoch_val_G_l1_loss = 0, 0
        epoch_train_D_adv_loss = 0
        train_psnr, val_psnr = 0, 0

        print('-----------------------')
        print(f'Epoch {epoch+1} / {N_EPOCHS+LOAD_MODEL_EPOCH}')
        print('-----------------------\n')

        for phase in ['train', 'val']:
            if phase == 'train':
                G.train();  D.train()
                cnt = 0
                
                for real_rgb, _ in loaders_dict[phase]:
                    real_rgb = real_rgb.to(DEVICE)
                    real_ycrcb = preprocess_generator(real_rgb, device=DEVICE)
                    gray = real_ycrcb[:, :1, :, :]
                    real_crcb = real_ycrcb[:, 1:, :, :]

                    ones = torch.ones(real_rgb.size()[0], 1, 11, 11).to(DEVICE)
                    zeros = torch.zeros(real_rgb.size()[0], 1, 11, 11).to(DEVICE)

                    fake_crcb = G(gray)
                    fake_ycrcb = torch.cat([gray, fake_crcb], dim=1)
                    fake_rgb = deprocess_generator(fake_ycrcb, device=DEVICE)
                    fake_rgb_copy = fake_rgb.detach()

                    d_out_fake = D(fake_rgb)
                    # change criterion_bce->criterion_hinge
                    G_loss_adv = criterion_hinge_g(d_out_fake, ones)
                    G_loss_l1_crcb = criterion_l1(fake_crcb, real_crcb)
                    G_loss_l1_rgb = criterion_l1(fake_rgb, real_rgb)
                    G_loss_l1 = LAMBDA1*G_loss_l1_crcb + LAMBDA2*G_loss_l1_rgb
                    G_loss = G_loss_adv + G_loss_l1

                    G_optimizer.zero_grad();  D_optimizer.zero_grad()
                    G_loss.backward()
                    G_optimizer.step()
                    epoch_train_G_adv_loss += G_loss_adv.item()
                    epoch_train_G_l1_loss += G_loss_l1.item()

                    d_out_real = D(real_rgb)
                    d_out_fake2 = D(fake_rgb_copy)

                    # change criterion_bce->criterion_hinge
                    D_loss_real = criterion_hinge_d_real(d_out_real, ones)
                    D_loss_fake = criterion_hinge_d_fake(d_out_fake2, zeros)
                    D_loss = (D_loss_real + D_loss_fake) / 2

                    # Update weights once every two times
                    if cnt%2==0:
                        D_loss_tmp = D_loss
                        epoch_train_D_adv_loss += D_loss_tmp.item() * 2
                    elif cnt%2==1:
                        G_optimizer.zero_grad();  D_optimizer.zero_grad()
                        D_loss_tmp.backward()
                        D_loss.backward()
                        D_optimizer.step()
                        epoch_train_D_adv_loss += D_loss.item() * 2

                    for fake, real in zip(fake_rgb, real_rgb):
                        fake = fake.detach().cpu().numpy()
                        real = real.detach().cpu().numpy()
                        fake = np.where(fake>1.0, 1.0, fake)
                        real = np.where(real>1.0, 1.0, real)
                        fake *= 255.
                        real *= 255.
                        train_psnr += psnr(fake, real)
                    cnt += 1

            else:
                G.eval()

                for real_rgb, img_id in loaders_dict[phase]:
                    real_rgb = real_rgb.to(DEVICE)
                    real_ycrcb = preprocess_generator(real_rgb, device=DEVICE)
                    gray = real_ycrcb[:, :1, :, :]
                    real_crcb = real_ycrcb[:, 1:, :, :]
                    fake_crcb = G(gray)
                    fake_ycrcb = torch.cat([gray, fake_crcb], dim=1)
                    fake_rgb = deprocess_generator(fake_ycrcb, device=DEVICE)   

                    G_loss_l1_crcb = criterion_l1(fake_crcb, real_crcb)
                    G_loss_l1_rgb = criterion_l1(fake_rgb, real_rgb)
                    G_loss_l1 = LAMBDA1*G_loss_l1_crcb + LAMBDA2*G_loss_l1_rgb
                    epoch_val_G_l1_loss += G_loss_l1.item()

                    for fake, real, fname in zip(fake_rgb, real_rgb, img_id):
                        fake = fake.detach().cpu().numpy()
                        real = real.detach().cpu().numpy()
                        fake = np.where(fake>1.0, 1.0, fake)
                        real = np.where(real>1.0, 1.0, real)
                        fake = np.where(fake<0.0, 0.0, fake)
                        real = np.where(real<0.0, 0.0, real)
                        fake *= 255.
                        real *= 255.
                        val_psnr += psnr(fake, real)

                        if (epoch+1)%20==0 or epoch==0:
                            pred_epoch_dir = pred_dir + f'/{str(epoch+1).zfill(3)}'
                            path_conf(pred_epoch_dir)
                            fake = np.squeeze(fake).transpose((1, 2, 0)).astype(np.uint8)

                            img_real = cv2.imread(os.path.join(img_dir, fname+'.jpg'))
                            img_real, org_size = picture_resize(img_real, size=SIZE)
                            img_real = cv2.cvtColor(img_real, cv2.COLOR_BGR2RGB)

                            fake = set_aspect(fake, org_size)
                            img_real = set_aspect(img_real, org_size, size=SIZE)
                            
                            img = np.hstack([fake, img_real])
                            pilImg = Image.fromarray(np.uint8(img))
                            pilImg.save(os.path.join(pred_epoch_dir, fname+'.png'))

        train_G_adv_loss = epoch_train_G_adv_loss * BATCH_SIZE / num_train_imgs
        train_G_l1_loss = epoch_train_G_l1_loss * BATCH_SIZE / num_train_imgs
        train_D_adv_loss = epoch_train_D_adv_loss * BATCH_SIZE / num_train_imgs
        train_psnr /= num_train_imgs

        val_G_l1_loss = epoch_val_G_l1_loss * BATCH_SIZE / num_val_imgs
        val_psnr /= num_val_imgs

        end_time = time.time()
        print(f'epoch: {epoch+1}')
        print(f'time: {(end_time - start_time):.3f}sec.\n')

        print(f'train_D_adv_loss: {train_D_adv_loss:.5f}')
        print(f'train_G_adv_loss: {train_G_adv_loss:.5f}')
        print(f'train_G_l1_loss: {train_G_l1_loss:.5f}')
        print(f'val_G_l1_loss: {val_G_l1_loss:.5f}\n')

        print(f'train_psnr: {train_psnr:.5f}')
        print(f'val_psnr: {val_psnr:.5f}\n')

        for g in G_optimizer.param_groups:
            print(f"Model lr: {g['lr']:.9f}\n\n")

        log_epoch = {
            'epoch': epoch+1,
            'train_D_adv_loss': train_D_adv_loss,
            'train_G_adv_loss': train_G_adv_loss,
            'train_G_l1_loss': train_G_l1_loss,
            'val_G_l1_loss': val_G_l1_loss,
            'train_psnr': train_psnr,
            'val_psnr': val_psnr,
            }
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv(f'{log_dir}/log.csv', index=False)

        if (epoch+1)%20==0:
            state = {
                'epoch': epoch,
                'model': G.state_dict(),
                'optimizer': G_optimizer.state_dict(),
            }
            torch.save(state, f'{model_g_dir}/model_psnr_{val_psnr:.5f}_epoch{str(epoch+1).zfill(3)}.pth')

            state = {
                'epoch': epoch,
                'model': D.state_dict(),
                'optimizer': D_optimizer.state_dict(),
            }
            torch.save(state, f'{model_d_dir}/model_psnr_{val_psnr:.5f}_epoch{str(epoch+1).zfill(3)}.pth')

        df = pd.read_csv(f'{log_dir}/log.csv')
        plt.plot(df['train_D_adv_loss'], label='Discriminator Adversarial Loss', color='blue')
        plt.plot(df['train_G_adv_loss'], label='Generator Adversarial Loss', color='red')
        plt.legend(bbox_to_anchor=(0, -0.1), loc='upper left', borderaxespad=0)
        plt.savefig(f'{figure_dir}/adv_loss.png', bbox_inches='tight')
        plt.close()

        plt.plot(df['train_G_l1_loss'], label='train L1 loss', color='red')
        plt.plot(df['val_G_l1_loss'], label='val L1 loss', color='blue')
        plt.legend(bbox_to_anchor=(0, -0.1), loc='upper left', borderaxespad=0)
        plt.savefig(f'{figure_dir}/l1_loss.png', bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    main()