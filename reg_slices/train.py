import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from src.datasets import Slice3DDataset
from src.models import Slices3DRegModel
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import logging
import shutil
import time
import glob
import cv2
import numpy as np
from src.utils import RGB2BGR, tensor2numpy, denorm
from options import get_parser
logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)


def cal_acc(x, gt, pred_type):
    if pred_type == 'occ':
        acc = ((x['occ_pred'].sigmoid() > 0.5) == (gt['occ'] > 0.5)).float().sum(dim=-1) / x['occ_pred'].shape[1]
    else:
        acc = ((x['sdf_pred']>=0) == (gt['sdf'] >=0)).float().sum(dim=-1) / x['sdf_pred'].shape[1]
    acc = acc.mean(-1)
    return acc

def cal_loss_pred(x, gt, pred_type):

    if pred_type == 'occ':
        loss_pred = F.binary_cross_entropy_with_logits(x['occ_pred'], gt['occ'])
    else:
        loss_pred = F.l1_loss(x['sdf_pred'], gt['sdf'])

    loss_img_rec = F.l1_loss(x['slices_rec'], gt['img_slices']) 
    loss_img_rec_vgg = x['vgg_loss']

    return loss_pred, loss_img_rec, loss_img_rec_vgg

def train_step(batch, model, opt, args):
    for key in batch: batch[key] = batch[key].cuda()
    opt.zero_grad()
    x = model(batch)

    loss_pred, loss_img, loss_img_vgg = cal_loss_pred(x, batch, args.pred_type)
    loss = loss_pred + loss_img + loss_img_vgg

    loss.backward()
    opt.step()
    with torch.no_grad():
        acc = cal_acc(x, batch, args.pred_type)
    return loss_pred.item(), loss_img.item(), loss_img_vgg.item(), acc.item()

def concat_imgs(img_input, img_slices_rec, dir_ckpt, n_iter):
    res_img = np.zeros((128 * 13, 0, 3))
    for idx in range(4):
        img_list = []
        img_list.append(RGB2BGR(tensor2numpy(denorm(img_input[idx])))),
        for slices_idx in range(12):
            img_list.append(RGB2BGR(tensor2numpy(denorm(img_slices_rec[idx, 3*slices_idx:3*(slices_idx+1), ...]))))
        
        
        img_list_ = np.concatenate(img_list, 0)

        res_img = np.concatenate((res_img, img_list_), 1)


    cv2.imwrite(os.path.join(dir_ckpt, f'rec_{str(n_iter)}_slice.png'), res_img * 255.0)

@torch.no_grad()
def val_step(model, val_loader, pred_type, dir_ckpt, n_iter):
    avg_loss_pred = 0
    avg_acc  = 0
    ni = 0
    for batch_id, batch in enumerate(val_loader):

        for key in batch: batch[key] = batch[key].cuda()
        x = model(batch)

        if batch_id == 0:
            concat_imgs(batch['img_input'][0:4], x['slices_rec'][0:4], dir_ckpt, n_iter)
        
        loss_pred, loss_img, loss_img_vgg = cal_loss_pred(x, batch, pred_type)

        acc = cal_acc(x, batch, pred_type)

        avg_loss_pred = avg_loss_pred + loss_pred.item()
        avg_acc  = avg_acc  + acc.item()
        ni += 1
    avg_loss_pred /= ni
    avg_acc /= ni
    return avg_loss_pred, avg_acc, loss_img

def backup_code(name_exp):
    os.makedirs(os.path.join('experiments', name_exp, 'code'), exist_ok=True)
    shutil.copy('src/models.py', os.path.join('experiments', name_exp, 'code', 'models.py') )
    shutil.copy('src/datasets.py', os.path.join('experiments', name_exp, 'code', 'datasets.py'))
    shutil.copy('src/unet_custom.py', os.path.join('experiments', name_exp, 'code', 'unet_custom.py'))
    shutil.copy('src/vgg16bn_feats.py', os.path.join('experiments', name_exp, 'code', 'vgg16bn_feats.py'))
    shutil.copy('src/layers.py', os.path.join('experiments', name_exp, 'code', 'layers.py'))
    shutil.copy('./train.py', os.path.join('experiments', name_exp, 'code', 'train.py'))
    shutil.copy('./options.py', os.path.join('experiments', name_exp, 'code', 'options.py'))

def train(args):

    name_exp = args.name_exp
    name_exp_stamp = name_exp
    os.makedirs(os.path.join('experiments', name_exp_stamp), exist_ok=True)
    backup_code(name_exp_stamp)

    # Dump options
    with open(os.path.join('experiments', name_exp_stamp, "opts.txt"), "w") as f:
        for key, value in vars(args).items():
            f.write(str(key) + ": " + str(value) + "\n")

    dir_ckpt = os.path.join('experiments', name_exp_stamp, 'ckpt')
    os.makedirs(dir_ckpt, exist_ok=True)

    writer = SummaryWriter(os.path.join('experiments', name_exp_stamp, 'log'))

    if args.name_dataset in ['abc','shapenet']:
        train_loader = DataLoader(Slice3DDataset(split='train', args=args), shuffle=True, batch_size=args.n_bs, num_workers=args.n_wk, drop_last=True)
        val_loader = DataLoader(Slice3DDataset(split='val', args=args), shuffle=False, batch_size=args.n_bs, num_workers=args.n_wk, drop_last=True)
    else:
        train_loader = DataLoader(Slice3DDataset(split='train', args=args), shuffle=True, batch_size=args.n_bs, num_workers=args.n_wk, drop_last=True)
        val_loader = DataLoader(Slice3DDataset(split='val', args=args), shuffle=False, batch_size=args.n_bs, num_workers=args.n_wk, drop_last=True)

    model = Slices3DRegModel(img_size=args.img_size, n_slices=args.n_slices, mode=args.mode)

    if args.multi_gpu:
        model = torch.nn.DataParallel(model)

    model.cuda()

    opt = optim.Adam(model.parameters(), lr=args.lr)

    if args.resume:
        fnames_ckpt = glob.glob(os.path.join(dir_ckpt, '*'))
        fname_ckpt_latest = max(fnames_ckpt, key=os.path.getctime)
        ckpt = torch.load(fname_ckpt_latest)
        if args.multi_gpu:
            model.module.load_state_dict(ckpt['model'])
        else:
            model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['opt'])
        epoch_latest = ckpt['n_epoch'] + 1
        n_iter = ckpt['n_iter']
        n_epoch = epoch_latest
    else:
        epoch_latest = 0
        n_iter = 0
        n_epoch = 0
    
    for i in range(epoch_latest, args.n_epochs):
        model.train()

        for batch in train_loader:
            loss_pred, loss_img, loss_img_vgg, acc = train_step(batch, model, opt, args)
            if n_iter % args.freq_log == 0:
                print('[train] epcho:', n_epoch, ' ,iter:', n_iter," loss_pred:", loss_pred, " loss_img:", loss_img, " loss_img_vgg:", loss_img_vgg, " acc:", acc)
                writer.add_scalar('Loss/train', loss_pred, n_iter)
                writer.add_scalar('Acc/train', acc, n_iter)
                
            n_iter += 1

        if n_epoch % args.freq_ckpt == 0:
            model.eval()
            avg_loss_pred, avg_acc, avg_loss_img = val_step(model, val_loader, args.pred_type, dir_ckpt, n_iter)
            writer.add_scalar('Loss/val', avg_loss_pred, n_iter)
            writer.add_scalar('Acc/val', avg_acc, n_iter)
            print('[val] epcho:', n_epoch,' ,iter:',n_iter," avg_loss_pred:",avg_loss_pred, " acc:",avg_acc)
            if args.multi_gpu:
                torch.save({'model':model.module.state_dict(), 'opt':opt.state_dict(), 'n_epoch':n_epoch, 'n_iter':n_iter}, f'{dir_ckpt}/{n_epoch}_{n_iter}_{avg_loss_pred:.4}_{avg_acc:.4}_{avg_loss_img:.4}.ckpt')
            else:
                torch.save({'model':model.state_dict(), 'opt':opt.state_dict(), 'n_epoch':n_epoch, 'n_iter':n_iter}, f'{dir_ckpt}/{n_epoch}_{n_iter}_{avg_loss_pred:.4}_{avg_acc:.4}_{avg_loss_img:.4}.ckpt')
        

        if n_epoch > 0 and n_epoch % args.freq_decay == 0:
            for g in opt.param_groups:
                g['lr'] = g['lr'] * args.weight_decay
        
        n_epoch += 1


def main():
    args = get_parser().parse_args()
    if args.mode == 'train':
        train(args)
    else:
        test(args)

main()

        
