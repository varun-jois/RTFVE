
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import save_image
import torch.nn as nn
import logging
from datetime import datetime
from glob import glob
from collections import defaultdict
import ast

def cfg_from_log(log):
    with open(log, 'r') as f:
        first_line = f.readline()
    i = first_line.find('{')
    cfg = ast.literal_eval(first_line[i:])
    return cfg

#####################################################
# main train and validation loops
def train_loop(dataloader, model_data, losses, device):
    """
    The main training loop
    """
    # epoch_loss = {k: 0 for k in ['total_g', 'rec', 'gan', 'dis', 'per', 'sty', 
    #                              'dis_r', 'dis_f', 'out_r', 'out_f']}
    epoch_loss = defaultdict(int)
    g_mdl, g_opt = model_data['g_mdl'], model_data['g_opt']
    g_mdl.train()
    # train the generator
    if 'd_mdl' in model_data:
        d_mdl, d_opt = model_data['d_mdl'], model_data['d_opt']
        d_mdl.train()
        for p in d_mdl.parameters():
            p.requires_grad = False
    
    for batch, data in enumerate(dataloader):
        g_opt.zero_grad()
        # load onto device
        hq, lq, refs = data
        hq, lq = hq.to(device), lq.to(device)
        if refs:
            for i in range(len(refs)):
                refs[i] = refs[i].to(device)
            pred = g_mdl(lq, refs)
        else:
            pred = g_mdl(lq)
        
        l_g_total = 0
        # Compute prediction loss
        if 'loss_rec' in losses:
            loss_rec = losses['loss_rec'](pred, hq)
            l_g_total += loss_rec
            epoch_loss['rec'] += loss_rec.item()
        if 'loss_per' in losses:
            loss_per, loss_sty = losses['loss_per'](pred, hq)
            if loss_per is not None:
                l_g_total += loss_per
                epoch_loss['per'] += loss_per.item()
            if loss_sty is not None:
                l_g_total += loss_sty
                epoch_loss['sty'] += loss_sty.item()
        if 'loss_gan' in losses:
            fake_g_pred = d_mdl(pred)
            loss_gan = losses['loss_gan'](fake_g_pred, target_is_real=True, is_disc=False)
            l_g_total += loss_gan
            epoch_loss['gan'] += loss_gan.item()
        # Backpropagation
        l_g_total.backward()
        g_opt.step()
        epoch_loss['total_g'] += l_g_total.item()

        # train the discriminator
        if 'd_mdl' in model_data:
            for p in d_mdl.parameters():
                p.requires_grad = True
            d_opt.zero_grad()
            # real data
            real_d_pred = d_mdl(hq)
            l_d_real = losses['loss_gan'](real_d_pred, True, is_disc=True)
            epoch_loss['dis_r'] += l_d_real.item()
            epoch_loss['out_r'] += torch.mean(real_d_pred.detach())
            l_d_real.backward()
            # fake data
            fake_d_pred = d_mdl(pred.detach())
            l_d_fake = losses['loss_gan'](fake_d_pred, False, is_disc=True)
            epoch_loss['dis_f'] += l_d_fake.item()
            epoch_loss['out_f'] += torch.mean(fake_d_pred.detach())
            l_d_fake.backward()
            d_opt.step()
            epoch_loss['dis'] += l_d_real.item() + l_d_fake.item()

        # # store epoch losses
        # epoch_loss['loss_tot'] += loss.item()
        # epoch_loss['loss_rec'] += loss_rec.item()
        # epoch_loss['loss_per'] += loss_per.item()
        # epoch_loss['loss_sty'] += loss_sty.item()
    # # get final epoch losses
    # epoch_loss['loss_tot'] /= (batch + 1)
    # epoch_loss['loss_rec'] /= (batch + 1)
    # epoch_loss['loss_per'] /= (batch + 1)
    # epoch_loss['loss_sty'] /= (batch + 1)

    # get the average loss
    epoch_loss = {k: v / (batch + 1) for k, v in epoch_loss.items()}
    # get the time
    now = datetime.now()
    logging.info(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}]")
    for k, v in epoch_loss.items():
        logging.info(f"Train loss: {k} - {v:.5f}")
    return epoch_loss
    


def valid_loop(dataloader, model_data, losses, device):
    """
    The main validation loop
    """
    # epoch_loss = {k: 0 for k in ['total_g', 'rec', 'gan', 'dis', 'per', 'sty', 
    #                              'dis_r', 'dis_f', 'out_r', 'out_f']}
    epoch_loss = defaultdict(int)
    g_mdl = model_data['g_mdl']
    g_mdl.eval()
    if 'd_mdl' in model_data:
        d_mdl = model_data['d_mdl']
        d_mdl.eval()
    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            # load onto device
            hq, lq, refs = data
            hq, lq = hq.to(device), lq.to(device)
            if refs:
                for i in range(len(refs)):
                    refs[i] = refs[i].to(device)
                pred = g_mdl(lq, refs)
            else:
                pred = g_mdl(lq)

            l_g_total = 0
            # Compute prediction loss
            if 'loss_rec' in losses:
                loss_rec = losses['loss_rec'](pred, hq)
                l_g_total += loss_rec
                epoch_loss['rec'] += loss_rec.item()
            if 'loss_per' in losses:
                loss_per, loss_sty = losses['loss_per'](pred, hq)
                if loss_per is not None:
                    l_g_total += loss_per
                    epoch_loss['per'] += loss_per.item()
                if loss_sty is not None:
                    l_g_total += loss_sty
                    epoch_loss['sty'] += loss_sty.item()
            if 'loss_gan' in losses:
                fake_g_pred = d_mdl(pred)
                loss_gan = losses['loss_gan'](fake_g_pred, target_is_real=True, is_disc=False)
                l_g_total += loss_gan
                epoch_loss['gan'] += loss_gan.item()
            # Backpropagation
            epoch_loss['total_g'] += l_g_total.item()

            # discriminator loss
            if 'd_mdl' in model_data:
                # real data
                real_d_pred = d_mdl(hq)
                l_d_real = losses['loss_gan'](real_d_pred, True, is_disc=True)
                epoch_loss['dis_r'] += l_d_real.item()
                epoch_loss['out_r'] += torch.mean(real_d_pred.detach())
                # fake data
                fake_d_pred = d_mdl(pred.detach())
                l_d_fake = losses['loss_gan'](fake_d_pred, False, is_disc=True)
                epoch_loss['dis_f'] += l_d_fake.item()
                epoch_loss['out_f'] += torch.mean(fake_d_pred.detach())
                epoch_loss['dis'] += l_d_real.item() + l_d_fake.item()
            
        #     # store epoch losses
        #     epoch_loss['loss_tot'] += loss.item()
        #     epoch_loss['loss_rec'] += loss_rec.item()
        #     epoch_loss['loss_per'] += loss_per.item()
        #     epoch_loss['loss_sty'] += loss_sty.item()
        # # get final epoch losses
        # epoch_loss['loss_tot'] /= (batch + 1)
        # epoch_loss['loss_rec'] /= (batch + 1)
        # epoch_loss['loss_per'] /= (batch + 1)
        # epoch_loss['loss_sty'] /= (batch + 1)
        
        # get the average loss
        epoch_loss = {k: v / (batch + 1) for k, v in epoch_loss.items()}
        # get the time
        now = datetime.now()
        logging.info(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}]")
        for k, v in epoch_loss.items():
            logging.info(f"Valid loss: {k} - {v:.5f}")
    return epoch_loss


#####################################################
# model saving and loading
def model_save(model_data, epoch, checkpt_path, best=False):
    for name, model in model_data.items():
         #print(f'Saving {checkpt_path}/{epoch}_{name}.pth')
         if best:
             logging.info(f'Saving {checkpt_path}/best_{name}.pth')
             torch.save(model.state_dict(), f'{checkpt_path}/best_{name}.pth')
         else:
            logging.info(f'Saving {checkpt_path}/{epoch}_{name}.pth')
            torch.save(model.state_dict(), f'{checkpt_path}/{epoch}_{name}.pth')


def model_load(model_data, epoch_start, checkpt_path):
    for name, model in model_data.items():
        #print(f'Loading weights {checkpt_path}/{epoch_start}_{name}.pth')
        if glob(f'{checkpt_path}/{epoch_start}_{name}.pth'):
            logging.info(f'Loading weights {checkpt_path}/{epoch_start}_{name}.pth')
            model.load_state_dict(torch.load(f'{checkpt_path}/{epoch_start}_{name}.pth'))
    


#####################################################
# extras

def performance_loop(dataloader, model, losses, device, metrics, img_path=None):
    """
    The main validation loop
    """
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            # load onto device
            hq, lq, refs = data
            hq, lq = hq.to(device), lq.to(device)
            for i in range(len(refs)):
                refs[i] = refs[i].to(device)
            
            # Compute prediction and loss
            pred = model(lq, refs)
            print(f'Pred range: {pred.max() - pred.min()}')
            print(f'hq range: {hq.max() - hq.min()}')
            loss = losses['loss_rec'](pred, hq)
            valid_loss += loss.item()

            # save two sets of images
            if img_path is not None and batch == 0:
                output_size = hq.shape[2]
                bic = F.interpolate(lq, (output_size, output_size), mode='bicubic', align_corners=False)
                for i in range(2):
                    save_image(bic[i], f"{img_path}/{i}_bi.png")
                    save_image(pred[i], f"{img_path}/{i}_sr.png")
                    save_image(hq[i], f"{img_path}/{i}_hq.png")
                    save_image(lq[i], f"{img_path}/{i}_lq.png")

            # unnormalize data to [0, 255] range. Taken from save_image in torchvision
            pred = pred.mul(255).add_(0.5).clamp_(0, 255).trunc()
            hq = hq.mul(255).add_(0.5).clamp_(0, 255).trunc()

            # compute metrics
            if metrics is not None:
                for m in metrics:
                    if m == 'lpips':  # need to bring to 0 to 1 range
                        _ = metrics[m](pred.div(255), hq.div(255))
                    else:
                        _ = metrics[m](pred, hq)
            
    print(f'loss: {valid_loss / (batch + 1)}')
    if metrics is not None:
        # final metric computation
        for m in metrics:
            score = metrics[m].compute()
            print(f'{m}: {score:.4f}')
