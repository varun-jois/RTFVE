import sys
import logging
import torch
import pathlib
import yaml
import numpy
import random
from torch.utils.data import DataLoader 
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from utils.utils import train_loop, valid_loop, model_save, model_load
from argparse import ArgumentParser
from models.loss import L1Loss, PerceptualLoss, GANLoss


# get the arguments
parser = ArgumentParser()
parser.add_argument('-c', '--cfg', required=True, help='The config yaml.')
args = parser.parse_args()

# load the config file
with open(args.cfg, 'r') as f:
    config = yaml.safe_load(f)
pth = config['paths']
mdl = config['model']
thp = config['train']


# set the seed for the RNG
seed = thp['seed']
g = torch.manual_seed(seed)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

# the device
device = torch.device(thp['device'])

# storing some params
checkpt_path = f"{pth['checkpoints']}/{config['name']}"
log = f'{checkpt_path}/{config["name"]}.log' 

# create the checkpoints folder and log file
pathlib.Path(checkpt_path).mkdir(parents=True, exist_ok=True)
pathlib.Path(log).touch(exist_ok=True)

# creating summary writer and logging file
file_handler = logging.FileHandler(filename=log)
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=handlers)
runs_path = f"runs/{config['name']}"
pathlib.Path(runs_path).mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(runs_path)

# store the config file in the log
logging.info(config)

# creating the dataset
if config['dataset'] == 'RefDataset':
    from data.RefDataset import RefDataset
    train_data = RefDataset(pth['train'], augment=True, use_refs=thp['use_refs'], 
                            blur=thp['blur'])
    train_dataloader = DataLoader(train_data, batch_size=thp['batch_size'], shuffle=True, 
                                worker_init_fn=seed_worker, generator=g)
    if thp['valid_epoch'] != 0:
        valid_data = RefDataset(pth['valid'], augment=False, use_refs=thp['use_refs'], 
                                blur=thp['blur'])
        valid_dataloader = DataLoader(valid_data, batch_size=thp['batch_size'], shuffle=True,
                                    worker_init_fn=seed_worker, generator=g)
else:
    raise ValueError('Invalid dataset selected in the config.yaml')


# creating the model
if config["model_name"] == 'RTFVE':
    from models.RTFVE import RTFVE as Net
else:
    raise ValueError('Invalid model name selected.')

# create the model
model = Net(**mdl).to(device)
logging.info(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logging.info(f'Total params: {total_params:,}')

# creating parameter groups
base_params = [p for n, p in model.named_parameters() if not n.startswith('module.feature_align') and p.requires_grad]
logging.info(f'Number of base params: {len(base_params)}')
offset_params = [p for n, p in model.named_parameters() if n.startswith('module.feature_align') and p.requires_grad]
logging.info(f'Number of offset params: {len(offset_params)}')


# create optimizer
optimizer = torch.optim.Adam([
    {'params': base_params},
    {'params': offset_params, 'lr': thp['offset_lr']}
], lr=thp['learning_rate'])

model_data = {'g_mdl': model, 'g_opt': optimizer}

# for discriminator
if thp['gan_weight'] > 0:
    logging.info('Creating GAN model')
    if config["discriminator_name"] == 'Shuffle_Discriminator':
        from models.discriminators import Shuffle_Discriminator
        dis = Shuffle_Discriminator().to(device)
    elif config["discriminator_name"] == 'Shuffle_Discriminator_Norm':
        from models.discriminators import Shuffle_Discriminator_Norm
        dis = Shuffle_Discriminator_Norm().to(device)
    elif config["discriminator_name"] == 'ResNet_Discriminator':
        from models.discriminators import ResNet_Discriminator
        dis = ResNet_Discriminator(64, 10).to(device)
    else:
        raise ValueError('Invalid discriminator name selected.')
    d_opt = torch.optim.Adam(dis.parameters(), lr=thp['learning_rate'])
    model_data['d_mdl'] = dis
    model_data['d_opt'] = d_opt


# creating a scheduler
if thp['scheduler']:
    scheduler = MultiStepLR(optimizer, thp['steps'], thp['gamma'])


# load pretrained model
if thp['epoch_start'] != 0:
    model_load(model_data, thp['epoch_start'], checkpt_path)
    # manually setting the lr for the groups if resuming training
    optimizer.param_groups[0]['lr'] = thp['learning_rate']
    optimizer.param_groups[1]['lr'] = thp['offset_lr']
    if thp['scheduler']:
        scheduler.last_epoch = thp['epoch_start']

# loss functions
losses = {'loss_rec': L1Loss(recreation_weight=thp['recreation_weight']).to(device)}
if thp['perceptual_weight'] > 0 or thp['style_weight'] > 0:
    logging.info('Adding perceptual loss')
    losses['loss_per'] = PerceptualLoss(thp['layer_weights'], perceptual_weight=thp['perceptual_weight'], style_weight=thp['style_weight']).to(device)
if thp['gan_weight'] > 0:
    logging.info('Adding gan loss')
    losses['loss_gan'] = GANLoss(config["gan_loss"], loss_weight=thp['gan_weight'])

# training loop
best_score = thp['save_thresh']
for epoch in range(1 + thp["epoch_start"], thp["epoch_start"] + thp["epochs"] + 1):
    #print(f"-------------------------------\nEpoch {epoch}")
    logging.info(f"-------------------------------\nEpoch {epoch}")
    
    # iterate over the data
    train_loss = train_loop(train_dataloader, model_data, losses, device)
    writer.add_scalars('loss/train', train_loss, epoch)

    # validation
    if (thp['valid_epoch'] != 0) and (epoch % thp["valid_epoch"] == 0):
        valid_loss = valid_loop(valid_dataloader, model_data, losses, device)
        writer.add_scalars('loss/valid', valid_loss, epoch)
        if thp['save_best'] and valid_loss['rec'] < best_score:
            model_save(model_data, epoch, checkpt_path, best=True)
            best_score = valid_loss['rec']

    # next step for scheduler
    if thp['scheduler']:
        scheduler.step()
        for n, group in enumerate(optimizer.param_groups):
            logging.info(f'lr for param group {n} is: {group["lr"]}')

    # save
    if epoch % thp["save_epoch"] == 0:
        model_save(model_data, epoch, checkpt_path)
