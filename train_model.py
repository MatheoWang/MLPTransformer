#======================================================================
# Script to train different models
#======================================================================
import os
import time
import numpy as np
import torch
import utils
import generators
import sys
from vxm_model import VxmDense
from mlp_model import PureMLP_subnet, MLPMixer_subnet, SwinTrans_subnet, PackNet
import losses
from utils import Early_stopping
from layers import SpatialTransformer
import pandas as pd
from metrics import *
from test_camus import test_camus

#=======================================================================================================================
# parse config file
print('Read config file')
config_path = sys.argv[1]
test_fold = int(sys.argv[2])
model_type = sys.argv[3]
num_scale = int(sys.argv[4])
args = utils.parse_config(config_path)
inshape = [args['img_size'], args['img_size']]
bidir = args['bidir']

#=======================================================================================================================
# load and prepare training data
print('Read train & valid files')
train_files, valid_files, _ = utils.get_train_valid_test_allfiles(test_fold, args['folder_prefix'], ch=args['ch'],
                                                                           generate_rand_pair = args['rand_pair'],
                                                                           delete_poor=False)
_, _, test_files = utils.get_train_valid_test_allfiles(test_fold, args['folder_prefix'], ch=args['ch'],
                                                                           generate_rand_pair = args['rand_pair'],
                                                                           delete_poor=True)
assert len(train_files) > 0, 'Could not find any training data.'
print('Set dataset')
transform_maps = {
    'zoom': generators.zoom,
    'data_aug': generators.data_aug
}
train_dataset = generators.CamusPairDataset(
    train_files, args['img_path'], imgsize=args['img_size'], transform=transform_maps[args['transform']])
valid_dataset = generators.CamusPairDataset(
    valid_files, args['img_path'], imgsize=args['img_size'], transform=transform_maps['zoom'])
test_dataset = generators.CamusPairDataset(
    test_files, args['img_path'], imgsize=args['img_size'], transform=transform_maps['zoom'])
#=======================================================================================================================

print('Set image shape')
# extract shape from sampled input
inshape = [args['img_size'], args['img_size']]

# prepare model folder
model_dir = args['model_dir'] + model_type + '_fold' + str(test_fold) + '_scale' + str(num_scale) + '_downsize' + str(args['int_downsize'])
os.makedirs(model_dir, exist_ok=True)
print(model_dir)
# device handling
gpus = args['gpu']
nb_gpus = len(gpus)
device = 'cuda'
torch.backends.cudnn.deterministic = args['cudnn_det']

print('Save args')
with open(os.path.join(model_dir, 'args.txt'), 'a') as f:
    for key in args:
        f.write(str(key) + ':' + str(args[key]) + '\n')

#=======================================================================================================================
print('Set model')
if model_type == 'vxm':
    enc_nf = [16, 32, 32, 32]
    dec_nf = [32, 32, 32, 32, 32, 16, 16]
    model = VxmDense(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        bidir=bidir,
        int_steps=args['int_steps'],
        int_downsize=args['int_downsize']
    )
elif num_scale > 1:
    model = PackNet(
        inshape=inshape,
        int_steps=args['int_steps'],
        patch_size=args['patch_size'],
        out_dim=args['out_dim'],
        dim=args['dim'],
        depth=args['depth'],
        forward_list=args['dim_forward'],
        num_heads=args['heads'],
        window_size=args['window_size'],
        int_downsize=args['int_downsize'],
        bidir=args['bidir'],
        learn_pos=True,
        subnet_type=model_type,
        num_scale=num_scale,
        int_first=args['int_first']
    )
else:
    if model_type =='PureMLP':
        modelname = PureMLP_subnet
    elif model_type =='MLPMixer':
        modelname = MLPMixer_subnet
    else:
        modelname = SwinTrans_subnet
    model = modelname(
            image_size=inshape[0],
            patch_size=args['patch_size'][0],
            out_dim=args['out_dim'][0],
            d_model=args['dim'][0],
            depth=args['depth'][0],
            dim_forward=args['dim_forward'][0],
            num_heads=args['heads'][0],
            window_size=args['window_size'][0],
            pool='cls',
            channels=1,
            bidir=args['bidir'],
            int_steps=args['int_steps'],
            int_downsize=args['int_downsize'],
            learn_pos=True,
            out_dis=args['single_scale'],
            int_first=args['int_first']
            )

#=======================================================================================================================

# check model parameters 
#pytorch_total_params = sum(p.numel() for p in model.parameters())
#pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)


if nb_gpus > 1:
    # use multiple GPUs via DataParallel
    model = torch.nn.DataParallel(model)
    model.save = model.module.save
# prepare the model for training and send to device
model.to(device)
# set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
earlystopper = Early_stopping(patience=30)
# prepare image loss
if args['image_loss'] == 'ncc':
    image_loss_func = losses.NCC().loss
elif args['image_loss'] == 'mse':
    image_loss_func = losses.MSE().loss
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args['image_loss'])
# need two image loss functions if bidirectional
if bidir:
    losses_list = [image_loss_func, image_loss_func]#, losses.Dice().loss, losses.Dice().loss]
    weights = [0.5, 0.5]#, 0.5, 0.5]
else:
    losses_list = [image_loss_func]#, losses.Dice().loss]
    weights = [1]#, 1]
losses_list += [losses.Grad2g()]
weights += [args['deform_lambda']]


#=======================================================================================================================

print('Begin training')
# training loops
best_loss = False
for epoch in range(args['initial_epoch'], args['epochs']+1):
    # =====================================================================================================
    # train models
    # =====================================================================================================
    #print('Training result')
    model.train()
    epoch_loss = []
    epoch_total_loss = []
    epoch_step_time = []

    for i, sample in enumerate(train_dataset, 1):

        step_start_time = time.time()
        # generate inputs (and true outputs) and convert them to tensors
        _, _, inputs, mask_es_gt, y_true, mask_ed_gt, sampling = sample

        inputs = [torch.from_numpy(d).to(device).float() for d in inputs]
        y_true = [torch.from_numpy(d).to(device).float() for d in y_true]

        # run inputs through the model to produce a warped image and flow field
        y_pred = model(*inputs)

        # calculate total loss
        loss = 0
        loss_list = []
        total_loss_num = len(losses_list)
        for n, loss_function in enumerate(losses_list):
            curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
            loss_list.append(curr_loss.item())
            loss += curr_loss

        epoch_loss.append(loss_list)
        epoch_total_loss.append(loss.item())

        # backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # get compute time
        epoch_step_time.append(time.time() - step_start_time)

    # print epoch info
    epoch_info = 'Training Epoch %d/%d' % (epoch, args['epochs'])
    time_info = '%.4f sec/step' % np.mean(epoch_step_time)
    losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
    loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
    #print(' - '.join((epoch_info, time_info, loss_info)), flush=True)
    with open(os.path.join(model_dir, 'training_info.csv'), 'a') as f:
        f.write(str(epoch) + ',' + losses_info + ',' + str(np.mean(epoch_total_loss)) + '\n')

    # =====================================================================================================
    # evaluate models
    # =====================================================================================================
    #print('Validation result')
    with torch.no_grad():
        model.eval()

        epoch_loss = []
        epoch_total_loss = []
        epoch_step_time = []

        for i, sample in enumerate(valid_dataset, 1):

            step_start_time = time.time()
            # generate inputs (and true outputs) and convert them to tensors
            _, _, inputs, mask_es_gt, y_true, mask_ed_gt, sampling = sample
            inputs = [torch.from_numpy(d).to(device).float() for d in inputs]
            y_true = [torch.from_numpy(d).to(device).float() for d in y_true]

            # run inputs through the model to produce a warped image and flow field
            y_pred = model(*inputs)

            # calculate total loss
            loss = 0
            loss_list = []
            total_loss_num = len(losses_list)
            for n, loss_function in enumerate(losses_list):
                curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
                loss_list.append(curr_loss.item())
                loss += curr_loss

            epoch_loss.append(loss_list)
            epoch_total_loss.append(loss.item())

            # get compute time
            epoch_step_time.append(time.time() - step_start_time)

        # print epoch info
        epoch_info = 'Validation Epoch %d/%d' % (epoch, args['epochs'])
        time_info = '%.4f sec/step' % np.mean(epoch_step_time)
        losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
        loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
        #print(' - '.join((epoch_info, time_info, loss_info)), flush=True)
        with open(os.path.join(model_dir, 'validation_info.csv'), 'a') as f:
            f.write(str(epoch) + ',' + losses_info + ',' + str(np.mean(epoch_total_loss)) + '\n')
        best_loss = earlystopper(np.mean(epoch_total_loss), epoch)

        # save model checkpoint
        if epoch % 5 == 0 or best_loss:
            model.save(os.path.join(model_dir, '%04d.pt' % (epoch)))

        if earlystopper.early_stop:
            break
# final model save
model.save(os.path.join(model_dir, '%04d.pt' % epoch))

#=======================================================================================================================
best_epoch = earlystopper.best_epoch
model_file = os.path.join(model_dir, str(best_epoch).zfill(4) + '.pt')
model = model.load(model_file, device)
model.to(device)
transformer = SpatialTransformer(inshape)
transformer.to(device)
test_camus(model, test_dataset, device, transformer, model_dir, best_epoch)

