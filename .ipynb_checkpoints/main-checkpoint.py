# coding: utf-8
import os
import time
import random
import logging
import argparse
import numpy as np
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate

import hp
from model import *
from dataset import *
from lr_scheduler import *
from utils import check_loss, convert_to_strings,check_sequences
from dtw import dtw_batch
from beam_search import ctc_beam_search

SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

def pad_zeros(x):
    """
    Pads all the frame embedding sequences with zero to make length
    of all sequences equal to length of the longest sequence
    of the batch.
    
    Arguments:
        list: list of numpy array of embedding sequences       
    Returns:
        numpy array: padded sequences (n_batch,max_len,emb_size) 
    """
    
    n_batch = len(x)
    max_len = max(seq.shape[0] for seq in x)
    padded = np.zeros((n_batch, max_len, x[0].shape[1]))
    for i in range(n_batch):
        padded[i, :x[i].shape[0]] = x[i]
    return padded

def showLR(optimizer):
    """
    Returns: A list of learning rate grouped by parameters.
    """
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr

def warpctc_collate(batch):
    """
    custom collate function for 0 padding.
    
    Returns:
        x(tensor): padded sequences (n_batch,max_len,emb_size).
        xlen(tensor): original length of the sequences.
        label(tensor): target cluster sequence indexes.
        ylen(tensor): length of target sequences.
    """
    
    x, xl, y, yl = zip(*batch)
    xlen = torch.LongTensor(xl)
    ylen = torch.LongTensor(yl)
    x = pad_zeros(x)
    x = default_collate(x)
    ys = []
    for i in y: ys += i
    label = torch.LongTensor(ys)
    
    return [x, xlen, label, ylen]

def data_loader(args):
    """
    Returns: A dictionary contained train, validation and test dataloaders. 
    """
    
    dsets = {x: MyDataset(hp.dataset_split[x],phase = x) for x in ['train','val','test']}
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=args.batch_size, shuffle=True,collate_fn=warpctc_collate ,num_workers=args.workers) for x in ['train','val','test']}
    dset_sizes = {x: len(dsets[x]) for x in ['train','val','test']}
    print('\nStatistics: train: {}'.format(dset_sizes['train']))
    print('\nStatistics: val: {}'.format(dset_sizes['val']))
    print('\nStatistics: test: {}'.format(dset_sizes['test']))
    
    return dset_loaders

def reload_model(model, logger, path=""):
    """
    Returns a pretrained model if model path is not empty.
    """
    
    if not bool(path):
        logger.info('train from scratch')
        
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info('*** model has been successfully loaded for TESTING! ***')
        
        return model

def train_test(model, dset_loaders, criterion, epoch, phase, optimizer, args, logger, use_gpu, save_path):
    """
    Perform training, validation or testing for one epoch.
    
    Arguments:
        model: Model instance.
        dset_loaders: Dataset loader dictionary of train, val and test dataloaders.
        criterion: Criterion instance (pytorch ctc loss).
        epoch(int): Current epoch
        optimzer: Optimzer instance.
        args: Command line arguments.
        logger: Logger instance.
        use_gpu: Flag show if gpu can be used when set true.
        save_path: Path for saving model.  
    Returns:
        float: Accuracy. 
    """
    
    print ("######################{}###################################".format(phase))
    
    if phase == 'test' or phase =='val':
        model.eval()
    elif phase == 'train':
        model.train()

    running_loss, running_all, running_dtw, running_acc = 0., 0., 0., 0. 

    for batch_idx, (inputs, ilen, targets, tlen) in enumerate(dset_loaders[phase]):
        if use_gpu:
            inputs, targets = inputs.cuda(), targets.cuda()
        print ("max sec: {}".format(inputs.shape[1]))
        outputs, softmax_output = model(inputs.float(), ilen)
        
        #loss calculation and checking numerical stability.
        loss = criterion(outputs, targets, ilen, tlen)
        loss_value = loss.item()
        valid_loss, error = check_loss(loss, loss_value)
        if valid_loss and phase=='train':
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            optimizer.step()
        elif not valid_loss and phase=='train':
            print(error)
            loss_value = 0

        targets = targets.cpu().numpy()
        split_targ = []
        idx = 0
        for i in tlen.numpy():
            split_targ.append(targets[idx:idx+i])
            idx += i
        target_seq = convert_to_strings(split_targ)
        decoded_seq = ctc_beam_search(softmax_output.cpu().detach().numpy(), ilen.cpu().numpy(),target_seq)
        if hp.constrained_beam_search:
            assert check_sequences(decoded_seq)==True, "decoded string not in lexicon"
        avg_dtw = dtw_batch(decoded_seq,target_seq)
                    
        #print results for 1st sample in batch
        print('Original: ', target_seq[0])
        print('Decoded: ', decoded_seq[0])
        
        running_acc += sum(p==t for p,t in zip(decoded_seq,target_seq))
        running_loss += loss_value * inputs.size(0)
        running_dtw += avg_dtw * inputs.size(0)
        running_all += int(inputs.shape[0])

        if batch_idx == 0:
            since = time.time()
        elif batch_idx % hp.display_interval == 0:
            print('Process: [{:5.0f}/{:5.0f} ({:.0f}%)]\tLoss: {:.4f}\tDTW: {:.4f} \tACC: {:.4f} \tCost time:{:5.0f}s\tEstimated time:{:5.0f}s\r'.format(
                running_all,
                len(dset_loaders[phase].dataset),
                100. * batch_idx / (len(dset_loaders[phase])-1),
                running_loss / running_all,
                running_dtw / running_all,
                running_acc / running_all,
                time.time()-since,
                (time.time()-since)*(len(dset_loaders[phase])-1) / batch_idx - (time.time()-since)))
    print()
    assert running_all == len(dset_loaders[phase].dataset)
    logger.info('{} Epoch:\t{:2}\tLoss: {:.4f}\tDTW: {:.4f}\tACC {:.4f}\n'.format(
        phase,
        epoch,
        running_loss / running_all,
        running_dtw / running_all,
        running_acc / running_all
        ))

    return model, running_acc / running_all

def main(args, use_gpu):

    save_path = hp.model_base_path
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    #logging
    if args.test:
        filename = save_path +'/' + args.desc + '/test_logs.txt'
    else:
        filename = save_path + '/' + args.desc + '/logs.txt'
    
    logger_name = "mylog"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filename, mode='a')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    
    #model loading
    model = get_model()
    if use_gpu:
        model = model.cuda()
    #default reduction for loss is mean over batch.
    criterion = nn.CTCLoss(zero_infinity = True)
    if use_gpu:
        criterion = criterion.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, nesterov=True, momentum = args.momentum)
    scheduler = AdjustLR(optimizer, [args.lr], sleep_epochs=hp.sleep_epochs, half=hp.half, verbose=1)
    
    dset_loaders = data_loader(args)
    
    #only test
    if args.test:
        assert args.model_path is not None and len(args.model_path)>0, "provide correct model path."
        model = reload_model(model, logger, args.model_path)
        with torch.no_grad():
            train_test(model, dset_loaders, criterion, epoch, 'test', optimizer, args, logger, use_gpu, save_path)
        return
        
    #train, val and test
    best_acc, best_epoch = 0, -1
    for epoch in range(1,args.epochs+1):
        if not args.no_scheduler:
            scheduler.step(epoch)
        logger.info('-' * 10)
        logger.info('Epoch {}/{}'.format(epoch, args.epochs))
        logger.info('Current Learning rate: {}'.format(showLR(optimizer)))
        
        model,_ = train_test(model, dset_loaders, criterion, epoch, 'train', optimizer, args, logger, use_gpu, save_path)
        with torch.no_grad():
            model,val_acc = train_test(model, dset_loaders, criterion, epoch, 'val', optimizer, args, logger, use_gpu, save_path)
            model,_ = train_test(model, dset_loaders, criterion, epoch, 'test', optimizer, args, logger, use_gpu, save_path)

        if val_acc>=best_acc:
            torch.save(model.state_dict(), save_path+'/'+args.desc+'/best_'+str(epoch)+'.pt')
            if best_epoch!=-1:
                os.remove(save_path+'/'+args.desc+'/best_'+str(best_epoch)+'.pt')
            best_acc = val_acc
            best_epoch = epoch
            print ("saved best model at {} epoch !".format(epoch))
        if epoch%5==0:
            torch.save(model.state_dict(), save_path+'/'+args.desc+'/'+str(epoch)+'.pt')
            print ("saved model at {} epoch".format(epoch))
        

if __name__=='__main__':
    # Settings
    parser = argparse.ArgumentParser(description='Type your thoughts')
    parser.add_argument('--test', default=False, action='store_true', help='run test on a pretrained model')
    parser.add_argument('--no_scheduler', default=False, action='store_true', help='not apply lr scheduler')
    parser.add_argument('--model_path', default=hp.model_path, help='path to model for testing')
    parser.add_argument('--desc', help='description of the run')
    parser.add_argument('--lr', default=hp.lr, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, help='momentum')
    parser.add_argument('--batch-size', default=hp.batch_size, type=int, help='mini-batch size (default: 32)')
    parser.add_argument('--workers', default=4, help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=hp.epochs, help='number of total epochs')
    parser.add_argument('--max-norm', default=hp.max_norm, type=int, help='Norm cutoff to prevent explosion of gradients')
    
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(hp.GPU)
    use_gpu = torch.cuda.is_available()
    if not use_gpu:
        print ("no gpu,exiting")
        exit()
    if not os.path.exists(os.path.join('model',args.desc+'/')):
        os.mkdir('model/'+args.desc)
    shutil.copy('hp.py','model/'+args.desc+'/'+'parameters.txt')
    main(args, use_gpu)
    shutil.move('model/'+args.desc+'.log','model/'+args.desc+'/'+'all_logs.log',)


