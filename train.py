import os
import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.unet import Unet
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import LossHistory, EvalCallback
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import download_weights, show_config
from utils.utils_fit import fit_one_epoch

'''
Training notes:
1. VOC format required. Input: .jpg, Label: .png
   Grayscale auto-converted to RGB.
   2-class: background=0, target=1 (not 255).
   Format fix: https://github.com/bubbliiiing/segmentation-format-fix

2. Loss indicates convergence. Validation loss should decrease.
   Absolute value meaningless. Divide by 10000 to look better.
   Loss saved in logs/loss_%Y_%m_%d_%H_%M_%S.
   
3. Weights in logs/. Epoch has multiple Steps.
   Few Steps won't save.
'''
if __name__ == "__main__":
    #---------------------------------#
    #   Use Cuda? False if no GPU
    #---------------------------------#
    Cuda = True
    #---------------------------------------------------------------------#
    #   Single-machine multi-GPU? Ubuntu only. Windows: DP mode
    #   DP: distributed=False, CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP: distributed=True, CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    #---------------------------------------------------------------------#
    distributed     = False
    #---------------------------------------------------------------------#
    #   Use sync_bn? DDP multi-GPU only
    #---------------------------------------------------------------------#
    sync_bn         = False
    #---------------------------------------------------------------------#
    #   Use fp16? Saves ~50% memory, needs pytorch 1.7.1+
    #---------------------------------------------------------------------#
    fp16            = False
    #-----------------------------------------------------#
    #   Classes: num_classes + 1, e.g., 2+1
    #-----------------------------------------------------#
    num_classes = 10
    #-----------------------------------------------------#
    #   Backbone: vgg or resnet50
    #-----------------------------------------------------#
    backbone    = "vgg"
    #----------------------------------------------------------------------------------------------------------------------------#
    #   Pretrained: backbone weights loaded during construction.
    #   model_path set -> backbone not loaded.
    #   model_path='', pretrained=True -> load backbone only.
    #   model_path='', pretrained=False -> train from scratch.
    #----------------------------------------------------------------------------------------------------------------------------#
    pretrained  = False
    #----------------------------------------------------------------------------------------------------------------------------#
    #   Weights from README. Universal across datasets.
    #   Essential for 99% cases. Random weights = poor results.
    #   Dimension mismatch normal for custom datasets.
    #   Resume: set model_path to logs/weights.
    #   model_path='' -> no full weights.
    #   Train from backbone: model_path='', pretrain=True.
    #   Train from scratch: model_path='', pretrain=False, Freeze_Train=False.
    #   NOT recommended to train from scratch!
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path  = "model_data/unet_vgg_voc.pth"
    
    #-----------------------------------------------------#
    #   Input size: multiple of 32
    #-----------------------------------------------------#
    input_shape = [512, 512]
    
    #----------------------------------------------------------------------------------------------------------------------------#
    #   Two stages: freeze (less memory) and unfreeze.
    #   Poor GPU: set Freeze_Epoch = UnFreeze_Epoch.
    #   
    #   Suggestions:
    #   (1) Full model pretrained:
    #       Adam: Init_lr=1e-4, SGD: Init_lr=1e-2
    #       UnFreeze_Epoch: 100-300
    #   (2) Backbone pretrained:
    #       Need more epochs (120-300) to escape local optimum.
    #       Adam faster than SGD.
    #   (3) batch_size: as large as GPU allows.
    #       resnet50: batch_size != 1
    #       Freeze_batch_size = 1-2x Unfreeze_batch_size
    #----------------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Freeze stage: backbone frozen, fine-tuning only
    #   Init_Epoch: start epoch (resume: > Freeze_Epoch)
    #   Freeze_Epoch: freeze epochs (invalid if Freeze_Train=False)
    #   Freeze_batch_size: freeze batch size
    #                       (Invalid when Freeze_Train=False)
    #------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 100
    Freeze_batch_size   = 2
    #------------------------------------------------------------------#
    #   Unfreeze stage training parameters
    #   Backbone unfrozen, feature extraction changes
    #   More memory, all parameters change
    #   UnFreeze_Epoch          Total training epochs
    #   Unfreeze_batch_size     Batch size after unfreeze
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 300
    Unfreeze_batch_size = 2
    #------------------------------------------------------------------#
    #   Freeze_Train    Whether to do freeze training
    #                   Default: freeze backbone first, then unfreeze.
    #------------------------------------------------------------------#
    Freeze_Train        = True

    #------------------------------------------------------------------#
    #   Other training parameters: learning rate, optimizer, learning rate decay
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr         Max learning rate
    #                   Adam: Init_lr=1e-4
    #                   SGD: Init_lr=1e-2
    #   Min_lr          Min learning rate, default 0.01 of max
    #------------------------------------------------------------------#
    Init_lr             = 1e-4
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    #   optimizer_type  Optimizer type: adam, sgd
    #                   Adam: Init_lr=1e-4
    #                   SGD: Init_lr=1e-2
    #   momentum        Optimizer momentum parameter
    #   weight_decay    Weight decay, prevents overfitting
    #                   adam causes weight_decay error, set to 0 when using adam.
    #------------------------------------------------------------------#
    optimizer_type      = "adam"
    momentum            = 0.9
    weight_decay        = 0
    #------------------------------------------------------------------#
    #   lr_decay_type   Learning rate decay: 'step', 'cos'
    #------------------------------------------------------------------#
    lr_decay_type       = 'cos'
    #------------------------------------------------------------------#
    #   save_period     Save weights every N epochs
    #------------------------------------------------------------------#
    save_period         = 5
    #------------------------------------------------------------------#
    #   save_dir        Folder for weights and logs
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    #   eval_flag       Evaluate during training? on validation set
    #   eval_period     Evaluate every N epochs
    #   mAP here vs get_map.py: (1) validation set, (2) conservative for speed.
    #------------------------------------------------------------------#
    eval_flag           = True
    eval_period         = 5
    
    #------------------------------#
    #   Dataset path
    #------------------------------#
    VOCdevkit_path  = 'VOCdevkit'
    #------------------------------------------------------------------#
    #   Few classes: True. Many classes, large batch (>10): True. Many classes, small batch (<10): False
    #------------------------------------------------------------------#
    dice_loss       = False
    #------------------------------------------------------------------#
    #   Use focal loss? Prevents class imbalance
    #------------------------------------------------------------------#
    focal_loss      = False
    #------------------------------------------------------------------#
    #   Class weights? Use numpy array, length = num_classes.
    #   Example: num_classes = 3, cls_weights = np.array([1, 2, 3], np.float32)
    #------------------------------------------------------------------#
    cls_weights     = np.ones([num_classes], np.float32)
    #------------------------------------------------------------------#
    #   num_workers     Multi-thread loading, 1 = disabled. Faster but more memory.
    #   Enable only when IO bottleneck (GPU >> read speed).
    #------------------------------------------------------------------#
    num_workers     = 4

    #------------------------------------------------------#
    #   Set GPU to use
    #------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0

    #----------------------------------------------------#
    #   Download pretrained weights
    #----------------------------------------------------#
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(backbone)  
            dist.barrier()
        else:
            download_weights(backbone)

    model = Unet(num_classes=num_classes, pretrained=pretrained, backbone=backbone).train()
    if not pretrained:
        weights_init(model)
    if model_path != '':
        #------------------------------------------------------#
        #   Load weights from README
        #------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        
        #------------------------------------------------------#
        #   Load weights by matching keys
        #------------------------------------------------------#
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        #------------------------------------------------------#
        #   Show unmatched keys
        #------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44mNote: head not loaded is normal, backbone not loaded is error.\033[0m")

    #----------------------#
    #   Record Loss
    #----------------------#
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None
        
    #------------------------------------------------------------------#
    #   torch 1.2 doesn't support amp, suggest torch 1.7.1+ for fp16
    #   So torch1.2 shows "could not be resolve"
    #------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    #----------------------------#
    #   Multi-card sync Bn
    #----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            #----------------------------#
            #   Multi-card parallel run
            #----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
    
    #---------------------------#
    #   Read dataset txt
    #---------------------------#
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),"r") as f:
        val_lines = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)
        
    if local_rank == 0:
        show_config(
            num_classes = num_classes, backbone = backbone, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )
    #------------------------------------------------------#
    #   Backbone universal, freeze speeds up, prevents corruption.
    #   Reduce Batch_size if OOM.
    #------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        #------------------------------------#
        #   Freeze partial training
        #------------------------------------#
        if Freeze_Train:
            model.freeze_backbone()
            
        #-------------------------------------------------------------------#
        #   If not freeze training, set batch_size to Unfreeze_batch_size
        #-------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        #-------------------------------------------------------------------#
        #   Adjust learning rate based on current batch_size
        #-------------------------------------------------------------------#
        nbs             = 16
        lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        #---------------------------------------#
        #   Select optimizer based on optimizer_type
        #---------------------------------------#
        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]

        #---------------------------------------#
        #   Get learning rate decay formula
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        #---------------------------------------#
        #   Determine epoch length
        #---------------------------------------#
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("Dataset too small, cannot continue training, please expand dataset.")

        train_dataset   = UnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset     = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
        
        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last = True, collate_fn = unet_dataset_collate, sampler=train_sampler)
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last = True, collate_fn = unet_dataset_collate, sampler=val_sampler)
        
        #----------------------#
        #   Record eval map curve
        #----------------------#
        if local_rank == 0:
            eval_callback   = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None
        
        #---------------------------------------#
        #   Start model training
        #---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            #---------------------------------------#
            #   If model has frozen parts
            #   Unfreeze and set parameters
            #---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                #-------------------------------------------------------------------#
                #   Adjust learning rate based on current batch_size
                #-------------------------------------------------------------------#
                nbs             = 16
                lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
                lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                #---------------------------------------#
                #   Get learning rate decay formula
                #---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
                model.unfreeze_backbone()
                            
                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("Dataset too small, cannot continue training, please expand dataset.")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last = True, collate_fn = unet_dataset_collate, sampler=train_sampler)
                gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last = True, collate_fn = unet_dataset_collate, sampler=val_sampler)

                UnFreeze_flag = True

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)

            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()
