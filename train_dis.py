import os
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

# Import models
from nets.unet import Unet as TeacherModel
from nets.fastlite_ssnet import FastLiteSSNet as StudentModel

# Import loss and training tools
from nets.distillation_loss import HDL
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import LossHistory, EvalCallback
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import download_weights, show_config
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    #---------------------------------#
    #   Use Cuda? False if no GPU
    #---------------------------------#
    Cuda = True
    #---------------------------------------------------------------------#
    #   Single-machine multi-GPU? Ubuntu only. Windows: DP mode
    #---------------------------------------------------------------------#
    distributed     = False
    #---------------------------------------------------------------------#
    #   Use sync_bn? DDP multi-GPU only
    #---------------------------------------------------------------------#
    sync_bn         = False
    #---------------------------------------------------------------------#
    #   Use fp16? Saves ~50% memory, needs pytorch 1.7.1+
    #---------------------------------------------------------------------#
    fp16            = True
    #-----------------------------------------------------#
    #   Classes: num_classes + 1, e.g., 2+1
    #-----------------------------------------------------#
    num_classes = 10
    #-----------------------------------------------------#
    #   Teacher model config
    #-----------------------------------------------------#
    teacher_backbone    = "vgg"
    teacher_pretrained  = False
    teacher_model_path  = "logs/last_epoch_weights.pth"
    
    #-----------------------------------------------------#
    #   Student model config
    #-----------------------------------------------------#
    student_width_mult = 0.3  # Width factor (reduced to 0.3 for memory)
    
    #----------------------------------------------------------------------------------------------------------------------------#
    #   Training parameters
    #----------------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------#
    #   Input size: multiple of 32
    #-----------------------------------------------------#
    input_shape = [512, 512]
    
    #------------------------------------------------------------------#
    #   Freeze stage parameters
    #------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 2
    
    #------------------------------------------------------------------#
    #   Unfreeze stage parameters
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 200
    Unfreeze_batch_size = 2
    
    #------------------------------------------------------------------#
    #   Freeze_Train: freeze backbone first, then unfreeze
    #------------------------------------------------------------------#
    Freeze_Train        = True
    
    #------------------------------------------------------------------#
    #   Other: lr, optimizer, decay
    #------------------------------------------------------------------#
    Init_lr             = 1e-4
    Min_lr              = Init_lr * 0.01
    optimizer_type      = "adam"
    momentum            = 0.9
    weight_decay        = 0
    lr_decay_type       = 'cos'
    
    #------------------------------------------------------------------#
    #   Save and eval parameters
    #------------------------------------------------------------------#
    save_period         = 5
    save_dir            = 'logs_distill'
    eval_flag           = True
    eval_period         = 5
    
    #------------------------------#
    #   Dataset path
    #------------------------------#
    VOCdevkit_path  = 'VOCdevkit'
    
    #------------------------------------------------------------------#
    #   Distillation loss weights
    #------------------------------------------------------------------#
    alpha = 0.5  # Classification loss weight
    beta = 0.3   # CS divergence weight
    gamma = 0.2  # Dice loss weight
    
    #------------------------------------------------------------------#
    #   Data loading parameters
    #------------------------------------------------------------------#
    num_workers     = 4
    
    #------------------------------------------------------#
    #   Set GPUs to use
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
    
    #-----------------------------------------------------#
    #   Create model directory
    #-----------------------------------------------------#
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    #-----------------------------------------------------#
    #   Load teacher model
    #-----------------------------------------------------#
    if local_rank == 0:
        print('Loading Teacher Model...')
    
    teacher_model = TeacherModel(num_classes=num_classes, pretrained=teacher_pretrained, backbone=teacher_backbone)
    
    if teacher_model_path != '':
        #------------------------------------------------------#
        #   Load teacher model weights
        #------------------------------------------------------#
        if local_rank == 0:
            print('Load Teacher weights {}.'.format(teacher_model_path))
        
        #------------------------------------------------------#
        #   Load based on pretrained weight keys and model keys
        #------------------------------------------------------#
        model_dict      = teacher_model.state_dict()
        pretrained_dict = torch.load(teacher_model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        teacher_model.load_state_dict(model_dict)
        
        if local_rank == 0:
            print("\nTeacher Model: Successful Load Key:", str(load_key)[:500], "……")
            print("Teacher Model: Fail To Load Key:", str(no_load_key)[:500], "……")
    
    #-----------------------------------------------------#
    #   Set teacher model to eval mode
    #-----------------------------------------------------#
    teacher_model = teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    #-----------------------------------------------------#
    #   Create student model
    #-----------------------------------------------------#
    if local_rank == 0:
        print('Creating Student Model...')
    
    student_model = StudentModel(num_classes=num_classes, width_mult=student_width_mult).train()
    weights_init(student_model)
    
    #----------------------#
    #   Record Loss
    #----------------------#
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, student_model, input_shape=input_shape)
    else:
        loss_history    = None
        
    #------------------------------------------------------------------#
    #   torch 1.2 doesn't support amp, recommend torch 1.7.1+ for fp16
    #------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None
    
    student_model_train = student_model.train()
    #----------------------------#
    #   Multi-GPU sync Bn
    #----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        student_model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(student_model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")
    
    if Cuda:
        if distributed:
            #----------------------------#
            #   Multi-GPU parallel training
            #----------------------------#
            student_model_train = student_model_train.cuda(local_rank)
            student_model_train = torch.nn.parallel.DistributedDataParallel(student_model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            student_model_train = torch.nn.DataParallel(student_model)
            cudnn.benchmark = True
            student_model_train = student_model_train.cuda()
            
            # Teacher model also needs to be moved to GPU
            teacher_model = teacher_model.cuda()
    
    #---------------------------#
    #   Read dataset txt files
    #---------------------------#
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),"r") as f:
        val_lines = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)
        
    if local_rank == 0:
        show_config(
            num_classes = num_classes, 
            teacher_backbone = teacher_backbone, 
            teacher_model_path = teacher_model_path,
            student_width_mult = student_width_mult,
            input_shape = input_shape, 
            Init_Epoch = Init_Epoch, 
            Freeze_Epoch = Freeze_Epoch, 
            UnFreeze_Epoch = UnFreeze_Epoch, 
            Freeze_batch_size = Freeze_batch_size, 
            Unfreeze_batch_size = Unfreeze_batch_size, 
            Freeze_Train = Freeze_Train, 
            Init_lr = Init_lr, 
            Min_lr = Min_lr, 
            optimizer_type = optimizer_type, 
            momentum = momentum, 
            lr_decay_type = lr_decay_type, 
            save_period = save_period, 
            save_dir = save_dir, 
            num_workers = num_workers, 
            num_train = num_train, 
            num_val = num_val
        )
    
    #------------------------------------------------------#
    #   Backbone common, freeze speeds up
    #------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        #------------------------------------#
        #   Freeze certain parts for training
        #------------------------------------#
        if Freeze_Train:
            # Student model freeze training - freeze everything except BEGM and FCSF
            for name, param in student_model.named_parameters():
                param.requires_grad = False
                # Unfreeze BEGM and FCSF modules
                if "begm" in name or "fcsf" in name:
                    param.requires_grad = True
        
        #-------------------------------------------------------------------#
        #   If not freeze training, directly set batch_size to Unfreeze_batch_size
        #-------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
        
        #-------------------------------------------------------------------#
        #   Determine current batch_size, adaptively adjust learning rate
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
            'adam'  : optim.Adam(filter(lambda p: p.requires_grad, student_model.parameters()), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(filter(lambda p: p.requires_grad, student_model.parameters()), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
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
            eval_callback   = EvalCallback(student_model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, 
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None
        
        #---------------------------------------#
        #   Initialize distillation loss function
        #---------------------------------------#
        distillation_loss = HDL(num_classes=num_classes, alpha=alpha, beta=beta, gamma=gamma)
        
        #---------------------------------------#
        #   Start model training
        #---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            #---------------------------------------#
            #   If model has frozen learning parts
            #   then unfreeze and set parameters
            #---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size
                
                #-------------------------------------------------------------------#
                #   Determine current batch_size, adaptively adjust learning rate
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
                    
                #---------------------------------------#
                #   Unfreeze all model parameters
                #---------------------------------------#
                for param in student_model.parameters():
                    param.requires_grad = True
                
                #---------------------------------------#
                #   Recreate optimizer with all parameters
                #---------------------------------------#
                optimizer = {
                    'adam'  : optim.Adam(filter(lambda p: p.requires_grad, student_model.parameters()), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
                    'sgd'   : optim.SGD(filter(lambda p: p.requires_grad, student_model.parameters()), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
                }[optimizer_type]
                
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
            
            #---------------------------------------#
            #   Distillation training
            #---------------------------------------#
            # Set default value for cls_weights to avoid None value issues
            cls_weights_default = np.ones([num_classes], np.float32)
            fit_one_epoch(student_model_train, student_model, teacher_model, loss_history, eval_callback, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, dice_loss=False, focal_loss=False, 
                    cls_weights=cls_weights_default, num_classes=num_classes, fp16=fp16, scaler=scaler, save_period=save_period, 
                    save_dir=save_dir, local_rank=local_rank, distillation_loss=distillation_loss)
            
            if distributed:
                dist.barrier()
        
        if local_rank == 0:
            loss_history.writer.close()
