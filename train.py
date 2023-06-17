import wandb
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm 
from datasets import *
from torchvision.transforms import ToTensor, Resize, Compose
from torch.utils.data import DataLoader, Dataset
from models.selective_vio import *
from utils.utils import *

# Hyperparameters
seq_dir = "sequences_05"
weight_path = "./weights/model-p3.pth"
pretrained_flag = False
wandb_flag = True
batch_size = 16
seq_len = 1
learning_rate = 1e-4 #0.00005
lr_warmup = 1e-4
lr_joint1 = 5e-5
lr_joint2 = 1e-5
lr_joint3 = 1e-6

warmup_epoch = 40
joint1_epoch = 80
joint2_epoch = 120
joint3_epoch = 150

start_epoch = 1
num_epochs = 500
min_loss = 30000
weight_decay = 5e-6
efficiency_lambda = 0 # 3e-5
info = f"batch_size: {batch_size}\n" \
       f"pretrained_flag: {pretrained_flag}\n" \
       f"efficiency_lambda: {efficiency_lambda}\n" \
       f"seq_dir: {seq_dir}\n" \
       f"lr_warmup: {lr_warmup}\n" \
       f"lr_joint1: {lr_joint1}\n" \
       f"lr_joint2: {lr_joint2}\n" \
       f"lr_joint3: {lr_joint3}\n" \
       f"warmup_epoch: {warmup_epoch}\n" \
       f"joint1_epoch: {joint1_epoch}\n" \
       f"joint2_epoch: {joint2_epoch}\n" \
       f"joint3_epoch: {joint3_epoch}\n" \
       f"num_epochs: {num_epochs}\n" \
       f"min_loss: {min_loss}\n" \
       f"weight_decay: {weight_decay}\n"

# Logging 
if wandb_flag:
    wandb.init(
        project="cmp-report",
        config={
        "learning_rate": lr_warmup,
        "architecture": "CNN",
        "dataset": "KITTI-Odometry",
        "epochs": num_epochs,
        }
    )
# tqdm logs
train_log = tqdm(total=0, position=3, bar_format='{desc}')
val_log = tqdm(total=0, position=3, bar_format='{desc}')

# Prepare your dataset and dataloaders
train_data = KITTIOdometryDataset(data_dir="./data", seq_dir=seq_dir, sequence_length=seq_len, batch_size=batch_size)
val_data = KITTIOdometryDataset(data_dir="./data", seq_dir=seq_dir, sequence_length=seq_len, batch_size=batch_size)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=True)

# Initialize model, loss function, and optimizer
model = SelectiveVIO().float()
optimizer = torch.optim.Adam(model.parameters(), lr=lr_warmup, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)

# Set GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else: 
    print("NO CUDA DEVICE FOUND")

# Upload Pretrained Weights
if not pretrained_flag:
    # Upload Flownet weights
    pretrained_w = torch.load("weights/VIO-FLOW-MODELS/flownets_bn_EPE2.459.pth.tar", map_location='cpu')
    model_dict = model.visual_encoder.state_dict()
    update_dict = {k: v for k, v in pretrained_w['state_dict'].items() if k in model_dict}
    model_dict.update(update_dict)
    model.visual_encoder.load_state_dict(model_dict)
    for key in update_dict.keys():
        print("Found Key: ", key)
else:
    # Upload full pre-trained model weights
    pretrained_w = torch.load(weight_path, map_location='cpu')
    pretrained_w = {key.replace("module.", ""): value for key, value in pretrained_w.items()}
    model.load_state_dict(pretrained_w)
    for pt_key in pretrained_w.keys():
        print("Found Key: ", pt_key)


# Upload model to GPU
model = model.to(device)
model = torch.nn.DataParallel(model, device_ids = [0])
torch.autograd.set_detect_anomaly(True)

# Make log files
make_log_file(info)

# Training loop
for epoch in (range(start_epoch, num_epochs)):
    model.train()
    train_loss = 0.0
    efficiency_loss = 0.0
    policy_usage = 0.0
    # Update learning rate according to current stage
    if epoch > warmup_epoch and epoch < joint1_epoch:
        optimizer.param_groups[0]['lr'] = lr_joint1
    if epoch > joint2_epoch and epoch < joint3_epoch:
        optimizer.param_groups[0]['lr'] = lr_joint2
    if epoch > joint3_epoch:
        optimizer.param_groups[0]['lr'] = lr_joint3

    # Iteration loop
    policy_list = []
    for i, (concat_image, imu_data, gt_pose, h_first) in enumerate(tqdm(train_loader)):
        # Upload data to GPU
        concat_image = concat_image.cuda().float()
        imu_data = imu_data.cuda().float()
        gt_pose = gt_pose.cuda().float()         

        pose_est_list = []
        gt_pose_list = []
        for j in range(seq_len):
            # Reset gradient
            optimizer.zero_grad()

            # Feed Forward
            if i==0:
                pose_estimation, last_hidden_state, prob, decision = model(epoch, warmup_epoch, i, concat_image[:,j], imu_data[:,j:j+11])
            else:
                # last_hidden_state is a part of autograd-graph and should not change during iteration
                last_hidden_state = (last_hidden_state[0].detach(), last_hidden_state[1].detach())
                pose_estimation, last_hidden_state, prob, decision = model(epoch, warmup_epoch, i, concat_image[:,j], imu_data[:,j:j+11], last_hidden_state)
            # Save the pose estimation of the each seqeunce data
            pose_est_list.append(pose_estimation)
            gt_pose_list.append(gt_pose)
            
        pose_estimation = torch.cat(pose_est_list, dim=1)
        gt_pose = torch.cat(gt_pose_list, dim=1)

        # Calculate Errors
        t_rmse, r_rmse  = calculate_errors(pose_estimation, decision, gt_pose)
        

        # Calculate Loss
        angle_loss       = torch.nn.functional.mse_loss(pose_estimation[:, 3:], gt_pose[:, 3:])
        translation_loss = torch.nn.functional.mse_loss(pose_estimation[:, :3], gt_pose[:, :3])
        pose_loss = 100 * angle_loss + translation_loss        

        # Efficiency Loss
        decision_loss = (decision[:,1].float()).sum(-1).mean()
        num_ones = torch.sum(decision[:, 1] == 1)
        policy_list.append(num_ones.item())

        # Check warm-up status
        if epoch > warmup_epoch:
            loss = pose_loss + efficiency_lambda * decision_loss
        else:
            loss = pose_loss

        # Backward
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        efficiency_loss += decision_loss
        policy_usage += num_ones.item() / len(decision)

    train_loss /= len(train_loader)
    efficiency_loss /= len(train_loader)
    policy_usage /= len(train_loader)
    
    # Log metrics to wandb
    print(policy_list)
    if wandb_flag:
        wandb.log({
            "train_loss": train_loss, 
            "angle_loss": angle_loss, 
            "translation_loss": translation_loss, 
            "efficiency_loss": efficiency_loss, 
            "visual_modelity": policy_usage, 
            "t_rmse": t_rmse, 
            "r_rmse": r_rmse
        })
    print(f"Epoch [{epoch+1}/{num_epochs}],"
        f" Train Loss: {train_loss:.4f},"
        f" Efficiency Loss: {efficiency_loss:.4f},"
        f" Policy Usage: {policy_usage:.4f},"
        f" num_ones {num_ones.item():.4f},"
        f" t_rmse: {t_rmse:.4f},"
        f" r_rmse: {r_rmse:.4f}")
    # --------------------------------------------------------------------------------------------------------------------------

    # Validation loop
    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            policy_usage = 0.0
            gt_list = []
            est_list = []
            policy_list = []

            # Iteration loop
            for i, (concat_image, imu_data, gt_pose, h_first) in enumerate(tqdm(val_loader)):
                # Upload to GPU
                concat_image = concat_image.cuda().float()
                imu_data = imu_data.cuda().float()
                gt_pose = gt_pose.cuda().float() 

                pose_est_list = []
                gt_pose_list = []
                for j in range(seq_len):
                    # Reset gradient
                    optimizer.zero_grad()
                    # Feed Forward
                    if i==0:
                        pose_estimation, last_hidden_state, prob, decision = model(epoch, warmup_epoch, i, concat_image[:,j], imu_data[:,j:j+11])
                    else:
                        # last_hidden_state is a part of autograd-graph and should not change during iteration?
                        last_hidden_state = (last_hidden_state[0].detach(), last_hidden_state[1].detach())
                        pose_estimation, last_hidden_state, prob, decision = model(epoch, warmup_epoch, i, concat_image[:,j], imu_data[:,j:j+11], last_hidden_state)
                    # Save the pose estimation of the each seqeunce data
                    pose_est_list.append(pose_estimation)
                    gt_pose_list.append(gt_pose)
                pose_estimation = torch.cat(pose_est_list, dim=1)
                gt_pose = torch.cat(gt_pose_list, dim=1)

                # Calculate Errors
                t_rmse, r_rmse  = calculate_errors(pose_estimation, decision, gt_pose)
                
                # Save results
                for k in range(gt_pose.shape[0]):
                    gt_list.append(gt_pose[k].cpu())
                    est_list.append(pose_estimation[k].cpu())
                
                # Calculate Loss
                angle_loss = torch.nn.functional.mse_loss(pose_estimation[:, 3:], gt_pose[:, 3:])
                translation_loss = torch.nn.functional.mse_loss(pose_estimation[:, :3], gt_pose[:, :3])
                pose_loss = 100 * angle_loss + translation_loss   
                
                # Efficiency Loss
                efficiency_loss = (decision[:,1].float()).sum(-1).mean()
                num_ones = torch.sum(decision[:, 1] == 1)
                policy_list.append(num_ones.item())
                
                # Check warm-up status
                if epoch > warmup_epoch:
                    loss = pose_loss + efficiency_lambda * efficiency_loss
                else:
                    loss = pose_loss
                
                val_loss += loss.item()
                efficiency_loss += efficiency_loss
                policy_usage += num_ones.item() / len(decision)

            val_loss /= len(val_loader)
            efficiency_loss /= len(val_loader)
            policy_usage /= len(val_loader)

            # Save estimations to txt
            if epoch%10 == 0:
                txt_name = "results/"+str(seq_dir)+"_"+str(epoch)
                est_name = txt_name+".txt"
                modelity_txt_name = txt_name+"_policty_usage.txt"
                write_to_txt(est_list, file_name=est_name)

            # Log metrics to wandb
            if wandb_flag:
                wandb.log({
                            "train_loss": train_loss, 
                            "validation_loss": val_loss,
                            "angle_loss": angle_loss, 
                            "translation_loss": translation_loss, 
                            "efficiency_loss": efficiency_loss, 
                            "visual_modelity": policy_usage, 
                            "t_rmse": t_rmse, 
                            "r_rmse": r_rmse
                        })
            print(f"Epoch [{epoch+1}/{num_epochs}],"
                f" Validation Loss: {val_loss:.4f},"
                f" Training Loss: {train_loss:.4f},"
                f" Efficiency Loss: {efficiency_loss:.4f},"
                f" Policy Usage: {policy_usage:.4f},"
                f" num_ones {num_ones.item():.4f},"
                f" t_rmse: {t_rmse:.4f},"
                f" r_rmse: {r_rmse:.4f}")

            # Save the best model
            if val_loss<min_loss:
                min_loss = val_loss
                model_name = "./weights/model-best.pth"
                torch.save(model.state_dict(), model_name)

# Save the final model
model_name = "./weights/model-"+str(epoch)+"-final-"+str(int(train_loss))+".pth"
torch.save(model.state_dict(), model_name)
if wandb_flag:
    wandb.finish()


