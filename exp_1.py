#=========================================================
# Developer: Vajira Thambawita
# Reference: https://github.com/meetshah1995/pytorch-semseg
#=========================================================



import argparse
from datetime import datetime
import os
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt

#Pytorch
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from torchvision import models, transforms,datasets, utils
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torchsummary import summary


from nets import UNet_009 as UNet
from data import PolypsDatasetWithGridEncoding
from data import PolypsDatasetWithGridEncoding_TestData
from utils import dice_coeff, iou_pytorch


#======================================
# Get and set all input parameters
#======================================

parser = argparse.ArgumentParser()

# Hardware
parser.add_argument("--device", default="gpu", help="Device to run the code")
parser.add_argument("--device_id", type=int, default=0, help="")

# Optional parameters to identify the experiments
parser.add_argument("--name", default="", type=str, help="A name to identify this test later")
parser.add_argument("--py_file",default=os.path.abspath(__file__)) # store current python file


# Directory and file handling
parser.add_argument("--img_root_train", 
                    default="/work/vajira/DATA/hyper_kvasir/data_new/segmented_train_val/data/segmented-images/train/images",
                    help="image root directory")

parser.add_argument("--mask_root_train",
                    default="/work/vajira/DATA/hyper_kvasir/data_new/segmented_train_val/data/segmented-images/train/masks",
                    help="corresponding mask images directory")

parser.add_argument("--img_root_val", 
                    default="/work/vajira/DATA/hyper_kvasir/data_new/segmented_train_val/data/segmented-images/val/images",
                    help="image root directory")

parser.add_argument("--mask_root_val",
                    default="/work/vajira/DATA/hyper_kvasir/data_new/segmented_train_val/data/segmented-images/val/masks",
                    help="corresponding mask images directory")

parser.add_argument("--out_dir", 
                    default="/work/vajira/DATA/GRID_GAN/output",
                    help="Main output dierectory")

parser.add_argument("--tensorboard_dir", 
                    default="/work/vajira/DATA/GRID_GAN/tensorboard",
                    help="Folder to save output of tensorboard")

parser.add_argument("--test_data_dir",
                    default="/work/vajira/DATA/mediaeval2020/test_data/all/Medico_automatic_polyp_segmentation_challenge_test_data/images",
                    help="Test data folder to load to : PolypsDatasetWithGridEncoding_TestData")   

parser.add_argument("--test_out_dir",
                   default= "/work/vajira/DATA/mediaeval2020/test_data_predictions",
                   help="Output folder for testing data"
)           

# Parameters
parser.add_argument("--bs", type=int, default=2, help="Mini batch size")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training")
parser.add_argument("--num_workers", type=int, default=4, help="Number of workers in dataloader")
parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay of the optimizer")
parser.add_argument("--lr_sch_factor", type=float, default=0.1, help="Factor to reduce lr in the scheduler")
parser.add_argument("--lr_sch_patience", type=int, default=25, help="Num of epochs to be patience for updating lr")


# Action handling 
parser.add_argument("--num_epochs", type=int, default=2000, help="Numbe of epochs to train")
parser.add_argument("--start_epoch", type=int, default=0, help="Start epoch in retraining")
parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to print from validation set")
parser.add_argument("action", type=str, help="Select an action to run", choices=["train", "retrain", "inference", "check"])
parser.add_argument("--checkpoint_interval", type=int, default=25, help="Interval to save checkpoint models")
#parser.add_argument("--fold", type=str, default="fold_1", help="Select the validation fold", choices=["fold_1", "fold_2", "fold_3"])
#parser.add_argument("--num_test", default= 200, type=int, help="Number of samples to test set from 1k dataset")
parser.add_argument("--model_path", default="", help="Model path to load weights")
parser.add_argument("--num_of_samples", default=30, type=int, help="Number of samples to validate (Montecalo sampling)")

opt = parser.parse_args()


#==========================================
# Device handling
#==========================================
torch.cuda.set_device(opt.device_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#===========================================
# Folder handling
#===========================================

#make output folder if not exist
os.makedirs(opt.out_dir, exist_ok=True)


# make subfolder in the output folder 
py_file_name = opt.py_file.split("/")[-1] # Get python file name (soruce code name)
checkpoint_dir = os.path.join(opt.out_dir, py_file_name + "/checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# make tensorboard subdirectory for the experiment
tensorboard_exp_dir = os.path.join(opt.tensorboard_dir, py_file_name)
os.makedirs( tensorboard_exp_dir, exist_ok=True)

#==========================================
# Tensorboard
#==========================================
# Initialize summary writer
writer = SummaryWriter(tensorboard_exp_dir)

#==========================================
# Prepare Data
#==========================================
def prepare_data():

    # Transforms
    data_transforms = transforms.Compose([ 
            transforms.ToTensor()
        ])

    dataset_train = PolypsDatasetWithGridEncoding(opt.img_root_train, opt.mask_root_train, grid_sizes=[256], transforms= data_transforms)
    dataset_val = PolypsDatasetWithGridEncoding(opt.img_root_val, opt.mask_root_val, grid_sizes=[256], transforms= data_transforms)
   
    print("dataset train=", len(dataset_train))
    print("dataset val=", len(dataset_val))

    data_loader_train = torch.utils.data.DataLoader(
                dataset_train, batch_size=opt.bs, shuffle=True, num_workers=opt.num_workers)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=False, num_workers=opt.num_workers)

    return data_loader_train, data_loader_val

#================================================
# Train the model
#================================================
def train_model(model, optimizer, criterion, data_loader_train, data_loader_val):
    
    global_step = opt.start_epoch * len(data_loader_train)

    best_model = copy.deepcopy(model)
    best_mIOU = 0.0
    best_epoch = 0
    best_val_loss = 0.0
    
    # Training
    for epoch in range(opt.start_epoch + 1, opt.start_epoch + opt.num_epochs + 1):
        
        model.train()
        epoch_loss = 0.0
        for i, sample in enumerate(data_loader_train, 0):

            # Input image and corresponding mask
            img = sample["img"].to(device)
            mask_gt = sample["mask"].to(device) # get only one channel
            grid_encode = sample["grid_encode"].to(device)

            #print("img shape:", sample["img"].shape)
            #print("mask shape:", sample["mask"].shape)
            #print("gird shape:", sample["grid_encode"].shape)

            # Prediction
            mask_pred = model(img, grid_encode)

            # Loss
            loss = criterion(mask_pred, mask_gt)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('Loss/train', loss.item(), global_step)

            global_step += 1

        print("Epoch loss:", epoch_loss)
        writer.add_scalar('Epoch_Loss/train', epoch_loss/len(data_loader_train), epoch) # Mean of lossess

        writer.add_images("train/0_img", img, epoch)
        writer.add_images("train/0_encode", grid_encode, epoch)
        writer.add_images("train/1_gt", mask_gt, epoch)
        writer.add_images("train/2_pred", torch.sigmoid(mask_pred) > 0.5, epoch)

        # Validation stats
        val_ans = validate_model(model, data_loader_val)

        if val_ans["IOU_mean"] > best_mIOU:
            best_mIOU = val_ans["IOU_mean"]
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            best_val_loss = val_ans["loss_mean"]
            print("Found a best model at epoch {} mLoss={}\t\t wiht mIOU={} ".format(
                best_val_loss, best_epoch,best_mIOU 
            ))

        writer.add_scalar('Epoch_Loss/val', val_ans["loss_mean"], epoch)
        writer.add_scalar('Epoch_Loss/val_IOU', val_ans["IOU_mean"], epoch)

        writer.add_images("val/0_img", val_ans["img"], epoch)
        writer.add_images("val/0_encode", val_ans["grid_encode"], epoch)
        writer.add_images("val/1_gt", val_ans["mask_gt"], epoch)
        writer.add_images("val/2_pred_mean", val_ans["mask_pred"], epoch)

        # making heatmaps to tensorboard
        heat_map_list = generate_heatmapts(val_ans["mask_std"])

        writer.add_figure("val/3_pred_std", heat_map_list, epoch)
    
        print("Epoch: {} \t\t Val Loss_mean: {} \t\t Val IOU_mean: {}".format(epoch, val_ans["loss_mean"], val_ans["IOU_mean"]))
        
        # save model
        if epoch % opt.checkpoint_interval == 0:
            save_model(model, optimizer, epoch, val_ans["loss_mean"],val_ans["IOU_mean"], "checkpoing")
            print("Models saved")

    save_model(best_model, optimizer, best_epoch, best_val_loss, best_mIOU, "best")
    print("Best model saved")

#==============================================
# Heatmap generator from tensor
#==============================================
def generate_heatmapts(img_tensor):
    print(img_tensor.shape)
    fig_list = []
    for n in range(img_tensor.shape[0]):
        img = img_tensor[n]
        img = img.squeeze(dim=0)
        img_np = img.detach().cpu().numpy()
        #img_np = np.transforms(img_np, (1,2,0))
        
        plt.imshow(img_np, cmap="hot")
        fig = plt.gcf()
        fig_list.append(fig)
        # plt.clf()
        plt.close()

    return fig_list

#===============================================
# Validate model
#===============================================
def validate_model(model, data_loader_test):
    
    model.train() # for montecalo sampling
    criterion_val = nn.BCEWithLogitsLoss()

    tot_dc = 0
    epoch_loss_val = 0
    iou = 0.0

    num_samples_to_print = 5
    img_val = []
    grid_encode_val = []
    mask_gt_val = []
    mask_pred_val = []
    mask_pred_val_std = []
    

    for i, sample in enumerate(data_loader_test, 0):

        # Input image and corresponding mask
        img = sample["img"].to(device)
        mask_gt = sample["mask"].to(device) # get only one channel
        grid_encode = sample["grid_encode"].to(device)

        
        mask_all_samples = [] # collection of samples for the same input

        # Prediction
        for s in range(opt.num_of_samples):
            mask_pred = model(img, grid_encode)
            mask_pred = torch.sigmoid(mask_pred)
            mask_all_samples.append(mask_pred)

        mask_all_in_one = torch.cat(mask_all_samples, dim=0) # concatenate all sampels in dim= 0

        mask_mean = mask_all_in_one.mean(dim=0)
        mask_std = mask_all_in_one.std(dim=0)

        mask_pred = (mask_mean > 0.5).float()
        mask_pred = mask_pred.unsqueeze(dim=0)

        # Loss
        loss_val = criterion_val(mask_pred, mask_gt)
        epoch_loss_val += loss_val.item()

        # Dice coefficient
        #print("mask pred:", mask_pred.shape)
        #print("mask gt:", mask_gt.shape)
        #tot_dc += dice_coeff(mask_pred, mask_gt).item()
        #print(mask_std.shape)
        #print(mask_pred.shape)
        mask_std_unsqueeze = mask_std.unsqueeze(dim=0)

        iou += iou_pytorch(mask_pred, mask_gt).item()
        # print(iou)
        

        if i < num_samples_to_print:
            img_val.append(img)
            grid_encode_val.append(grid_encode)
            mask_gt_val.append(mask_gt)
            mask_pred_val.append(mask_pred)
            mask_pred_val_std.append(mask_std_unsqueeze)

    # Make tensor by concatenating list of tensors
    img_val = torch.cat(img_val, dim=0)
    grid_encode_val = torch.cat(grid_encode_val, dim=0)
    mask_gt_val = torch.cat(mask_gt_val, dim=0)
    mask_pred_val = torch.cat(mask_pred_val, dim=0)
    mask_pred_val_std = torch.cat(mask_pred_val_std, dim=0)

    epoch_loss_val_mean = epoch_loss_val / len(data_loader_test)
    iou_mean = iou / len(data_loader_test)

    # Printing
    # print("tot DC:", tot_dc)
    # print("IOU:", iou)

    return {"loss_mean": epoch_loss_val_mean, "IOU_mean": iou_mean, "img": img_val, "mask_gt": mask_gt_val, "grid_encode":grid_encode_val,
    "mask_pred": mask_pred_val, "mask_std": mask_pred_val_std}



#===============================================
# Prepare models
#===============================================
def prepare_model():
    model = UNet(n_channels=4, n_classes=1) # 4 = 3 channels + 1 grid encode

    model.to(device)

    return model

#====================================
# Run training process
#====================================
def run_train():
    model = prepare_model()

    data_loader_train, data_loader_val = prepare_data()

    optimizer = optim.RMSprop(model.parameters(), lr=opt.lr, weight_decay=1e-8)
    
    #lr_schdlr = lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=opt.lr_sch_factor, patience=opt.lr_sch_patience)


    if model.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
        #criterion_val = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
        #criterion_val = nn.BCEWithLogitsLoss()

    train_model(model, optimizer, criterion,  data_loader_train, data_loader_val)
#====================================
# Re-train process
#====================================
def run_retrain():

    if opt.model_path == "":
        print("Please pass correct model path")
        exit()

    model = prepare_model()

    checkpoint = torch.load(opt.model_path) # load the checkpoint
    model.load_state_dict(checkpoint["model_state_dict"])
    model_epoch = checkpoint["epoch"]
    print("Model loadded succesfully")

    # set default start epoch
    setattr(opt, "start_epoch", model_epoch)
    # opt.set_default(start_epoch = model_epoch)

    data_loader_train, data_loader_test = prepare_data()

    optimizer = optim.RMSprop(model.parameters(), lr=opt.lr, weight_decay=1e-8)

    if model.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
        #criterion_val = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
        #criterion_val = nn.BCEWithLogitsLoss()

    train_model(model, optimizer, criterion,  data_loader_train, data_loader_test)

#=====================================
# Save models
#=====================================
def save_model(model, optimizer,  epoch,  validation_loss, mIOU, checkpoint_type):
    check_point_name = py_file_name + "_epoch:{}_mIOU_{}_{}.pt".format(epoch, mIOU, checkpoint_type) # get code file name and make a name
    check_point_path = os.path.join(checkpoint_dir, check_point_name)
    # save torch model
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        # "train_loss": train_loss,
        "val_loss": validation_loss,
        "mIOU":mIOU
    }, check_point_path)

#=====================================
# Check model
#====================================
def check_model_graph():
    raise NotImplementedError


#===================================
# Inference from pre-trained models
#===================================

def do_inference():

    if opt.model_path == "":
        print("Please pass correct model path")
        exit()

    model = prepare_model()

    checkpoint = torch.load(opt.model_path) # load the checkpoint
    model.load_state_dict(checkpoint["model_state_dict"])

    model.train()

    print("Model loaded with checkpoint from:", opt.model_path)

    # Transforms
    data_transforms = transforms.Compose([ 
            transforms.ToTensor()
        ])
    dataset_test = PolypsDatasetWithGridEncoding_TestData(opt.test_data_dir, grid_sizes=[256], transforms=data_transforms)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=opt.num_workers)

    for i, sample in enumerate(data_loader_test, 0):

        # Input image and corresponding mask
        img = sample["img"].to(device)
        grid_encode = sample["grid_encode"].to(device)
        img_name = sample["img_name"]

        mask_all_samples = [] # collection of samples for the same input

        # Prediction
        with torch.no_grad():
            for s in range(opt.num_of_samples):
                mask_pred = model(img, grid_encode)
                mask_pred = torch.sigmoid(mask_pred)
                mask_all_samples.append(mask_pred)

        mask_all_in_one = torch.cat(mask_all_samples, dim=0) # concatenate all sampels in dim= 0

        mask_mean = mask_all_in_one.mean(dim=0)
        mask_std = mask_all_in_one.std(dim=0)

        mask_pred = (mask_mean > 0.5).float()
        mask_pred = mask_pred.unsqueeze(dim=0)

        out_dir = opt.test_out_dir + "/" + py_file_name + "/" +  str(opt.num_of_samples)
        os.makedirs(out_dir, exist_ok=True)
        utils.save_image(mask_pred, os.path.join(out_dir, str(img_name[0])), format="PNG")



if __name__ == "__main__":

    data_loaders = prepare_data()
    print(vars(opt))
    print("Test OK")

    # Train or retrain or inference
    if opt.action == "train":
        print("Training process is strted..!")
        run_train()
        pass

    elif opt.action == "retrain":
        print("Retrainning process is strted..!")
        run_retrain()
        pass

    elif opt.action == "inference":
        print("Inference process is strted..!")
        do_inference()
        print("Done")

    elif opt.action == "check":
        check_model_graph()
        print("Check pass")

    # Finish tensorboard writer
    writer.close()

