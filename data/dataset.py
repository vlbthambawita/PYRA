# Vajira Thambawit
# Loading polyps and corresponding mask and bounding box
import os
import cv2
import numpy as np
import torch
import sys
sys.path.append("..") #

from PIL import Image 
from torchvision import models, transforms
from utils.utils  import generate_checkerboard, get_tiled_ground_truth

import imgaug as ia
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import imgaug.augmenters as iaa



 # Conver mask image to BB
def mask_to_bb(pil_mask): #image= gray scaled image

    #image = cv2.imread(mask_path)
    image = cv2.cvtColor(np.array(pil_mask), cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(image, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    boxes = []

    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        #centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])

        boxes.append([boundRect[i][0], boundRect[i][1], boundRect[i][0] + boundRect[i][2], boundRect[i][1] + boundRect[i][3]]) #xmin, ymin, xmax, ymax

    im_bb = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    for i in range(len(contours)):
        
        cv2.rectangle(im_bb, (int(boundRect[i][0]), int(boundRect[i][1])), \
        (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), (255,255, 255), cv2.FILLED)

    return im_bb, boxes

class PolypsDatasetWithGridEncoding(object):

    def __init__(self, imgs_root, masks_root, grid_sizes=[2,4,8,16,32,64,128,256] , transforms = None):

        self.imgs_root = imgs_root
        self.masks_root = masks_root

        self.transforms = transforms

        self.length_of_grid_sizes = len(grid_sizes)

        self.imgs = list(sorted(os.listdir(self.imgs_root)))
        self.masks = list(sorted(os.listdir(self.masks_root)))

        self.num_imgs = len(self.imgs)

        self.imgs = self.imgs * self.length_of_grid_sizes
        self.masks = self.masks * self.length_of_grid_sizes

        self.grid_sizes_repeated = np.repeat(grid_sizes, self.num_imgs)

        self.all_in_one = list(zip(self.imgs, self.masks, self.grid_sizes_repeated)) #(img, mask, grid_size)

       # self.imgs_repeated
        #self.masks_repea

    def __len__(self):
        return len(self.all_in_one)


    def __getitem__(self, idx):

        img_path = os.path.join(self.imgs_root, self.all_in_one[idx][0]) # 0th one= image
        mask_path = os.path.join(self.masks_root, self.all_in_one[idx][1]) # 1st one = mask
        grid_size = self.all_in_one[idx][2] # 2nd one = grid size

        #print("img path=", img_path)
        #print("mask_pth", mask_path)
        #print("grid size=", grid_size)

        img = Image.open(img_path)
        mask = Image.open(mask_path)

        grid_encode = generate_checkerboard(256, 256, grid_size)#[:, :, 0]
        #print("finished grid encode")
       # print()

        

        # resizing
        img = img.resize((256, 256), Image.NEAREST)
        mask = mask.resize((256, 256), Image.NEAREST)

        # covert to numpy

        img = np.array(img)
        mask = np.array(mask) #



        # clean mask values (this is done to remove small values in mask images)
        mask = (mask > 128) * 255 # if mask value > 128, set it to 255 (this will remove 254, 253 values and 1,2 etc)
         
        mask = get_tiled_ground_truth(mask, grid_size)
        #mask = mask[:, :, 0]

        # To tensor
       # mask = np.array(mask)
        #mask = torch.from_numpy(mask)


        # TO tensor
        # img = np.asarray(img)
        # img = torch.from_numpy(img)

        # mask to BB
        #bb_img, boxes = mask_to_bb(mask)
        # convert everything into a torch.Tensor
        #boxes = torch.as_tensor(boxes, dtype=torch.float32)
        #image_id = torch.tensor([idx])
        
        #area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        #iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        #labels = torch.ones((len(boxes),), dtype=torch.int64)
        # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        #target = {}
        
        # target["bb_img"] = bb_img
        #target["boxes"] = boxes
        
        #target["labels"]  = labels
       # target["masks"] = mask
        #target["image_id"] = image_id
        #target["area"] = area
        #target["iscrowd"] = iscrowd

        if self.transforms is not None:
            # img, target= self.transforms(img, target)

            img = self.transforms(img)
            mask = self.transforms(mask)
            grid_encode = self.transforms(grid_encode)

            #t = torch.Tensor([0.5])  # threshold
            #mask = (mask > t).float() * 1
            # print("transform applited..!")
            # print("Image:", img.type)
            #print("mask:", target["masks"].type)
            # print("boxes;", target["boxes"].type)
            # print("image id:", target["image_id"].type)
            #print("labels;", target["labels"].type)
            # print("iscrowd:",target["iscrowd"].type)

            # print(img.type)
        return {"img":img, "grid_encode": grid_encode, "mask":mask}



class PolypsDatasetWithGridEncoding_TestData(object):

    def __init__(self, imgs_root, grid_sizes=[256] , transforms = None):

        self.imgs_root = imgs_root
        #self.masks_root = masks_root

        self.transforms = transforms

        self.length_of_grid_sizes = len(grid_sizes)

        self.imgs = list(sorted(os.listdir(self.imgs_root)))
        #self.masks = list(sorted(os.listdir(self.masks_root)))

        self.num_imgs = len(self.imgs)

        self.imgs = self.imgs * self.length_of_grid_sizes
       # self.masks = self.masks * self.length_of_grid_sizes

        self.grid_sizes_repeated = np.repeat(grid_sizes, self.num_imgs)

        self.all_in_one = list(zip(self.imgs, self.grid_sizes_repeated)) #(img, mask, grid_size)

       # self.imgs_repeated
        #self.masks_repea

    def __len__(self):
        return len(self.all_in_one)


    def __getitem__(self, idx):

        img_path = os.path.join(self.imgs_root, self.all_in_one[idx][0]) # 0th one= image
        img_name = self.all_in_one[idx][0]
        #mask_path = os.path.join(self.masks_root, self.all_in_one[idx][1]) # 1st one = mask
        grid_size = self.all_in_one[idx][1] # 2nd one = grid size

        

        img = Image.open(img_path)
        #mask = Image.open(mask_path)

        grid_encode = generate_checkerboard(256, 256, grid_size)#[:, :, 0]
        #print("finished grid encode")
       # print()

        

        # resizing
        img = img.resize((256, 256), Image.NEAREST)

        img = np.array(img)

        if self.transforms is not None:
            # img, target= self.transforms(img, target)

            img = self.transforms(img)
            #mask = self.transforms(mask)
            grid_encode = self.transforms(grid_encode)
        

        return {"img":img, "grid_encode": grid_encode, "img_name": img_name}



# Grid Encoding with new data augmentation

class PolypsDatasetWithGridEncoding_withNewAug(object):

    def __init__(self, imgs_root, masks_root, grid_sizes=[2,4,8,16,32,64,128,256] , transforms = None):

        self.imgs_root = imgs_root
        self.masks_root = masks_root

        self.transforms = transforms

        self.length_of_grid_sizes = len(grid_sizes)

        self.imgs = list(sorted(os.listdir(self.imgs_root)))
        self.masks = list(sorted(os.listdir(self.masks_root)))

        self.num_imgs = len(self.imgs)

        self.imgs = self.imgs * self.length_of_grid_sizes
        self.masks = self.masks * self.length_of_grid_sizes

        self.grid_sizes_repeated = np.repeat(grid_sizes, self.num_imgs)

        self.all_in_one = list(zip(self.imgs, self.masks, self.grid_sizes_repeated)) #(img, mask, grid_size)


        # Initiating augmentation
        # augmentation pipeline for images
        self.aug_img = iaa.Sequential([
            iaa.Affine(rotate=(-20, 20), random_state=1),
            iaa.CoarseDropout(0.2, size_percent=0.05, random_state=2),
            iaa.AdditiveGaussianNoise(scale=0.2*255, random_state=3),
            #iaa.ElasticTransformation(alpha=10, sigma=1)
        ], random_state=4)

        # augmentation pipeline for segmentation maps - with coarse dropout, but without gaussian noise
        self.aug_mask = iaa.Sequential([
            iaa.Affine(rotate=(-20, 20), random_state=1),
            iaa.CoarseDropout(0.2, size_percent=0.05, random_state=2),
            #iaa.ElasticTransformation(alpha=10, sigma=1)
        ], random_state=4)

       # self.imgs_repeated
        #self.masks_repea

    def __len__(self):
        return len(self.all_in_one)


    def __getitem__(self, idx):

        img_path = os.path.join(self.imgs_root, self.all_in_one[idx][0]) # 0th one= image
        mask_path = os.path.join(self.masks_root, self.all_in_one[idx][1]) # 1st one = mask
        grid_size = self.all_in_one[idx][2] # 2nd one = grid size

        #print("img path=", img_path)
        #print("mask_pth", mask_path)
        #print("grid size=", grid_size)

        img = Image.open(img_path)
        mask = Image.open(mask_path)

        grid_encode = generate_checkerboard(256, 256, grid_size)#[:, :, 0]
        #print("finished grid encode")
       # print()

        

        # resizing
        img = img.resize((256, 256), Image.NEAREST)
        mask = mask.resize((256, 256), Image.NEAREST)

        # covert to numpy

        img = np.array(img)
        mask = np.array(mask) #



        # clean mask values (this is done to remove small values in mask images)
        mask = (mask > 128) * 255 # if mask value > 128, set it to 255 (this will remove 254, 253 values and 1,2 etc)

        #========================================
        # add new augmentations to img and mask
        #========================================
        # convert array to SegmentationMapsOnImage instance
        mask = mask.astype(np.int8)
        #print("mask type:", type(mask))
        segmap = SegmentationMapsOnImage(mask, shape=img.shape)

        # First, augment image.
        image_aug = self.aug_img(image=img)
        img = image_aug

        # Second, augment segmentation map.
        # We convert to uint8 as that dtype has usually best support and hence is safest to use.
        mask_arr_aug = self.aug_mask(image=segmap.get_arr().astype(np.uint8))
        mask_aug = SegmentationMapsOnImage(mask_arr_aug, shape=segmap.shape)

        mask = mask_aug.get_arr()


         # get tiled ground truth for the augmented mask
        mask = get_tiled_ground_truth(mask, grid_size)
        #mask = mask[:, :, 0]

        # To tensor
       # mask = np.array(mask)
        #mask = torch.from_numpy(mask)


        # TO tensor
        # img = np.asarray(img)
        # img = torch.from_numpy(img)

        # mask to BB
        #bb_img, boxes = mask_to_bb(mask)
        # convert everything into a torch.Tensor
        #boxes = torch.as_tensor(boxes, dtype=torch.float32)
        #image_id = torch.tensor([idx])
        
        #area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        #iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        #labels = torch.ones((len(boxes),), dtype=torch.int64)
        # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        #target = {}
        
        # target["bb_img"] = bb_img
        #target["boxes"] = boxes
        
        #target["labels"]  = labels
       # target["masks"] = mask
        #target["image_id"] = image_id
        #target["area"] = area
        #target["iscrowd"] = iscrowd

        if self.transforms is not None:
            # img, target= self.transforms(img, target)

            img = self.transforms(img)
            mask = self.transforms(mask)
            grid_encode = self.transforms(grid_encode)

            #t = torch.Tensor([0.5])  # threshold
            #mask = (mask > t).float() * 1
            # print("transform applited..!")
            # print("Image:", img.type)
            #print("mask:", target["masks"].type)
            # print("boxes;", target["boxes"].type)
            # print("image id:", target["image_id"].type)
            #print("labels;", target["labels"].type)
            # print("iscrowd:",target["iscrowd"].type)

            # print(img.type)
        return {"img":img, "grid_encode": grid_encode, "mask":mask}






if __name__ == "__main__":
    img_root = "/work/vajira/DATA/hyper_kvasir/data_new/segmented/data/segmented-images/images"
    mask_root = "/work/vajira/DATA/hyper_kvasir/data_new/segmented/data/segmented-images/masks"

    data_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            # transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

    dataset = PolypsDataset(img_root, mask_root, transforms=data_transforms)

    data = dataset[1]

    

    print("img shape=", data["img"].shape)
    print("mask shape=", data["mask"].shape)

    print("img=", data["img"])
    print("mask=", list(data["mask"].reshape(-1)))