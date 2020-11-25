import numpy as np
import matplotlib.pyplot as plt

def generate_checkerboard(img_width, img_height, grid_size):
    ck = np.kron([[255, 0] * int(grid_size/2), [0, 255] * int(grid_size/2)]*int(grid_size/ 2) , np.ones((int(img_height / grid_size), int(img_width/ grid_size))))
    
    #ck = np.expand_dims(ck, axis=2)

    ck = ck.astype(np.uint8) # this is needed ot use ToTensor() transformaion in pytorch

    return ck



def get_tiled_ground_truth(mask, grid_size):
    height = mask.shape[0]
    width = mask.shape[1]
    tile_height = height / grid_size
    tile_width = width / grid_size
    mask = mask[:, : , 0] # get only 0th channel
    
    tile_mask = np.zeros_like(mask, dtype=np.uint8)

   # if grid_size == mask.shape[0]:
    #    return mask
    
    for c in range(grid_size):
        
        for r in range(grid_size):
        
            #mask = (mask > 128) # convert mask to binary
            #print(mask)

            row_start = int(r*tile_height)
            row_end = int(r*tile_height + tile_height)
            column_start =  int(c * tile_width) 
            column_end = int(c * tile_width + tile_width)
            
            #print("row start=", row_start)
            #print("row end=", row_end)
            #print("column start=", column_start)
            #print("colum end=", column_end)

            tile_sum = np.sum(mask[row_start:row_end, column_start:column_end])

            #print(tile_sum)  
            
            if tile_sum > 0:
                tile_mask[row_start:row_end, column_start:column_end] = 255
            
        #tile = 
    tile_mask = tile_mask.astype(np.uint8)
    #plt.imshow(tile_mask)
    return tile_mask