import torch
import cv2
import numpy as np
from datasets.dataset import CUBDataset

cub = CUBDataset()

conv = torch.hub.load('pytorch/vision:v0.9.0', 'vgg19', pretrained=True)

img, label = cub[2]
img = torch.Tensor(img).unsqueeze(0)
print("image shape:", img.shape)



# show original image
base_img = np.array(img.squeeze(0).permute(1,2,0)).astype(np.uint8)
print("base image shape:", base_img.shape)
cv2.imshow("base", base_img)



# create pattern
patterns = conv.features(img)
print("patter shape:", patterns.shape) # 1 img; 512 pattern; breite/höhe



# (grayscale) pattern
img_pattern = patterns[0,0,:,:].detach().numpy().astype(np.uint8)
print("pattern single image shape:",img_pattern.shape)
#img_pattern = cv2.resize(img_pattern, [i*2 for i in img_pattern.shape])
#cv2.imshow("pattern", img_pattern)



# iterate through all patterns and concatenate all patterns to one single frame
square_size = 5
c_pattern = None
row_img = None

for row in range(square_size):
    if isinstance(c_pattern, np.ndarray):
        c_pattern = np.hstack((c_pattern, row_img))
    else:
        c_pattern = row_img
    row_img = None
    for col in range(square_size):
        index = 10 * row + col
        _img = patterns[0,index,:,:].detach().numpy().astype(np.uint8)
        _img = np.pad(_img, [(0, 1), (0, 1)], mode='constant', constant_values=(255))
        if isinstance(row_img, np.ndarray):
            row_img = np.vstack((row_img, _img))
        else:
            row_img = _img
            
print(c_pattern.shape)
c_pattern = cv2.resize(c_pattern, [i*4 for i in c_pattern.shape])
#cv2.imshow("pattern", c_pattern)



# mean of all patterns to get most interessting patter images
p = patterns[0,:,:,:].detach().numpy().astype(np.uint8)
total_mean = np.mean(p)

i = 0
store = list()
for i in range(p.shape[0]):
    if np.mean(p[i]) > total_mean:
        store.append(np.pad(p[i], [(0, 1), (0, 1)], mode='constant', constant_values=(255)))
print(len(store))

square = int(np.sqrt(len(store)))
c_pattern = None
row_img = None
for col in range(square):
    if isinstance(c_pattern, np.ndarray):
        c_pattern = np.hstack((c_pattern, row_img))
    else:
        c_pattern = row_img
    row_img = None
    for row in range(square):
        index = 10 * col + row
        if isinstance(row_img, np.ndarray):
            row_img = np.vstack((row_img, _img))
        else:
            row_img = store[index]

print(c_pattern.shape)
c_pattern = cv2.resize(c_pattern, [i*7 for i in c_pattern.shape])
cv2.imshow("pattern", c_pattern)
