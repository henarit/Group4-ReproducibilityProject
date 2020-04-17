import numpy as np
import torch
from torchvision.utils import make_grid
from torchvision import transforms
import matplotlib
import matplotlib.pyplot as plt

#rois=np.load('/Users/arturo/Desktop/rois.npy',  mmap_mode=None)
#print(a)
rois=np.load('/Users/arturo/Downloads/GDC_download-master/tmp/Pancreas_PAAD_slide/raw/65c41111-288e-46ed-bb22-9beaec693480/TCGA-IB-AAUT-01Z-00-DX1.69FBD770-916E-47BC-9E02-A9F4AD9E3E40.svs.npy',  mmap_mode=None)
rois = torch.tensor(rois).permute((0, 3, 1, 2))[:, 0:3]
transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(64), transforms.ToTensor()])
rois = torch.stack([transform(x) for x in rois.cpu()])
print (rois.shape)
image = make_grid(rois, nrow=40)
print (image.shape)
plt.imsave("/Users/arturo/Desktop/rois.png", image.permute((1, 2, 0)).data.cpu().numpy())
