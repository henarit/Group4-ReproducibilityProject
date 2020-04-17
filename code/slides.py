
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import os, sys, random, yaml, shutil, time, glob
import IPython

import openslide
#OLD
#from scipy.misc import imresize

#NEW
from PIL import Image
import pickle
#END NEW

def open_slide(slide_image_path):
	return openslide.open_slide(slide_image_path)


def sample_from_slides(slides, window_size=400, view_size=100, num=10):
	
	samples = []
	while len(samples) < num:
		slide = random.choice(slides)
		XMAX, YMAX = slide.dimensions[0], slide.dimensions[1]

		xv, yv = random.randint(0, XMAX - window_size), random.randint(0, YMAX - window_size)
		window = np.array(slide.read_region((xv, yv), 0, (window_size, window_size)))
		#plt.imshow(window); plt.show()
		if np.array(window).mean() > 200:
			continue

		if np.array(window[:, :, 0]).mean() < 50:
			continue

		if np.array(window[:, :, 2]).mean() > 160:
			continue
		#OLD
		#window = imresize(window, (view_size, view_size, 4))
                #window= np.array(Image.fromarray(window).resize(view_size, view_size, 4))

		#NEW
		img = Image.fromarray(window)
		window = img.resize(size = (view_size, view_size))
		window = np.array(window)
		#END NEW
		
                #window.resize((view_size, view_size, 4))
		#plt.imshow(window); plt.show()
		samples.append(window)
		

	return np.array(samples)

def sample_from_patient(case, window_size=400, view_size=100, num=10):
        #OLD
        #slide_files = glob.glob(f"/Volumes/Seagate Backup Plus Drive/tissue-slides/{case}*.svs")

        #NEW
        path="/Users/arturo/Downloads/GDC_download-master/tmp/Pancreas_PAAD_slide/raw" #specific for each computer
        slide_files = [os.path.join(path,case)]
        #slide_files = os.path.join(path,case)
        #print(slide_files)
        #for file in slide_files:
        #        print(file)
        #END NEW
        
        slides = [open_slide(file) for file in slide_files]
        if len(slides) == 0: return None
        return sample_from_slides(slides, window_size=window_size, view_size=view_size, num=num)

if __name__ == "__main__":

        #OLD
        #data = yaml.load(open(f"data/processed/case_files_locs.yaml"))
	#cases = data.keys()

        #NEW
        cases =[]
        path="/Users/arturo/Downloads/GDC_download-master/tmp/Pancreas_PAAD_slide/raw"

        data=os.listdir(path)  #This contains all the names of the folders within raw
        #print(data)
        data.remove("metadata_Pancreas_slide")
        data.remove("query_results_Pancreas_slide")
        data.remove(".DS_Store")
        for i in data:
                a=os.path.join(path,i)
                #print(a)
                file=os.listdir(a)
                #print(file)
                #file.remove(".DS_Store")
                #print(file)

                if file[0].endswith(".svs") or file[0].endswith(".svs.partial"):
                        cases.append(os.path.join(a,file[0]))
                if file[1].endswith(".svs") or file[1].endswith(".svs.partial"):
                        cases.append(os.path.join(a,file[1]))
        #print (cases)
        #END NEW

        #OLD
        #rois = [sample_from_patient(case, num=40) for case in cases]
        #print(len(rois))
        #rois=[roi_set for roi_set in rois if roi_set is not None]
        #rois= np.array([roi for roi_set in rois for roi in roi_set])
        #print(rois)

        #NEW
        for j in cases:
                rois=[sample_from_patient(j, num=40)]
                print(len(rois))
                rois=[roi_set for roi_set in rois if roi_set is not None]
                rois= np.array([roi for roi_set in rois for roi in roi_set])
                print(rois)
                np.save(j, rois)
                print(j)
        #END NEW
        
        
        #import torch
        #from torchvision.utils import make_grid
        #from torchvision import transforms


        #OLD
        #
	#print (rois.shape)
	#rois = torch.tensor(rois).permute((0, 3, 1, 2))[:, 0:3]
	#print (rois.shape)
	#transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize(64),transforms.ToTensor()])
	#rois = torch.stack([transform(x) for x in rois.cpu()])
	#print (rois.shape)	
	#image = make_grid(rois, nrow=4)
	#print (image.shape)
	#plt.imsave("results/rois.png", image.permute((1, 2, 0)).data.cpu().numpy())
	





        #NEW    
        #print(rois.shape)
        #np.save('rois', rois)
        #rois = torch.tensor(rois).permute((0, 3, 1, 2))[:, 0:3]
        #transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(64), transforms.ToTensor()])
        #rois = torch.stack([transform(x) for x in rois.cpu()])
        #print (rois.shape)

        #image = make_grid(rois, nrow=40)
        #print (image.shape)
        #plt.imsave("/Users/arturo/Downloads/GDC_download-master/tmp/results/rois.png", image.permute((1, 2, 0)).data.cpu().numpy())
        #np.save('rois', rois)
        #END NEW
