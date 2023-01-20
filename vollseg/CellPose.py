import os
import numpy as np
from tifffile import imread
from cellpose import models, metrics
import concurrent

class CellPose(object):

    def __init__(self, base_dir, model_name, model_dir, raw_dir, real_mask_dir, test_raw_dir, test_real_mask_dir, 
                 n_epochs = 400, learning_rate = 0.0001, nimg_per_epochs=10, weight_decay = 1.0E-4,
                 cellpose_model_name = None, pretrained_cellpose_model_path = None, gpu = True, real_train_3D = False):
        
        self.base_dir = base_dir
        self.model_name = model_name 
        self.model_dir = model_dir 
        self.raw_dir = os.path.join(base_dir,raw_dir)
        self.real_mask_dir = os.path.join(base_dir,real_mask_dir) 
        self.test_raw_dir = os.path.join(base_dir,test_raw_dir)
        self.test_real_mask_dir = os.path.join(base_dir,test_real_mask_dir) 
        self.n_epochs = n_epochs 
        self.learning_rate = learning_rate 
        self.nimg_per_epochs = nimg_per_epochs
        self.weight_decay = weight_decay
        self.cellpose_model_name = cellpose_model_name 
        self.real_train_3D = real_train_3D
        self.pretrained_cellpose_model_path = pretrained_cellpose_model_path
        self.gpu = gpu
        self.acceptable_formats = [".tif", ".TIFF", ".TIF", ".png"]

        self.train()
        self.evaluate()

    def train(self):

        files_labels = os.listdir(self.real_mask_dir)
        self.train_images, self.train_labels, self.train_names = self.load_data(files_labels)

        files_test_labels = os.listdir(self.test_real_mask_dir)
        self.test_images, self.test_labels, self.test_names = self.load_data(files_test_labels) 
        
        if self.cellpose_model_name is not None:
             self.cellpose_model = models.Cellpose(gpu = self.gpu, model_type = self.cellpose_model_name)
        else:
             self.cellpose_model = models.CellposeModel(gpu = self.gpu, pretrained_model = self.pretrained_cellpose_model_path)      
        
        self.new_cellpose_model_path = self.cellpose_model.train(self.train_images, self.train_labels, test_data = self.test_images, test_labels = self.test_labels,
                                                       save_path = self.model_dir, n_epochs = self.n_epochs, learning_rate = self.learning_rate,
                                                       weight_decay = self.weight_decay, nimg_per_epoch = self.nimg_per_epochs, model_name = self.model_name)
        self.diam_labels = self.cellpose_model.diam_labels.copy()

    def evaluate(self):
         
         self.masks = self.cellpose_model.eval(self.test_images, diameter = self.diam_labels)[0]
         ap = metrics.average_precision(self.test_labels, self.masks)[0]
         print('')
         print(f'>>> average precision at iou threshold 0.5 = {ap[:,0].mean():.3f}')



    def load_data(self, files_labels):

        images = []
        labels = []
        names = []
        for fname in files_labels:
                if any(fname.endswith(f) for f in self.acceptable_formats):
                    name = os.path.splitext(fname)[0]
                    labelimage = imread(os.path.join(self.real_mask_dir,fname)).astype(np.uint16) 
                    image = imread(os.path.join(self.raw_dir,fname)) 
                    if not self.real_train_3D:
                            future_labels = []
                            future_raw = []
                            with concurrent.futures.ThreadPoolExecutor(max_workers = os.cpu_count() - 1) as executor: 
                                 future_labels.append(executor.submit(slicer, labelimage, i) for i in range(labelimage.shape[0]))
                                 future_raw.append(executor.submit(slicer, image, i) for i in range(image.shape[0]))
                            current_labels = [r.result() for r in concurrent.futures.as_completed(future_labels)]   
                            current_raw = [r.result() for r in concurrent.futures.as_completed(future_raw)]  
                            for i in range(len(current_labels)):
                                 labels.append(current_labels[i])
                                 current_name = name + str(i)
                                 names.append(current_name)
                            for raw in current_raw:
                                 images.append(raw)
                    else:
                         labels.append(labelimage)
                         images.append(image)  
                         names.append(name)

        return images, labels, names                 


def slicer(image, i):
     
     return image[i,:]
     
     
                
                 
