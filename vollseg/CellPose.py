from cellpose import models 


class CellPose(object):

    def __init__(self, base_dir, model_name, model_dir, raw_image_dir, real_mask_dir, test_raw_image_dir, test_real_mask_dir, 
                 n_epochs = 400, learning_rate = 0.0001, nimg_per_epochs=10, weight_decay = 1.0E-4,
                 cellpose_model_name = None, pretrained_cellpose_model_path = None):
        
        self.base_dir = base_dir
        self.model_name = model_name 
        self.model_dir = model_dir 
        self.raw_image_dir = raw_image_dir
        self.real_mask_dir = real_mask_dir 
        self.test_raw_image_dir = test_raw_image_dir
