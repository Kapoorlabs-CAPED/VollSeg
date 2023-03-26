import os
from pathlib import Path 
from cellposeutils3D import prepare_images, prepare_masks, create_csv
from . import save_json

class CellPose3D(object):

    def __init__(self, base_dir, model_dir, model_name, patch_size = (8,256,256), epochs = 100, in_channels = 1, out_channels = 4, feat_channels = 16,
                  samples_per_epoch = -1, batch_size = 16, learning_rate = 0.001, background_weight = 1, flow_weight = 1, 
                  raw_dir = 'raw/', real_mask_dir = 'real_mask/', identifier="*.tif",
                 save_train = '_train.csv', save_test = '_test.csv', save_val = '_val.csv',
    axis_norm = (0,1,2),
    variance_size=(5, 5, 5),
    fg_footprint_size=5,
    bg_label = 0,
    channel=0,
    corrupt_prob = 0,
    zoom_factor=(1, 1, 1)
    ):
        
        self.base_dir = base_dir 
        self.model_dir = model_dir
        self.model_name = model_name 
        self.patch_size = patch_size 
        self.epochs = epochs
        self.in_channels = in_channels
        self.out_channels = out_channels 
        self.feat_channels = feat_channels
        self.samples_per_epoch = samples_per_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.background_weight = background_weight
        self.flow_weight = flow_weight
        self.raw_dir = os.path.join(base_dir,raw_dir) 
        self.real_mask_dir = os.path.join(base_dir,real_mask_dir) 
        self.identifier = identifier
        self.save_train = save_train
        self.save_test = save_test 
        self.save_val = save_val  
        self.axis_norm = axis_norm 
        self.variance_size = variance_size
        self.fg_footprint_size = fg_footprint_size
        self.channel = channel
        self.bg_label = bg_label 
        self.corrupt_prob = corrupt_prob
        self.zoom_factor = zoom_factor
        self.save_raw_h5_name = 'raw_h5/'
        self.save_real_mask_h5_name = 'real_mask_h5/'

        self.save_raw_h5 = os.path.join(base_dir, self.save_raw_h5_name)
        Path(self.save_raw_h5).mkdir(exist_ok=True)

        self.save_real_mask_h5 = os.path.join(base_dir,self.save_real_mask_h5_name)
        Path(self.save_real_mask_h5).mkdir(exist_ok=True)


    def _create_training_h5(self):

        prepare_images(
                data_path=self.raw_dir,
                save_path=self.save_raw_h5,
                identifier=self.identifier,
                axis_norm = self.axis_norm,
                get_distance=False,
                get_illumination=False,
                get_variance=False,
                variance_size=self.variance_size,
                fg_footprint_size=self.fg_footprint_size,
                channel=self.channel
            )
        
        prepare_masks(
                data_path=self.real_mask_dir,
                save_path=self.save_real_mask_h5,
                identifier=self.identifier,
                bg_label=self.bg_label,
                get_flows=True,
                get_boundary=False,
                get_seeds=False,
                get_distance=True,
                corrupt_prob=self.corrupt_prob,
                zoom_factor= self.zoom_factor
            )
        data_list = []
        for imagename in os.listdir(self.save_raw_h5):
              data_list.append([os.path.join(self.save_raw_h5, imagename), os.path.join(self.save_real_mask_h5, imagename)])
        create_csv(data_list, self.base_dir,save_train = self.save_train, save_test = self.save_test, save_val = self.save_val)
        self.train_list = os.path.join(self.base_dir, self.save_train)
        self.val_list = os.path.join(self.base_dir, self.save_val)
        self.test_list = os.path.join(self.base_dir, self.save_test)
        hparams = {
                'train_list' : self.train_list,
                'test_list' : self.test_list,
                'val_list' : self.val_list,
                'data_root' : '',
                'patch_size' : self.patch_size,
                'epochs' : self.epochs,
                'image_groups': ('data/image'),
                'mask_groups':('data/distance', 'data/seeds', 'data/boundary'),
                'dist_handling':'bool_inv',
                'in_channels': self.in_channels,
                'out_channels': self.out_channels,
                'feat_channels':self.feat_channels,
                'norm_method':'instance',
                'samples_per_epoch' : self.samples_per_epoch,
                'batch_size' : self.batch_size,
                'learning_rate' : self.learning_rate,
                'background_weight' : self.background_weight,
                'flow_weight' : self.flow_weight,


               

        }

        save_json(hparams, str(self.base_dir) + '/' + self.model_name + '.json')


        
