from skimage import measure
import numpy as np

from skimage.util import map_array


class NMSLabel(object):


    def __init__(self, image, z_thresh = 2):
        self.image = image 
        self.z_thresh = z_thresh

    
           
    def supressregions(self):
        
        print('Supressing spurious regions, this can take some time')
        properties = measure.regionprops(self.image)
        Bbox = [prop.bbox for prop in properties] 
        Labels = [prop.label for prop in properties]
        
        
        self.originallabels = []
        self.newlabels = []
        for pos in (range(len(Labels))):
                        current_label = Labels[pos]
                        self.smallz(Bbox[pos], current_label)

        if len(self.originallabels) > 0:   
            print(np.asarray(self.newlabels),np.asarray(self.originallabels) )     
            relabeled = map_array(
                    self.image, np.asarray(self.originallabels), np.asarray(self.newlabels)
                ) 
        else:
            relabeled = self.image  
        return relabeled
        
        
        
    def smallz(self, box, label):
        
        ndim = len(self.image.shape)
        if ndim == 3:
            z = abs(box[2] - box[5])
            print(z, self.z_thresh, box)
            if z <= self.z_thresh:
                self.originallabels.append(label)
                self.newlabels.append(0)
            else:
                self.originallabels.append(label)
                self.newlabels.append(label)
                    
        
    