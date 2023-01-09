from skimage import measure
import numpy as np
import vollseg.utils
class NMSLabel(object):

    def __init__(self, image, z_thresh = 2):
        self.image = image 
        self.z_thresh = z_thresh

    def supresslabels(self):
        
        print(f'Supressing spurious labels, this can take some time')
        properties = measure.regionprops(self.image)
        Bbox = [prop.bbox for prop in properties] 
        Labels = [prop.label for prop in properties]
        Centroids = [prop.centroid for prop in properties]
        Sizes = [prop.area for prop in properties]
        
        self.supresslabel = {}
        while len(Labels) > 0:
                last = len(Labels) - 1
                i = Labels[last]
                suppress = [last] 
                for pos in (range(0, last)):
                    # grab the current index
                        j = Labels[pos]
                        self.iou(Bbox[last], Bbox[pos], Centroids[last], Centroids[pos],Sizes[last], Sizes[pos],  i, j)

                Labels = np.delete(Labels, suppress)
                
        for (k,v) in self.supresslabel.items():
                pixel_condition = (self.image == k)
                pixel_replace_condition = v
                
                self.image = vollseg.utils.image_conditionals(self.image,pixel_condition,pixel_replace_condition )

        return self.image
           
    def supressregions(self):
        
        print('Supressing spurious regions, this can take some time')
        properties = measure.regionprops(self.image)
        Bbox = [prop.bbox for prop in properties] 
        Labels = [prop.label for prop in properties]
        
        
        self.supressregion = {}
        for pos in (range(len(Labels))):
                        current_label = Labels[pos]
                        self.smallz(Bbox[pos], current_label)

                
        for (k,v) in self.supressregion.items():
                pixel_condition = (self.image == k)
                pixel_replace_condition = v
                self.image = vollseg.utils.image_conditionals(self.image,pixel_condition,pixel_replace_condition )

        return self.image
        
        
        
    def smallz(self, box, label):
        
        ndim = len(self.image.shape)
        if ndim == 3:
            z = abs(box[2] - box[5])
            if z <= self.z_thresh:
                self.supressregion[label] = 0
        
    def iou(self, boxA, boxB, centroidA, centroidB, sizeA, sizeB, labelA, labelB):

        ndim = len(self.image.shape)
        
        if ndim == 2:
                
            if sizeA < sizeB:
                replace = []
                for p in range(ndim):
                     if centroidA[p] >= boxB[p] and centroidA[p] <= boxB[p + ndim]:
                            replace.append(True)
                     else:
                            replace.append(False)
                if all(replace):            
                        self.supresslabel[labelA] = labelB
            else:            
                replace = []
                for p in range(ndim):
                        if centroidB[p] >= boxA[p] and centroidB[p] <= boxA[p + ndim]:
                            replace.append(True)
                        else:
                            replace.append(False)
                if all(replace):            
                        self.supresslabel[labelB] = labelA

        if ndim == 3:

               
         
            if sizeA < sizeB:
                replace = []
                for p in range(1,ndim):
                     if centroidA[p] >= boxB[p] and centroidA[p] <= boxB[p + ndim]:
                            replace.append(True)
                     else:
                            replace.append(False)
                if all(replace):            
                        self.supresslabel[labelA] = labelB
            else:            
                replace = []
                for p in range(1,ndim):
                        if centroidB[p] >= boxA[p] and centroidB[p] <= boxA[p + ndim]:
                            replace.append(True)
                        else:
                            replace.append(False)
                if all(replace):            
                        self.supresslabel[labelB] = labelA    
                        print('good?')    