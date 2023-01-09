from skimage import measure
import numpy as np
import vollseg.utils
class NMSLabel(object):


    def __init__(self, boxA, cordB):

        self.boxA = boxA 
        self.cordB = cordB

    def merging(self):
        
          self.iou3D()

          return self.merge
         
    def iou3D(self):

            self.ndim = len(self.cordB)
            
            self.merge = False
            if self.ndim == 3:
               mergelist = [self.Conditioncheck(p) for p in range(1, self.ndim)]
            else:
               mergelist = [self.Conditioncheck(p) for p in range(0, self.ndim)]    
            if True in mergelist:
                 self.merge = True        
           
    def Conditioncheck(self, p):

        merge = False
        
        if self.cordB[p] >= self.boxA[p] and self.cordB[p] <= self.boxA[p + self.ndim]:

            merge = True


        return merge 









