from skimage import measure
import numpy as np
import vollseg.utils
class NMSLabel(object):

    def __init__(self, image, nms_thresh):
        self.image = image 
        self.nms_thresh = nms_thresh

    def supresslabels(self):
        properties = measure.regionprops(self.image)
        Bbox = [prop.bbox for prop in properties] 
        Labels = [prop.label for prop in properties]
        self.supresslabel = {}
        while len(Labels) > 0:
                last = len(Labels) - 1
                i = Labels[last]
                suppress = [last] 
                for pos in (range(0, last)):
                    # grab the current index
                        j = Labels[pos]
                        self.iou(Bbox[last], Bbox[pos], i, j)

                Labels = np.delete(Labels, suppress)
                
        for (k,v) in self.supresslabel.items():
                pixel_condition = (self.image == k)
                pixel_replace_condition = v
                self.image = vollseg.utils.image_conditionals(self.image,pixel_condition,pixel_replace_condition )

        return self.image       
    def iou(self, boxA, boxB, labelA, labelB):

        ndim = len(self.image.shape)
        
        if ndim == 2:
                xA = max(boxA[0], boxB[0])
                yA = max(boxA[1], boxB[1])
                xB = min(boxA[2], boxB[2])
                yB = min(boxA[3], boxB[3])

                #BoxA contains BoxB
                if boxA[0] <= boxB[0] and boxA[2] >= boxB[2] and boxA[1] <= boxB[1] and boxA[3] >= boxB[3]: 
                    self.supresslabel[labelB] = labelA
                #BoxB contains BoxA
                if boxB[0] <= boxA[0] and boxB[2] >= boxA[2] and boxB[1] <= boxA[1] and boxB[3] >= boxA[3]: 
                    self.supresslabel[labelA] = labelB    
                else:    
                    # compute the area of intersection rectangle
                    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
                    # compute the area of both the prediction and ground-truth
                    # rectangles
                    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
                    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
                    # compute the intersection over union by taking the intersection
                    # area and dividing it by the sum of prediction + ground-truth
                    # areas - the interesection area
                    iou = interArea / float(boxAArea + boxBArea - interArea)
                    if iou >= self.nms_thresh:
                        self.supresslabel[labelA] = labelB

        if ndim == 3:

                xA = max(boxA[0], boxB[0])
                yA = max(boxA[1], boxB[1])
                zA = max(boxA[2], boxB[2])
                xB = min(boxA[3], boxB[3])
                yB = min(boxA[4], boxB[4])
                zB = min(boxA[5], boxB[5])
                if boxA[0] <= boxB[0] and boxA[3] >= boxB[3] and boxA[1] <= boxB[1] and boxA[4] >= boxB[4] and boxA[2] <= boxB[2] and boxA[5] >= boxB[5]: 
                    self.supresslabel[labelB] = labelA
                elif boxB[0] <= boxA[0] and boxB[3] >= boxA[3] and boxB[1] <= boxA[1] and boxB[4] >= boxA[4] and boxB[2] <= boxA[2] and boxB[5] >= boxA[5]:
                    self.supresslabel[labelA] = labelB
                    
                else:    
                    # compute the area of intersection rectangle
                    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1) * max(0, zB - zA + 1)
                    # compute the area of both the prediction and ground-truth
                    # rectangles
                    boxAArea = (boxA[3] - boxA[0] + 1) * (boxA[4] - boxA[1] + 1) * (boxA[5] - boxA[2] + 1)
                    boxBArea = (boxB[3] - boxB[0] + 1) * (boxB[4] - boxB[1] + 1) * (boxB[5] - boxB[2] + 1)
                    # compute the intersection over union by taking the intersection
                    # area and dividing it by the sum of prediction + ground-truth
                    # areas - the interesection area
                    iou = interArea / float(boxAArea + boxBArea - interArea)
                    
                    if iou >= self.nms_thresh:
                        self.supresslabel[labelA] = labelB