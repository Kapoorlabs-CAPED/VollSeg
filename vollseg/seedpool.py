


class SeedPool(object):

    def __init__(self, boxA, boxB, nms_thresh):

        self.boxA = boxA 
        self.boxB = boxB
        self.nms_thresh = nms_thresh

    def pooling(self):
        
          self.inside = self.iou3D()

    def iou3D(self):

            self.ndim = len(self.boxB)//2
            
            self.inside = self.Conditioncheck()
           




    def Conditioncheck(self):

        condition = False

        if self.ndim == 2:
            xA = max(self.boxA[0], self.boxB[0])
            yA = max(self.boxA[1], self.boxB[1])
            xB = min(self.boxA[2], self.boxB[2])
            yB = min(self.boxA[3], self.boxB[3])

            if self.boxA[0] <= self.boxB[0] and self.boxA[2] >= self.boxB[2] and self.boxA[1] <= self.boxB[1] and self.boxA[3] >= self.boxB[3]: 
                condition = True
            else:    
                # compute the area of intersection rectangle
                interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
                # compute the area of both the prediction and ground-truth
                # rectangles
                boxAArea = (self.boxA[2] - self.boxA[0] + 1) * (self.boxA[3] - self.boxA[1] + 1)
                boxBArea = (self.boxB[2] - self.boxB[0] + 1) * (self.boxB[3] - self.boxB[1] + 1)
                # compute the intersection over union by taking the intersection
                # area and dividing it by the sum of prediction + ground-truth
                # areas - the interesection area
                iou = interArea / float(boxAArea + boxBArea - interArea)
                if iou >= self.nms_thresh:
                    condition = True

        if self.ndim == 3:

            xA = max(self.boxA[0], self.boxB[0])
            yA = max(self.boxA[1], self.boxB[1])
            zA = max(self.boxA[2], self.boxB[2])
            xB = min(self.boxA[3], self.boxB[3])
            yB = min(self.boxA[4], self.boxB[4])
            zB = min(self.boxA[5], self.boxB[5])
            if self.boxA[0] <= self.boxB[0] and self.boxA[3] >= self.boxB[3] and self.boxA[1] <= self.boxB[1] and self.boxA[4] >= self.boxB[4] and self.boxA[2] <= self.boxB[2] and self.boxA[5] >= self.boxB[5]: 
                condition = True
            elif self.boxB[0] <= self.boxA[0] and self.boxB[3] >= self.boxA[3] and self.boxB[1] <= self.boxA[1] and self.boxB[4] >= self.boxA[4] and self.boxB[2] <= self.boxA[2] and self.boxB[5] >= self.boxA[5]:
                condition = True
                
            else:    
                # compute the area of intersection rectangle
                interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1) * max(0, zB - zA + 1)
                # compute the area of both the prediction and ground-truth
                # rectangles
                boxAArea = (self.boxA[3] - self.boxA[0] + 1) * (self.boxA[4] - self.boxA[1] + 1) * (self.boxA[5] - self.boxA[2] + 1)
                boxBArea = (self.boxB[3] - self.boxB[0] + 1) * (self.boxB[4] - self.boxB[1] + 1) * (self.boxB[5] - self.boxB[2] + 1)
                # compute the intersection over union by taking the intersection
                # area and dividing it by the sum of prediction + ground-truth
                # areas - the interesection area
                iou = interArea / float(boxAArea + boxBArea - interArea)
                
                if iou >= self.nms_thresh:
                    condition = True



        return condition    

    