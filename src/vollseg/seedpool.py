class SeedPool:
    def __init__(self, boxA, cordB):

        self.boxA = boxA
        self.cordB = cordB

    def pooling(self):

        self.iou3D()

        return self.include

    def iou3D(self):

        self.ndim = len(self.cordB)

        self.include = False
        includelist = [self.Conditioncheck(p) for p in range(0, self.ndim)]
        if True in includelist:
            self.include = True

    def Conditioncheck(self, p):

        include = True

        if self.cordB[p] >= self.boxA[p] and self.cordB[p] <= self.boxA[p + self.ndim]:

            include = False

        return include
