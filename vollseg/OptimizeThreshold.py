#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 16:22:00 2021

@author: vkapoor
"""
from .helpers import SimplePrediction
import numpy as np
from tqdm import tqdm
from csbdeep.utils import save_json
from csbdeep.utils import normalize
from stardist.matching import matching_dataset
from scipy.optimize import minimize_scalar
import datetime
from csbdeep.utils import _raise
from skimage.measure import label

class OptimizeThreshold(object):
    
    
    def __init__(self, Starmodel, Unetmodel, X, Y, basedir, UseProbability = True, n_tiles = (4,4), min_size = 20, nms_threshs=[0,0.05,0.1,0.15,0.2,0.3,0.4], iou_threshs=[0.6, 0.65, 0.68], measure='accuracy', axis = 'ZYX'):
        
        
        self.Starmodel = Starmodel
        self.Unetmodel = Unetmodel
        self.n_tiles = n_tiles
        self.basedir = basedir
        self.UseProbability = UseProbability
        self.X = X
        self.Y = Y
        self.axis = axis
        self.nms_threshs = nms_threshs
        self.iou_threshs = iou_threshs
        self.measure = measure
        
        self.min_size = min_size
        
        self.Optimize()
        
    def Optimize(self):
   
        
            
                     
                
        self.Y = [label(y) for y in tqdm(self.Y)]
        self.X = [normalize(x,1,99.8,axis= (0,1)) for x in tqdm(self.X)]
        
        print('Images to apply prediction on',len(self.X))     
        Yhat_val = [self.Starmodel.predict(x) for x in self.X]
        
        opt_prob_thresh, opt_measure, opt_nms_thresh = None, -np.inf, None
        for _opt_nms_thresh in self.nms_threshs:
            _opt_prob_thresh, _opt_measure = self.optimize_threshold(self.Y, Yhat_val, model=self.Starmodel, nms_thresh=_opt_nms_thresh)
            if _opt_measure > opt_measure:
                opt_prob_thresh, opt_measure, opt_nms_thresh = _opt_prob_thresh, _opt_measure, _opt_nms_thresh
        opt_threshs = dict(prob=opt_prob_thresh, nms=opt_nms_thresh)

        self.thresholds = opt_threshs
        print("Using optimized values: prob_thresh={prob:g}, nms_thresh={nms:g}.", opt_threshs)
        if self.basedir is not None:
            print("Saving to 'thresholds.json'.")
            save_json(opt_threshs, str(self.basedir +  '/' + 'thresholds.json'))
        return opt_threshs
        
        

    def optimize_threshold(self, Y, Yhat, model, nms_thresh, measure='accuracy', bracket=None, tol=1e-2, maxiter=20, verbose=1):
                """ Tune prob_thresh for provided (fixed) nms_thresh to maximize matching score (for given measure and averaged over iou_threshs). """
                np.isscalar(nms_thresh) or _raise(ValueError("nms_thresh must be a scalar"))
                self.iou_threshs = [self.iou_threshs] if np.isscalar(self.iou_threshs) else self.iou_threshs
                values = dict()
            
                if bracket is None:
                    max_prob = max([np.max(prob) for prob, dist in Yhat])
                    bracket = max_prob/2, max_prob
                # print("bracket =", bracket)
            
                with tqdm(total=maxiter, disable=(verbose!=1), desc="NMS threshold = %g" % nms_thresh) as progress:
            
                    def fn(thr):
                        prob_thresh = np.clip(thr, *bracket)
                        value = values.get(prob_thresh)
                        if value is None:
                            
                            Y_instances = [SimplePrediction(x, self.Unetmodel, self.Starmodel, n_tiles = self.n_tiles, UseProbability = self.UseProbability, min_size = self.min_size, axis = self.axis) for x in tqdm(self.X)]
                            stats = matching_dataset(Y, Y_instances, thresh=self.iou_threshs, show_progress=False, parallel=True)
                            values[prob_thresh] = value = np.mean([s._asdict()[measure] for s in stats])
                        if verbose > 1:
                            print("{now}   thresh: {prob_thresh:f}   {measure}: {value:f}".format(
                                now = datetime.datetime.now().strftime('%H:%M:%S'),
                                prob_thresh = prob_thresh,
                                measure = measure,
                                value = value,
                            ))
                        else:
                            progress.update()
                            progress.set_postfix_str("{prob_thresh:.3f} -> {value:.3f}".format(prob_thresh=prob_thresh, value=value))
                            progress.refresh()
                        return -value
            
                    opt = minimize_scalar(fn, method='golden', bracket=bracket, tol=tol, options={'maxiter': maxiter})
            
                return opt.x, -opt.fun

def _is_floatarray(x):
    return isinstance(x.dtype.type(0),np.floating)



    
