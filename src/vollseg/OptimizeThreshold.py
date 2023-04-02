#!/usr/bin/env python2
"""
Created on Mon Mar  8 16:22:00 2021

@author: vkapoor
"""
import numpy as np
from tqdm import tqdm
from csbdeep.utils import save_json
from csbdeep.utils import normalize
from stardist.matching import matching_dataset
from scipy.optimize import minimize_scalar
from csbdeep.utils import _raise
from .utils import VollSeg


class OptimizeThreshold:
    def __init__(
        self,
        X,
        Y,
        basedir,
        star_model,
        UseProbability=True,
        unet_model=None,
        noise_model=None,
        seedpool=True,
        dounet=True,
        n_tiles=(1, 1, 1),
        min_size=20,
        nms_threshs=[0, 0.3, 0.4, 0.5],
        iou_threshs=[0.3, 0.5, 0.7],
        measure="accuracy",
        RGB=False,
        axes="ZYX",
    ):

        self.star_model = star_model
        self.unet_model = unet_model
        self.noise_model = noise_model
        self.n_tiles = n_tiles
        self.basedir = basedir
        self.UseProbability = UseProbability
        self.X = X
        self.Y = Y
        self.RGB = RGB
        self.dounet = dounet
        self.seedpool = seedpool
        self.axes = axes
        self.nms_threshs = nms_threshs
        self.iou_threshs = iou_threshs
        self.measure = measure

        self.min_size = min_size

        self.Optimize()

    def Optimize(self):

        self.Y = [y for y in tqdm(self.Y)]
        self.X = [normalize(x, 1, 99.8, axis=(0, 1)) for x in tqdm(self.X)]

        print("Images to apply prediction on", len(self.X))

        (
            opt_prob_thresh_voll,
            opt_measure_voll,
            opt_nms_thresh,
            opt_prob_thresh_star,
            _,
        ) = (None, -np.inf, None, None, -np.inf)
        for _opt_nms_thresh in self.nms_threshs:
            (
                _opt_prob_thresh_voll,
                _opt_measure_voll,
                _opt_prob_thresh_star,
                _opt_measure_star,
            ) = self.optimize_threshold(self.Y, nms_thresh=_opt_nms_thresh)
            if _opt_measure_voll > opt_measure_voll:
                (
                    opt_prob_thresh_voll,
                    opt_measure_voll,
                    opt_nms_thresh,
                    opt_prob_thresh_star,
                    _,
                ) = (
                    _opt_prob_thresh_voll,
                    _opt_measure_voll,
                    _opt_nms_thresh,
                    _opt_prob_thresh_star,
                    _opt_measure_star,
                )
        opt_threshs_voll = dict(prob=opt_prob_thresh_voll, nms=opt_nms_thresh)
        opt_threshs_star = dict(prob=opt_prob_thresh_star, nms=opt_nms_thresh)

        self.thresholds_voll = opt_threshs_voll
        self.thresholds_star = opt_threshs_star
        print(
            "Using optimized values for vollseg: prob_thresh={prob:g}, nms_thresh={nms:g}.",
            opt_threshs_voll,
        )
        if self.basedir is not None:
            print("Saving to 'thresholds_voll.json'.")
            save_json(
                opt_threshs_voll, str(self.basedir + "/" + "thresholds_voll.json")
            )
        print(
            "Using optimized values for stardist: prob_thresh={prob:g}, nms_thresh={nms:g}.",
            opt_threshs_star,
        )
        if self.basedir is not None:
            print("Saving to 'thresholds_star.json'.")
            save_json(
                opt_threshs_star, str(self.basedir + "/" + "thresholds_star.json")
            )

        return opt_threshs_voll, opt_threshs_star

    def optimize_threshold(
        self, Y, nms_thresh, measure="accuracy", tol=1e-2, maxiter=20, verbose=1
    ):
        """Tune prob_thresh for provided (fixed) nms_thresh to maximize matching score (for given measure and averaged over iou_threshs)."""
        np.isscalar(nms_thresh) or _raise(ValueError("nms_thresh must be a scalar"))
        self.iou_threshs = (
            [self.iou_threshs] if np.isscalar(self.iou_threshs) else self.iou_threshs
        )
        values_voll = dict()
        values_star = dict()
        with tqdm(
            total=maxiter,
            disable=(verbose != 1),
            desc="NMS threshold = %g" % nms_thresh,
        ) as progress:

            def fn(thr):
                prob_thresh = thr
                value_voll = values_voll.get(prob_thresh)
                value_star = values_star.get(prob_thresh)
                if value_voll is None:
                    res = tuple(
                        zip(
                            *tuple(
                                VollSeg(
                                    x,
                                    unet_model=self.unet_model,
                                    star_model=self.star_model,
                                    axes=self.axes,
                                    noise_model=self.noise_model,
                                    n_tiles=self.n_tiles,
                                    UseProbability=self.UseProbability,
                                    dounet=self.dounet,
                                    seedpool=self.seedpool,
                                    RGB=self.RGB,
                                )
                                for x in tqdm(self.X)
                            )
                        )
                    )

                    if self.noise_model is None and self.star_model is not None:
                        (
                            Sizedsmart_seeds,
                            SizedMask,
                            star_labels,
                            proabability_map,
                            Markers,
                            Skeleton,
                        ) = res

                    elif self.noise_model is not None and self.star_model is not None:
                        (
                            Sizedsmart_seeds,
                            SizedMask,
                            star_labels,
                            proabability_map,
                            Markers,
                            Skeleton,
                            image,
                        ) = res

                    elif self.star_model is None:

                        raise ValueError(
                            f"StarDist model can not be {self.star_model} for evaluating optimized threshold"
                        )

                    stats_voll = matching_dataset(
                        Y,
                        Sizedsmart_seeds,
                        thresh=self.iou_threshs,
                        show_progress=False,
                        parallel=True,
                    )
                    values_voll[prob_thresh] = value_voll = np.mean(
                        [s._asdict()[measure] for s in stats_voll]
                    )

                    stats_star = matching_dataset(
                        Y,
                        star_labels,
                        thresh=self.iou_threshs,
                        show_progress=False,
                        parallel=True,
                    )
                    values_star[prob_thresh] = value_star = np.mean(
                        [s._asdict()[measure] for s in stats_star]
                    )

                    progress.update()
                    progress.set_postfix_str(
                        f"VollSeg-StarDist, {prob_thresh:.3f} -> {value_voll:.3f}, {value_star:.3f}"
                    )

                    progress.refresh()

                return -value_voll, -value_star

            opt_voll, opt_star = minimize_scalar(
                fn, method="golden", tol=tol, options={"maxiter": maxiter}
            )

        return opt_voll.x, -opt_voll.fun, opt_star.x, -opt_star.fun


def _is_floatarray(x):
    return isinstance(x.dtype.type(0), np.floating)
