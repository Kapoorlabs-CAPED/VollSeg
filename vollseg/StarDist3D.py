
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 19:34:47 2022
@author: varunkapoor
"""
from stardist.models import StarDist3D
import warnings
from .pretrained import get_registered_models, get_model_details, get_model_instance
import sys

class StarDist3D(StarDist3D):
     def __init__(self, config, name=None, basedir='.'):
        super().__init__(config=config, name=name, basedir=basedir)  
     @classmethod   
     def local_from_pretrained(cls, name_or_alias=None):
           try:
               get_model_details(cls, name_or_alias, verbose=True)
               return get_model_instance(cls, name_or_alias)
           except ValueError:
               if name_or_alias is not None:
                   print("Could not find model with name or alias '%s'" % (name_or_alias), file=sys.stderr)
                   sys.stderr.flush()
               get_registered_models(cls, verbose=True)
  
    
     def predict_vollseg(self, img, axes=None, normalizer=None,
                          
                          prob_thresh=None, nms_thresh=None,
                          n_tiles=None, show_tile_progress=True,
                          verbose=False,
                          return_labels=True,
                          predict_kwargs=None, nms_kwargs=None,
                          overlap_label=None, return_predict=False):
        """Predict instance segmentation from input image.
        Parameters
        ----------
        img : :class:`numpy.ndarray`
            Input image
        axes : str or None
            Axes of the input ``img``.
            ``None`` denotes that axes of img are the same as denoted in the config.
        normalizer : :class:`csbdeep.data.Normalizer` or None
            (Optional) normalization of input image before prediction.
            Note that the default (``None``) assumes ``img`` to be already normalized.
        sparse: bool
            If true, aggregate probabilities/distances sparsely during tiled
            prediction to save memory (recommended).
        prob_thresh : float or None
            Consider only object candidates from pixels with predicted object probability
            above this threshold (also see `optimize_thresholds`).
        nms_thresh : float or None
            Perform non-maximum suppression that considers two objects to be the same
            when their area/surface overlap exceeds this threshold (also see `optimize_thresholds`).
        n_tiles : iterable or None
            Out of memory (OOM) errors can occur if the input image is too large.
            To avoid this problem, the input image is broken up into (overlapping) tiles
            that are processed independently and re-assembled.
            This parameter denotes a tuple of the number of tiles for every image axis (see ``axes``).
            ``None`` denotes that no tiling should be used.
        show_tile_progress: bool
            Whether to show progress during tiled prediction.
        verbose: bool
            Whether to print some info messages.
        return_labels: bool
            Whether to create a label image, otherwise return None in its place.
        predict_kwargs: dict
            Keyword arguments for ``predict`` function of Keras model.
        nms_kwargs: dict
            Keyword arguments for non-maximum suppression.
        overlap_label: scalar or None
            if not None, label the regions where polygons overlap with that value
        return_predict: bool
            Also return the outputs of :func:`predict` (in a separate tuple)
            If True, implies sparse = False
        Returns
        -------
        (:class:`numpy.ndarray`, dict), (optional: return tuple of :func:`predict`)
            Returns a tuple of the label instances image and also
            a dictionary with the details (coordinates, etc.) of all remaining polygons/polyhedra.
        """
        if predict_kwargs is None:
            predict_kwargs = {}
        if nms_kwargs is None:
            nms_kwargs = {}

        
        nms_kwargs.setdefault("verbose", verbose)

        _axes         = self._normalize_axes(img, axes)
        _axes_net     = self.config.axes
        _permute_axes = self._make_permute_axes(_axes, _axes_net)
        _shape_inst   = tuple(s for s,a in zip(_permute_axes(img).shape, _axes_net) if a != 'C')

        res = self.predict(img, axes=axes, normalizer=normalizer, n_tiles=n_tiles,
                                       show_tile_progress=show_tile_progress, **predict_kwargs)
       
        res = tuple(res) + (None,)
        prob, dist, points = res
        prob_class = None

        res_instances, polys = self._instances_from_prediction(_shape_inst, prob, dist,
                                                        points=points,
                                                        prob_class=prob_class,
                                                        prob_thresh=prob_thresh,
                                                        nms_thresh=nms_thresh,
                                                        return_labels=return_labels,
                                                        overlap_label=overlap_label,
                                                        **nms_kwargs)

        return res_instances, prob, dist


     def predict(self, img, axes=None, normalizer=None, n_tiles=None, show_tile_progress=True, **predict_kwargs):
        """Predict.

        Parameters
        ----------
        img : :class:`numpy.ndarray`
            Input image
        axes : str or None
            Axes of the input ``img``.
            ``None`` denotes that axes of img are the same as denoted in the config.
        normalizer : :class:`csbdeep.data.Normalizer` or None
            (Optional) normalization of input image before prediction.
            Note that the default (``None``) assumes ``img`` to be already normalized.
        n_tiles : iterable or None
            Out of memory (OOM) errors can occur if the input image is too large.
            To avoid this problem, the input image is broken up into (overlapping) tiles
            that are processed independently and re-assembled.
            This parameter denotes a tuple of the number of tiles for every image axis (see ``axes``).
            ``None`` denotes that no tiling should be used.
        show_tile_progress: bool or callable
            If boolean, indicates whether to show progress (via tqdm) during tiled prediction.
            If callable, must be a drop-in replacement for tqdm.
        show_tile_progress: bool
            Whether to show progress during tiled prediction.
        predict_kwargs: dict
            Keyword arguments for ``predict`` function of Keras model.

        Returns
        -------
        (:class:`numpy.ndarray`, :class:`numpy.ndarray`, [:class:`numpy.ndarray`])
            Returns the tuple (`prob`, `dist`, [`prob_class`]) of per-pixel object probabilities and star-convex polygon/polyhedra distances.
            In multiclass prediction mode, `prob_class` is the probability map for each of the 1+'n_classes' classes (first class is background)

        """

        x, axes, axes_net, axes_net_div_by, _permute_axes, resizer, n_tiles, grid, grid_dict, channel, predict_direct, tiling_setup = \
            self._predict_setup(img, axes, normalizer, n_tiles, show_tile_progress, predict_kwargs)

        if np.prod(n_tiles) > 1:
            tile_generator, output_shape, create_empty_output = tiling_setup()

            prob = create_empty_output(1)
            dist = create_empty_output(self.config.n_rays)
            if self._is_multiclass():
                prob_class = create_empty_output(self.config.n_classes+1)
                result = (prob, dist, prob_class)
            else:
                result = (prob, dist)

            for tile, s_src, s_dst in tile_generator:
                # predict_direct -> prob, dist, [prob_class if multi_class]
                result_tile = predict_direct(tile)
                # account for grid
                s_src = [slice(s.start//grid_dict.get(a,1),s.stop//grid_dict.get(a,1)) for s,a in zip(s_src,axes_net)]
                s_dst = [slice(s.start//grid_dict.get(a,1),s.stop//grid_dict.get(a,1)) for s,a in zip(s_dst,axes_net)]
                # prob and dist have different channel dimensionality than image x
                s_src[channel] = slice(None)
                s_dst[channel] = slice(None)
                s_src, s_dst = tuple(s_src), tuple(s_dst)
                # print(s_src,s_dst)
                for part, part_tile in zip(result, result_tile):
                    part[s_dst] = part_tile[s_src]
        else:
            # predict_direct -> prob, dist, [prob_class if multi_class]
            result = predict_direct(x)

        result = [resizer.after(part, axes_net) for part in result]

        # result = (prob, dist) for legacy or (prob, dist, prob_class) for multiclass

        # prob
        result[0] = np.take(result[0],0,axis=channel)
        # dist
        result[1] = np.maximum(1e-3, result[1]) # avoid small dist values to prevent problems with Qhull
        result[1] = np.moveaxis(result[1],channel,-1)

        if self._is_multiclass():
            # prob_class
            result[2] = np.moveaxis(result[2],channel,-1)

        return tuple(result)



     def _predict_setup(self, img, axes, normalizer, n_tiles, show_tile_progress, predict_kwargs):
        """ Shared setup code between `predict` and `predict_sparse` """
        if n_tiles is None:
            n_tiles = [1]*img.ndim
        try:
            n_tiles = tuple(n_tiles)
            img.ndim == len(n_tiles) or _raise(TypeError())
        except TypeError:
            raise ValueError("n_tiles must be an iterable of length %d" % img.ndim)
        all(np.isscalar(t) and 1<=t and int(t)==t for t in n_tiles) or _raise(
            ValueError("all values of n_tiles must be integer values >= 1"))

        n_tiles = tuple(map(int,n_tiles))

        axes     = self._normalize_axes(img, axes)
        axes_net = self.config.axes

        _permute_axes = self._make_permute_axes(axes, axes_net)
        x = _permute_axes(img) # x has axes_net semantics

        channel = axes_dict(axes_net)['C']
        self.config.n_channel_in == x.shape[channel] or _raise(ValueError())
        axes_net_div_by = self._axes_div_by(axes_net)

        grid = tuple(self.config.grid)
        len(grid) == len(axes_net)-1 or _raise(ValueError())
        grid_dict = dict(zip(axes_net.replace('C',''),grid))

        normalizer = self._check_normalizer_resizer(normalizer, None)[0]
        resizer = StarDistPadAndCropResizer(grid=grid_dict)

        x = normalizer.before(x, axes_net)
        x = resizer.before(x, axes_net, axes_net_div_by)

        if not _is_floatarray(x):
            warnings.warn("Predicting on non-float input... ( forgot to normalize? )")

        def predict_direct(x):
            ys = self.keras_model.predict(x[np.newaxis], **predict_kwargs)
            return tuple(y[0] for y in ys)

        def tiling_setup():
            assert np.prod(n_tiles) > 1
            tiling_axes   = axes_net.replace('C','') # axes eligible for tiling
            x_tiling_axis = tuple(axes_dict(axes_net)[a] for a in tiling_axes) # numerical axis ids for x
            axes_net_tile_overlaps = self._axes_tile_overlap(axes_net)
            # hack: permute tiling axis in the same way as img -> x was permuted
            _n_tiles = _permute_axes(np.empty(n_tiles,np.bool)).shape
            (all(_n_tiles[i] == 1 for i in range(x.ndim) if i not in x_tiling_axis) or
                _raise(ValueError("entry of n_tiles > 1 only allowed for axes '%s'" % tiling_axes)))

            sh = [s//grid_dict.get(a,1) for a,s in zip(axes_net,x.shape)]
            sh[channel] = None
            def create_empty_output(n_channel, dtype=np.float16):
                sh[channel] = n_channel
                return np.empty(sh,dtype)

            if callable(show_tile_progress):
                progress, _show_tile_progress = show_tile_progress, True
            else:
                progress, _show_tile_progress = tqdm, show_tile_progress

            n_block_overlaps = [int(np.ceil(overlap/blocksize)) for overlap, blocksize
                                in zip(axes_net_tile_overlaps, axes_net_div_by)]

            num_tiles_used = total_n_tiles(x, _n_tiles, block_sizes=axes_net_div_by, n_block_overlaps=n_block_overlaps)

            tile_generator = progress(tile_iterator(x, _n_tiles, block_sizes=axes_net_div_by, n_block_overlaps=n_block_overlaps),
                                                    disable=(not _show_tile_progress), total=num_tiles_used)

            return tile_generator, tuple(sh), create_empty_output

        return x, axes, axes_net, axes_net_div_by, _permute_axes, resizer, n_tiles, grid, grid_dict, channel, predict_direct, tiling_setup