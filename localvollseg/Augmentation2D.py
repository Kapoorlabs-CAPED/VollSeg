#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 19:09:26 2020

@author: kapoorlab
"""

from tqdm import tqdm
from glob import glob
from tifffile import imread, imwrite
import numpy as np
import scipy
import math
import random
from scipy.ndimage.interpolation import shift, zoom
from scipy.ndimage import rotate
from scipy import ndimage
from pathlib import Path
    
    
    
class Augmentation2D(object):



    """
    Data generator of 2D voxel data.
    This generator is for data augmentation(flip, shift...).
    Note:
        Only one type of augmentation can be applied for one generator.
    """
    def __init__(self,
                 flip_axis=None,
                 shift_axis=None,
                 shift_range=None,
                 zoom_axis=None,
                 zoom_range=None,
                 rotate_axis=None,
                 rotate_angle=None
                 ):
        """
        Arguments:
         flip_axis: int(1, 2 ) or 'random'
                Integers 1, 2 mean x axis, y axis for each.
                Axis along which data is flipped.
         shift_axis: int(1, 2 ) or 'random'
                Integers 1, 2 mean x axis, y axis for each.
                Axis along which data is shifted.
         shift_range: float([-1, 1]) or 'random'
                Rate with which data is shifted along the specified axis.
                Positive value means data is shifted towards the positive direction.
                Negative value means the negative direction as well.
         zoom_axis: int(1, 2 ), 'random' or 'same'.
                Integers 1, 2 mean x axis, y axis for each.
                Axis along which data is zoomed. 'same' means the same zoom_range is applied for all axis
         zoom_range: float(>= 0) or 'random'
                Magnification with which data is zoomed along the specified axis.
                Value more than 1 means data is expanded and value less than means data is shrunk.
         rotate_axis: int(1, 2 ) or 'random'
                Integers 1, 2  mean x axis, y axis for each.
                Axis along which data is rotated.
         rotate_angle: int or 'random'
                Angle by which data is rotated along the specified axis.
        """
        self.flip_axis = flip_axis
        self.shift_axis = shift_axis
        self.shift_range = shift_range
        self.zoom_axis = zoom_axis
        self.zoom_range = zoom_range
        self.rotate_axis = rotate_axis
        self.rotate_angle = rotate_angle

    def build(self,
              data=None,
              label=None,
              batch_size=None):
        """
        Arguments:
        build generator to augment input data according to initialization
        data: array
            Input data to be augmented.
            The shape of input data should have 3 dimension(batch, x, y).
        label : Integer label images
            The shape of this labels should match the shape of data (batch, x, y).
        batch_size: int
            The size of data to generate at one batch.
        Return:
            generator
        """
        if data.ndim != 3:
            raise ValueError('Input data should have 3 dimensions.')

       
        if data.ndim != label.ndim:
                raise ValueError('Input data and label size do not much.')

        self.data = data
        self.label = label
        self.batch_size = batch_size
        self.data_dim = data.ndim
        self.data_shape = data.shape
        self.data_size = self.data_shape[0]
        self.idx_list = None

        parse_dict = {}
        callback = None

        # flip
        if self.flip_axis is not None:
            callback = self._flip_data
            if self.flip_axis in (1, 2):
                parse_dict['flip_axis'] = self.flip_axis
            elif self.flip_axis == 'random':
                parse_dict['flip_axis'] = random.randint(1, 2)
            else:
                raise ValueError('Flip axis should be 1, 2 or random')

        # shift
        if (self.shift_axis is not None) and (self.shift_range is not None):
            callback = self._shift_data
            if self.shift_axis in (1, 2):
                parse_dict['shift_axis'] = self.shift_axis
            elif self.shift_axis == 'random':
                parse_dict['shift_axis'] = random.randint(1, 2)
            else:
                raise ValueError('Shift axis should be 1, 2 or random')

            if not (type(self.shift_range) is str) and abs(self.shift_range) <= 1:
                parse_dict['shift_range'] = self.shift_range
            elif self.shift_range == 'random':
                parse_dict['shift_range'] = np.random.rand() - 0.5
            else:
                raise ValueError('Shift range should be in range [-1, 1] or random')

        # zoom
        if(self.zoom_axis is not None) and (self.zoom_range is not None):
            callback = self._zoom_data
            if self.zoom_axis in (1, 2):
                parse_dict['zoom_axis'] = self.zoom_axis
            elif self.zoom_axis == 'random':
                parse_dict['zoom_axis'] = random.randint(1, 2)
            elif self.zoom_axis == 'same':
                parse_dict['zoom_axis'] = None
            else:
                raise ValueError('Zoom axis should be 1, 2 or random')

            if not (type(self.zoom_range) is str) and (type(self.zoom_range) in (int, float)):
                parse_dict['zoom_range'] = self.zoom_range
            elif self.zoom_range == 'random':
                parse_dict['zoom_range'] = np.random.uniform(0.25, 1) * 2
            else:
                raise ValueError('Zoom range should be type of int, float or random')

        # rotate
        if (self.rotate_axis is not None) and (self.rotate_angle is not None):
            callback = self._rotate_data
            if self.rotate_axis in (1, 2):
                parse_dict['rotate_axis'] = self.rotate_axis
            elif self.rotate_axis == 'random':
                parse_dict['rotate_axis'] = random.randint(1, 2)
            else:
                raise ValueError('Rotate axis should be 1, 2 or random')

            if self.rotate_angle == 'random':
                parse_dict['rotate_angle'] = int(np.random.uniform(-180, 180))
            elif type(self.rotate_angle) == int:
                parse_dict['rotate_angle'] = self.rotate_angle
            else:
                raise ValueError('Rotate angle should be int or random')

        # build and return generator with specified callback function
        if callback:
            return self._return_generator(callback, parse_dict)
        else:
            raise ValueError('No generator returned. Arguments are not set properly.')

    def _return_generator(self, callback, parse_dict):
        """return generator according to callback"""
        self.idx_list = [i for i in range(self.data_size)]
        np.random.shuffle(self.idx_list)
        rp_num = self.data_size // self.batch_size
        cnt = 0

        while True:
            target_idx = self.idx_list[cnt * self.batch_size: (cnt + 1) * self.batch_size]
            target_data = self.data[target_idx]
            target_label = self.label[target_idx]

            # data augmentation by callback function
            ret_data = np.array([callback(target_data[[i]], parse_dict)[0, ...] for i in range(self.batch_size)])
            ret_label =  np.array([callback(target_label[[i]], parse_dict)[0, ...] for i in range(self.batch_size)])
            
            if cnt < rp_num - 1:
                cnt += 1
            elif cnt == rp_num - 1:
                cnt = 0
                np.random.shuffle(self.idx_list)

                yield ret_data, ret_label

    def _flip_data(self, data, parse_dict):
        """flip array along specified axis(x, y)"""
        return np.flip(data, parse_dict['flip_axis'])

    def _shift_data(self, data, parse_dict):
        """shift array by specified range along specified axis(x, y)"""
        shift_lst = [0] * self.data_dim
        shift_lst[parse_dict['shift_axis']] = math.floor(
            parse_dict['shift_range'] * self.data_shape[parse_dict['shift_axis']])

        return shift(data, shift=shift_lst, cval=0)
    def _zoom_data(self, data, parse_dict):
        """zoom array by specified range along specified axis(x, y). After zoomed, the voxel size is the same as
        before"""
        # functions to calculate target range of arrays(outside of the target range is not used to zoom)
        # - d/2 <= zoom_range * (x - d/2) <= d/2
        f1 = lambda d: math.floor((d / 2) * (1 + 1 / parse_dict['zoom_range']))
        f2 = lambda d: math.ceil((d / 2) * (1 - 1 / parse_dict['zoom_range']))

        if parse_dict['zoom_range'] > 1.0:
            # expand
            z_win1 = list(map(f1, self.data_shape[1:]))
            z_win2 = list(map(f2, self.data_shape[1:]))

            if parse_dict['zoom_axis'] is None:
                # same for all axis
                target_data = data[:, z_win2[0]:z_win1[0], z_win2[1]:z_win1[1]]
            else:
                # only one axis
                if parse_dict['zoom_axis'] == 1:
                    target_data = data[:, z_win2[0]:z_win1[0], :]
                elif parse_dict['zoom_axis'] == 2:
                    target_data = data[:, :, z_win2[1]:z_win1[1]]
        else:
            # shrink
            target_data = data

        if parse_dict['zoom_axis'] is None:
            zoom_lst = [1] + [parse_dict['zoom_range']] * (self.data_dim - 1)
        else:
            zoom_lst = [1] * self.data_dim
            zoom_lst[parse_dict['zoom_axis']] = parse_dict['zoom_range']

        zoomed = zoom(target_data, zoom=zoom_lst, cval=0)
        temp = [[math.floor((i - j) / 2), math.ceil((i - j) / 2)] for i, j in zip(self.data_shape[1:], zoomed.shape[1:])]

        cast_zoomed = np.zeros(self.data_shape)
        cast_zoomed[:,
        temp[0][0]:self.data_shape[1] - temp[0][1],
        temp[1][0]:self.data_shape[2] - temp[1][1]]= zoomed

        return cast_zoomed

    def _rotate_data(self, data, parse_dict):
        """rotate array by specified range along specified axis(x, y or z)"""
        if parse_dict['rotate_axis'] == 1:
            ax_tup = (1, 2)
        elif parse_dict['rotate_axis'] == 2:
            ax_tup = (2, 1)
        else:
            raise ValueError('rotate axis should be 1, 2')

        return rotate(data, axes=ax_tup, angle=parse_dict['rotate_angle'], cval=0.0, reshape=False)