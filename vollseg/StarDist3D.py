#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 19:34:47 2022

@author: varunkapoor
"""
from stardist.models import StarDist3D

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
  