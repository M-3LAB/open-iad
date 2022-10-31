import os
import numpy as np


__all__ = ['RecordHelper']

class RecordHelper():
    def __init__(self, config):
        self.config = config

    def update(self, config):
        self.config = config
    
    def print(self, info):
        pass

    def paradigm(self):
        if self.config['vanilla']:
            return 'vanilla'
        if self.config['semi']:
            return 'semi'
        if self.config['continual']:
            return 'continual'
        if self.config['fewshot']:
            return 'fewshot'
        
        return 'unknown'