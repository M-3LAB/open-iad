from configuration.device import assign_service
from rich import print
from tools.utils import *
import yaml

import warnings
warnings.filterwarnings("ignore")

class CentralizedAD3D():
    def __init__(self, args):
        self.args = args
    
    # TODO