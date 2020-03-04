# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import os
import math
from random import randint, uniform
from copy import deepcopy
from datetime import datetime
from typing import List, Dict
from .model import HyperparemetersSmtagModel
from ..common.utils import cd


class HyperScan():

    def __init__(self, hp: HyperparemetersSmtagModel, scans_dir: str, scan_name: str, scan_params: List):
        """
    
        """

        self.hp = hp
        self.scan_params = scan_params
        for k in self.scan_params:
            assert k in dir(self.hp) 
        timestamp = datetime.now().isoformat("-",timespec='minutes').replace(":", "-") # dir to save scan results
        self.path = os.path.join(scans_dir, timestamp + "_" + scan_name + "_" +  "scan.csv")

    def append_to_csv(self, row: Dict, hp: HyperparemetersSmtagModel, iteration):  
        NL = "\n"
        SEP= "\t"
        if os.path.isfile(self.path):
            mode = 'a' # append row at the end of the file
        else:
            mode = 'w' # create and write to file for the first time
        with open(self.path, mode) as f:
            if mode == 'w': 
                # write a header line when writing for the first time to the file
                header_params = SEP.join([k for k in self.scan_params])
                header_results = SEP.join([k for k in row])
                header = SEP.join([header_params, header_results])
                f.write(header+NL)
                print(header)
            line_params = SEP.join([str(hp.__getattribute__(k)) for k in self.scan_params])
            line_results = SEP.join([str(row[k]) for k in row])
            line = SEP.join([line_params, line_results])
            f.write(line+NL)
            print(line)
        # os.chmod(self.path, mode=0o777)


    def randhp(self) -> HyperparemetersSmtagModel:
        """
        Model and training hyperparameter are sampled randomly given the specified ranges.

        Args: 
            selected_hyperparamters (str) from 'learning_rate' | 'batch_size' | 'N_layers'| 'hidden_channels'

        Returns:
            randomly sample hyperparameters (HyperparemetersSmtagModel)
        """

        hp = deepcopy(self.hp)
        #randomly sampling selected hyperparameters
        if 'learning_rate' in self.scan_params: 
            hp.learning_rate = 10 ** uniform(-4, -1)
        if 'minibatch_size' in self.scan_params:
            hp.best_model_name =  int(2 ** uniform(4, 8))
        if 'N_layers' in self.scan_params:
            hp.N_layers = randint(1,10)
        if  'hidden_channels' in self.scan_params:
            hp.hidden_channels = int(2 ** randint(3, 6))

        return hp