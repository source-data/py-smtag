# -*- coding: utf-8 -*-
#T. Lemberger, 2018

NBITS = 32
DATA_DIR = "data"
MODEL_DIR = "models"
PROD_DIR = "rack"
RUNS_LOG_DIR = "runs"
MARKING_CHAR = u'\uE000'
MARKING_CHAR_ORD = ord(MARKING_CHAR)
SD_PANEL_OPEN =  "<sd-panel>"
SD_PANEL_CLOSE = "</sd-panel>"
MIN_PADDING = 20 # this should be a common param with dataimport
MIN_SIZE = 140 # input needs to be of minimal size to survive successive convergent convolutions; ideally, should be calculated analytically
