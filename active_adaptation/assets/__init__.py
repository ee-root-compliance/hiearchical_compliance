import os
import copy
from isaaclab.assets import Articulation
from .humanoid import *


ASSET_PATH = os.path.dirname(__file__)

ROBOTS = {
    "g1": G1_CFG,
    "g1_col_full": G1_COL_FULL,
}