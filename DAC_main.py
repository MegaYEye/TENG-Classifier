import numpy as np
from DAC_lib import read_DAC_data

tape_files = [
    "data_DAC_structure/tape/hexagonal wrench with double side tape.data",
    "data_DAC_structure/tape/screw head with double side tape.data",
    "data_DAC_structure/tape/tape edge.data",
    "data_DAC_structure/tape/tape side.data",
]
data = [read_DAC_data(f) for f in tape_files]
