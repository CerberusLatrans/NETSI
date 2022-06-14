import skeletor as sk
import pyglet
import trimesh
import pandas as pd
import numpy as np
import neurom as nm
from neurom import viewer

#path = "/Users/olivertoh/Documents/NETSI Research/allen cell types/Source-Version/H15-06-016-01-03-02_549378277_m.swc"
#path = "/Users/olivertoh/Documents/NETSI Research/allen cell types/CNG version/H15-06-016-01-03-02_549378277_m.CNG.swc"
#path = "/Users/olivertoh/Documents/NETSI Research/allen cell types/Source-Versionbackup/H15-06-016-01-03-02_549378277_m.swc"
path = "/Users/olivertoh/Documents/NETSI Research/allman/CNG version/02a_pyramidal2aFI.CNG.swc"
nrn = nm.load_morphology(path)
fig, ax = viewer.draw(nrn)
fig.show()
#fig, ax = viewer.draw(nrn, mode='3d')
#fig.show()