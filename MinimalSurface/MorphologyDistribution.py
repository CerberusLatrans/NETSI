import neuprint as npt
from neuprint import Client
from neuprint import fetch_neurons, fetch_synapses, NeuronCriteria as NC, SynapseCriteria as SC
import skeletor as sk
from skeletor import Skeleton
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import trimesh as tm

from NeuronMorphology import NeuronMorphology

class MorphologyDistribution():
  TOKEN = """eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6InRvaC5vQGh1c2t5Lm5ldS5lZHUiLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hL0FBVFhBSnd6TUtWbHk2ZnhvbTloRDl1UXBXMUlrXzBWWjkxXzdJblF5ajBhPXM5Ni1jP3N6PTUwP3N6PTUwIiwiZXhwIjoxODM0MTE2OTE3fQ.ZJPJsTgsFRwv-obvkDrAhRjRRKfC2WmhYxWmc3gwhb8"""

  DATASET = 'hemibrain:v1.2.1'

  def __init__(self, match, where=""):
    self.client = Client('neuprint.janelia.org', dataset=self.DATASET, token=self.TOKEN)
    if where == "":
      q = """\
      MATCH {}
      RETURN n.bodyId
      """.format(match)
    else:
      q = """\
      MATCH {}
      WHERE {}
      RETURN n.bodyId
      """.format(match, where)

    
    df = self.client.fetch_custom(q)
    self.ids = df["n.bodyId"]
    self.lengths = []
    self.diameters = []
    self.volumes = []
    self.skeletons = {}

    for id in self.ids:
      neuron = NeuronMorphology(id)
      self.diameters.append(neuron.getDiameters())
      self.lengths.append(neuron.getLengths())
      df = neuron.skeleton.rename({"rowId" : "node_id", "link" : "parent_id"}, axis = 1)
      df["node_id"] = df["node_id"] - np.full(len(df["node_id"]), 1)
      df["parent_id"] = df["parent_id"] - np.full(len(df["parent_id"]), 1)
      self.skeletons[id] = df

  def getDiameters(self):
    return self.diameters
  
  def getLengths(self):
    return self.lengths

  def getAngles(self):
    pass

  def histogram(self, values):
    n, bins, patches = plt.hist(values, bins=100)#math.floor(len(values) / 10))
    plt.plot(bins)
    plt.show()

  def saveSWCs(self, folderPath):
    abspath = os.path.abspath(os.getcwd() + folderPath)
    if not os.path.isdir(abspath):
      os.makedirs(abspath)
    for id, skel in self.skeletons.items():
      filePath = abspath + "/" + str(id) + ".swc"
      print(filePath)
      npt.skeleton.skeleton_df_to_swc(skel, filePath)
  
  def visSkeletons(self, ids):
    scene = tm.Scene()
    bounds = []
    for id in ids:
      df = self.skeletons[id]
      skeleton = Skeleton(df).skeleton.copy()
      rgb = tm.visual.color.random_color()
      scene.visual = tm.visual.color.ColorVisuals(mesh=skeleton, vertex_colors=rgb)
      scene.add_geometry(skeleton)
      bounds.append(skeleton.bounds)
      #scene = skeleton.scene(mesh=False)

    fac = 5 / np.fabs(bounds).max()
    scene.apply_transform(np.diag([fac, fac, fac, 1]))
    return scene.show()

if __name__ == "__main__":
  m = "(n:Neuron)"
  w = "n.bodyId IN [707854989, 707863263, 707858790, 327933027, 612371421]"
  #w = "n.instance =~ 'MBON.*'"
  extractor = MorphologyDistribution(match=m, where=w)
  #extractor.histogram(extractor.getLengths())
  #extractor.histogram(extractor.getDiameters())
  #extractor.saveSWCs("/MBON")
  extractor.visSkeletons(extractor.ids)