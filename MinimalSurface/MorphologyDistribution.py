from cv2 import DFT_INVERSE
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
      {}
      RETURN n.bodyId, m.bodyId, a.location, b.location, c.location, d.location, n.somaLocation, n.somaRadius, m.somaLocation, m.somaRadius
      LIMIT 2
      """.format(match)
    else:
      q = """\
      {}
      {}
      RETURN n.bodyId, m.bodyId, a.location, b.location, c.location, d.location, n.somaLocation, n.somaRadius, m.somaLocation, m.somaRadius
      LIMIT 2
      """.format(match, where)

    
    df = self.client.fetch_custom(q)
    self.df = df
    self.ids = df["n.bodyId"].append(df["m.bodyId"])#.append(df["o.bodyId"])
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
  
  def visSkeletons(self, setNum):
    ids = self.df.iloc[setNum,:2]
    synapses = self.df.iloc[setNum,2:6]
    scene = tm.Scene()
    bounds = []
    for id in ids:
      df = self.skeletons[id]
      skeleton = Skeleton(df).skeleton.copy()
      color = tm.visual.color.random_color()
      skeleton.colors = np.full((len(skeleton.entities), 4), color)
      scene.add_geometry(skeleton)
      bounds.append(skeleton.bounds)
    for s in synapses:
      point = tm.primitives.Sphere(radius=0.1, center=s["coordinates"])
      scene.add_geometry(point)

    soma1 = tm.primitives.Sphere(radius=self.df["n.somaRadius"][setNum]*0.0005, center=self.df["n.somaLocation"][setNum]["coordinates"])
    scene.add_geometry(soma1)
    soma2 = tm.primitives.Sphere(radius=self.df["m.somaRadius"][setNum]*0.0005, center=self.df["m.somaLocation"][setNum]["coordinates"])
    scene.add_geometry(soma2)

    fac = 5 / np.fabs(bounds).max()
    scene.apply_transform(np.diag([fac, fac, fac, 1]))
    return scene.show()

if __name__ == "__main__":
  m = "(n:Neuron)"
  w = "n.bodyId IN [707854989, 707863263, 707858790, 327933027, 612371421]"
  #w = "n.`CA(R)`"
  #w = "n.instance =~ 'MBON.*'"
  m2 = "MATCH (n:Neuron)-[:ConnectsTo]->(m:Neuron)"
  m2 += "\nMATCH (m:Neuron)-[:ConnectsTo]->(o:Neuron)"
  m2 += "\nMATCH (o:Neuron)-[:ConnectsTo]->(n:Neuron)"
  #m2 += "\nMATCH (p:Neuron)-[:ConnectsTo]->(n:Neuron)"
  #w2 = "WHERE w.weight < w2.weight OR w.weight > w2.weight"

  m3 = "MATCH (n:Neuron)-[:Contains]->(:SynapseSet)-[:Contains]->(a:Synapse)-[:`SynapsesTo`]->(b:Synapse)<-[:Contains]-(:SynapseSet)<-[:Contains]-(m:Neuron)"
  m3 += "\nMATCH (m:Neuron)-[:Contains]->(:SynapseSet)-[:Contains]->(c:Synapse)-[:`SynapsesTo`]->(d:Synapse)<-[:Contains]-(:SynapseSet)<-[:Contains]-(n:Neuron)"
  w3 = "WHERE n.bodyId = 5813020698 AND a.location.x < b.location.x"
  w3 = "WHERE n.bodyId = 707854989"

  dual = "MATCH (n:Neuron)-[:ConnectsTo]->(m:Neuron)"
  dual += "\nMATCH (m:Neuron)-[:ConnectsTo]->(n:Neuron)"
  extractor = MorphologyDistribution(match=m3, where=w3)#, where=w3)
  extractor.visSkeletons(0)
  #extractor.visSkeletons(1)
  #for i in range(1, 10):
    #extractor.visSkeletons(i)
  