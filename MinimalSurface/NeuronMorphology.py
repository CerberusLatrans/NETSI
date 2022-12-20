from tkinter import TRUE
import neuprint as npt
from neuprint import Client
from neuprint import fetch_custom, fetch_synapses, NeuronCriteria as NC, SynapseCriteria as SC, skeleton_segments
import skeletor as sk

import matplotlib.pyplot as plt
import math
import pandas as pd
import json
import trimesh
import navis

class NeuronMorphology():
  TOKEN = """eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6InRvaC5vQGh1c2t5Lm5ldS5lZHUiLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hL0FBVFhBSnd6TUtWbHk2ZnhvbTloRDl1UXBXMUlrXzBWWjkxXzdJblF5ajBhPXM5Ni1jP3N6PTUwP3N6PTUwIiwiZXhwIjoxODM0MTE2OTE3fQ.ZJPJsTgsFRwv-obvkDrAhRjRRKfC2WmhYxWmc3gwhb8"""

  DATASET = 'hemibrain:v1.2.1'
  def __init__(self, id):
    self.client = Client('neuprint.janelia.org', dataset=self.DATASET, token=self.TOKEN)
    self.id = id

    q = """\
      MATCH (n:Neuron)
      WHERE n.bodyId = {}
      RETURN n.roiInfo
      """.format(id)

    df = self.client.fetch_custom(q)
    #self.roi = json.loads(self.client.fetch_custom(q)["n.roiInfo"][0])#["SNP(R)\\"[:-1]]
  

    self.synapses = fetch_synapses(id)
    try:
      self.skeleton = self.client.fetch_skeleton(id, heal=True)
      augmented_skeleton = npt.skeleton.attach_synapses_to_skeleton(self.skeleton, self.synapses)
      self.segments = skeleton_segments(augmented_skeleton)
    except:
      print("NO SKEL FOR", self.id)
      self.skeleton = pd.DataFrame()

  def getLengths(self, visualize = False):
    lengths  = self.segments["length"]
    if visualize:
      self.histogram(lengths)

    return lengths

  def getDiameters(self, visualize = False):
    radii = self.segments["avg_radius"]
    diameters = [r*2 for r in radii]
    if visualize:
      self.histogram(diameters)
    
    return diameters

  def getVolumes(self, visualize = False):
    volumes = self.segments["volume"]
    if visualize:
      self.histogram(volumes)
    
    return volumes

  def getAngles(self, visualize = False):
    pass

  def histogram(self, values):
    n, bins, patches = plt.hist(values, bins=math.floor(len(values) / 100))
    plt.plot(bins)
    plt.show()

  def visualize(self, ):
    
    colors = {
    'neurite': 'green',
    'pre': 'blue',
    'post': 'red'
    }
    self.segments['color'] = self.segments['structure'].map(colors)

    fig, axs = plt.subplots(2,2)
    for i in self.segments.index:
      xs = self.segments.loc[i, "x"], self.segments.loc[i, "x_parent"]
      ys = self.segments.loc[i, "y"], self.segments.loc[i, "y_parent"]
      zs = self.segments.loc[i, "z"], self.segments.loc[i, "z_parent"]
      axs[0,0].plot(xs, ys)
      axs[0,1].plot(xs, zs)
      axs[1,0].plot(ys, zs)
    plt.show()



if __name__ == "__main__":
  morphology = NeuronMorphology(327933027)
  morphology.getLengths(visualize=False)
  morphology.getDiameters(visualize=False)
  morphology.getVolumes(visualize=False)
  
  #morphology.visualize()