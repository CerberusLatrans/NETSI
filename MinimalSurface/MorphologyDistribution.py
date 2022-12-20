from optparse import Values
from tkinter import N
import neuprint as npt
from neuprint import Client
import skeletor as sk
from skeletor import Skeleton
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import trimesh as tm
import networkx as nx
from pyknotid.spacecurves import Knot
import pyknotid as pki
#from cloudvolume import CloudVolume
import math
import pickle
import re

from NeuronMorphology import NeuronMorphology

import os
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/Users/olivertoh/Documents/NETSI Research/MinimalSurface/neuron-morphology-95b302b4f4ff.json"

class MorphologyDistribution():
  TOKEN = """eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6InRvaC5vQGh1c2t5Lm5ldS5lZHUiLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hL0FBVFhBSnd6TUtWbHk2ZnhvbTloRDl1UXBXMUlrXzBWWjkxXzdJblF5ajBhPXM5Ni1jP3N6PTUwP3N6PTUwIiwiZXhwIjoxODM0MTE2OTE3fQ.ZJPJsTgsFRwv-obvkDrAhRjRRKfC2WmhYxWmc3gwhb8"""

  DATASET = 'hemibrain:v1.2.1'

  def __init__(self, num_neurons, match, where="", data=None):
    self.num_neurons = num_neurons

    client = Client('neuprint.janelia.org', dataset=self.DATASET, token=self.TOKEN)
    

    fetch = "RETURN DISTINCT "
    for i in range(num_neurons):
      fetch += "n" + str(i) + ".bodyId, "
      fetch += "n" + str(i) + ".somaLocation, "
      fetch += "n" + str(i) + ".somaRadius, "
      if not data:
        fetch += "s" + str(i) + "pre" + ".location, "
        fetch += "s" + str(i) + "post" + ".location, "
    
    fetch = fetch[:-2]

    q = """
      {}
      {}
      {}
      LIMIT 1
      """.format(match, where, fetch)
    
    print(q)
    df = client.fetch_custom(q)
    #print(df.head(n=10))

    if data:

      needed_rows = []
      for i in range(num_neurons):
        postId = data[i][1][0]
        preId = data[i][1][1]

        post_row = postId - 99000000000
        pre_row = preId - 99000000000
        needed_rows.append(post_row)
        needed_rows.append(pre_row)

      currdir = "MinimalSurface/hemibrain_v1.0.1_neo4j_inputs/"
      syninfo  = pd.read_csv(currdir + "Neuprint_Synapses_52a133.csv",\
         usecols=[0, 3],\
          skiprows = lambda x: x not in needed_rows,\
            names = ["id", "pos"])
      #print(syninfo.head(n=5))

      for i in range(num_neurons):
        postId = data[i][1][0]
        preId = data[i][1][1]

        post_xyz = syninfo.loc[syninfo["id"]==postId, "pos"].values[0]
        pre_xyz = syninfo.loc[syninfo["id"]==preId, "pos"].values[0]

        post_xyz = post_xyz.replace(":", " ").replace(",", " ").replace("}", " ")
        pre_xyz = pre_xyz.replace(":", " ").replace(",", " ").replace("}", " ")

        df["s" + str(i) + "post" + ".location"] = [{"coordinates" : [int(s) for s in post_xyz.split() if s.isdigit()]}]
        df["s" + str(i) + "pre" + ".location"] = [{"coordinates" : [int(s) for s in pre_xyz.split() if s.isdigit()]}]

    
    self.df = df

    if data:
      for i in range(self.num_neurons):
        self.df.at[0, "n"+str(i)+".bodyId"] = data[i][0]

    #print(self.df.head(n=5))

    self.ids = []
    for i in range(num_neurons):
      self.ids.extend(df["n" + str(i) + ".bodyId"].values)
    self.ids = [*set(self.ids)]
    
    self.lengths = []
    self.diameters = []
    self.volumes = []

    self.skeletons = {}
    self.meshes = {}

    #vol = CloudVolume('gs://neuroglancer-janelia-flyem-hemibrain/v1.1/segmentation')

    for id in self.ids:
      neuron = NeuronMorphology(id)
      #self.diameters.append(neuron.getDiameters())
      #self.lengths.append(neuron.getLengths())

      if not neuron.skeleton.empty:
        skeldf = neuron.skeleton.rename({"rowId" : "node_id", "link" : "parent_id"}, axis = 1)
        skeldf["node_id"] = skeldf["node_id"] - np.full(len(skeldf["node_id"]), 1)
        skeldf["parent_id"] = skeldf["parent_id"] - np.full(len(skeldf["parent_id"]), 1)
        #print(skeldf)
      self.skeletons[id] = skeldf

      #self.meshes[id] = vol.mesh.get(int(id))[id]
    
  def getDiameters(self):
    return self.diameters
  
  def getLengths(self):
    return self.lengths

  def getAngles(self):
    pass

  def histogram(self, values):
    n, bins, patches = plt.hist(values, bins=100)
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
  
  def saveMeshes(self, folderPath):
    abspath = os.path.abspath(os.getcwd() + folderPath)
    if not os.path.isdir(abspath):
        os.makedirs(abspath)

    for id in self.meshes.keys():
      mesh = self.meshes[id]
      mesh = tm.Trimesh(vertices=mesh.vertices)
      
      filepath = abspath + "/" + str(id) + "MESH.obj"
      print("SAVED" + str(id))
      mesh.export(filepath, "obj")

  def to_graph(self, df):
    G = nx.Graph()
    for idx,row in df.iterrows():
      G.add_edge(row["node_id"], row["parent_id"])

    return G
    
  def getKnot(self, setNum):
    full_path = []
    start_nodes = []
    #create map from id name to post-pre synaptic pair names for each neuron
    neurons = {"n" + str(i) + ".bodyId" : \
      ["s" + str(i) + "post" ".location", "s" + str(i) + "pre" ".location"]\
         for i in range(self.num_neurons)}
    
    #for each neuron, find the path from the post to pre synapse and add it to the full path
    for id_name, syn_pair in neurons.items():
      skel_df = self.skeletons[self.df.loc[setNum, id_name]]
      if skel_df.empty:
        continue
      graph = self.to_graph(skel_df)

      post_pre = self.df.loc[setNum, syn_pair]
      post_coord = post_pre[0]["coordinates"]
      pre_coord = post_pre[1]["coordinates"]

      position_dict = {}
      best_dist_post = math.inf
      best_dist_pre = math.inf
      best_id_post = 0
      best_id_pre = 0

      #identify the node closest to the pre and the node closest to the post
      for idx,row in skel_df.iterrows():
        position = [row["x"], row["y"], row["z"]]
        position_dict[row["node_id"]] = position
        dist_to_post = math.dist(position, post_coord)
        dist_to_pre = math.dist(position, pre_coord)
        if  dist_to_post < best_dist_post:
          best_id_post = row["node_id"]
          best_dist_post = dist_to_post
        if dist_to_pre < best_dist_pre:
          best_id_pre = row["node_id"]
          best_dist_pre = dist_to_pre
      #nx.set_node_attributes(graph, position_dict, name="position")

      #for neuron skeleton, find path from post to pre
      skel_id_path = nx.shortest_path(graph, source=best_id_post, target=best_id_pre)
      skel_posn_path = [position_dict[i] for i in skel_id_path]
      start_nodes.append(skel_posn_path[0])

      #extend existing path with this neuron's post-pre path
      full_path.extend(skel_posn_path)
    
    #create knot
    k = Knot(np.array(full_path), verbose=True)

    try:
      print(k.identify())
    except:
      pass
    #k.plot(mode="vispy")
    #pki.visualise.plot_line(full_path)
    #pki.visualise.plot_projection(full_path)
    alex_poly = None
    try:
        alex_poly = k.alexander_polynomial()
    except:
        pass
    return full_path, alex_poly, start_nodes
  
  def visSkeletons(self, setNum, full_path, showMeshes=False, showSkeletons = True, start_nodes=None):
    idCols = ["n" + str(i) + ".bodyId" for i in range(self.num_neurons)]
    ids = self.df.loc[setNum, idCols]
    print(ids)
    post_syn_cols = ["s" + str(i) + "post" ".location" for i in range(self.num_neurons)]
    pre_syn_cols = ["s" + str(i) + "pre" ".location" for i in range(self.num_neurons)]
    synapses = self.df.loc[setNum, post_syn_cols + pre_syn_cols]
    scene = tm.Scene()
    bounds = []
    colors = []

    for id in ids:
        
        df = self.skeletons[id]
        skeleton = sk.post.postprocessing.drop_parallel_twigs(Skeleton(df), theta=1, inplace=False).skeleton
        color = tm.visual.color.random_color()
        colors.append(color)
        skeleton.colors = np.full((len(skeleton.entities), 4), color)  
        bounds.append(skeleton.bounds)
        if showSkeletons:
            scene.add_geometry(skeleton)

        if showMeshes:
            mesh = self.meshes[id]
            cloud = tm.points.PointCloud(mesh.vertices)
            #cloud = tm.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
            cloud.colors = np.full((len(cloud.vertices), 4), color)
            scene.add_geometry(cloud)

    

    """
    for s in synapses:
      point = tm.primitives.Sphere(radius=0.05, center=s["coordinates"])
      scene.add_geometry(point)
    
    for i in range(self.num_neurons):
      rad = self.df["n" + str(i) + ".somaRadius"][setNum]
      if rad:
        soma = tm.primitives.Sphere(radius=rad*0.0005, center=self.df["n" + str(i) + ".somaLocation"][setNum]["coordinates"])
        scene.add_geometry(soma)
    """
    color_idx = 0
    for pt in full_path:
      if start_nodes is not None and pt in start_nodes:
        color = colors[color_idx]
        color_idx += 1
       
      sphere = tm.primitives.Sphere(radius=0.01, center=pt)
      tm.visual.color.ColorVisuals(sphere, color, color)
      #sphere.colors = np.full((len(sphere.entities), 4), color) 
      scene.add_geometry(sphere)
      

    
    fac = 5 / np.fabs(bounds).max()
    scene.apply_transform(np.diag([fac, fac, fac, 1]))

    return scene.show()
  
  @staticmethod
  def getCycle(threshold=4, max=100, coverage = 1.0, source_node=None, pickle_path=None, num_cycles=100):
    currdir = "MinimalSurface/hemibrain_v1.0.1_neo4j_inputs/"
    df = pd.read_csv(currdir + "Neuprint_Neuron_Connections_52a133.csv")
    num_edges = int(len(df)*coverage)
    print(num_edges)
    from_neurons = df[":START_ID(Body-ID)"].to_numpy()
    to_neurons = df[":END_ID(Body-ID)"].to_numpy()


    G = nx.DiGraph()

    for i in range(num_edges):
      G.add_edge(from_neurons[i], to_neurons[i])
    
    #cycle = nx.find_cycle(G, source=source_node, orientation = "original")
    cycles = nx.simple_cycles(G)

    all_paths = []
    for cycle in cycles:
    
      #path = [x[0] for x in cycle]
      path = cycle
      #print(len(path))
      #print(path)
      if len(path) < threshold or len(path) > max:
        print("DISCARD")
        continue
      

      #FORMAT:  neuronID : (postSynID, preSynID)
      ss2syn = pd.read_csv(currdir + "Neuprint_SynapseSet_to_Synapses_52a133.csv")
      path_to_synapses = []
      for i in range(len(path)):
        post_syn = ss2syn.loc[ss2syn[':START_ID']==str(path[i]) + "_" + str(path[i-1]) + "_post", ':END_ID(Syn-ID)'].values
        pre_syn = ss2syn.loc[ss2syn[':START_ID']==str(path[i]) + "_" + str(path[i+1 if i < len(path)-1 else 0]) + "_pre", ':END_ID(Syn-ID)'].values
        # if there are more than one, just choose first synapse
        if len(post_syn) > 0:
          post_syn = post_syn[0]
        else:
          print("NO POST SYNAPSE", i)
        if len(pre_syn) > 0:
          pre_syn = pre_syn[0]
        else:
          print("NO PRE SYNAPSE", i)
        path_to_synapses.append([path[i], [post_syn, pre_syn]])

      #print("DICT: " , path_to_synapses)

      if pickle_path:
        with open(pickle_path, "wb") as f:
          pickle.dump(path_to_synapses, f)
      
      all_paths.append(path_to_synapses)
      print("PATH SO FAR: ", all_paths)
      if len(all_paths) >= num_cycles:
        break
    
    return all_paths

  @staticmethod
  def getLoop(threshold, source_edge=(612371421, 986828383), pickle_path=None):
    currdir = "MinimalSurface/hemibrain_v1.0.1_neo4j_inputs/"
    df = pd.read_csv(currdir + "Neuprint_Neuron_Connections_52a133.csv")
    ss2syn = pd.read_csv(currdir + "Neuprint_SynapseSet_to_Synapses_52a133.csv")
    df = df[[':START_ID(Body-ID)', ":END_ID(Body-ID)"]]

    edge_idx = df.loc[(df[':START_ID(Body-ID)']==source_edge[0]) & (df[':END_ID(Body-ID)']==source_edge[1])].index[0]
    count = 0
    curr = df.iloc[edge_idx, 0]
    next = df.iloc[edge_idx, 1]
    source = curr

    path = []
    synapses = []
    
    while next != source:
      
      #df.drop([edge_idx])
      if curr in path:
        print("DUPLICATE")
      path.append(curr)
      print(path)
      curr_pre = ss2syn.loc[ss2syn[':START_ID']==str(curr)+"_"+str(next)+"_pre", ':END_ID(Syn-ID)'].values[0]
      next_post = ss2syn.loc[ss2syn[':START_ID']==str(next)+"_"+str(curr)+"_post", ':END_ID(Syn-ID)'].values[0]
      synapses.append([curr_pre, next_post])
      count += 1
        
      candidates = df.loc[df[':START_ID(Body-ID)']==next]

      if count >= threshold:
        end_candidate = df.loc[(df[':START_ID(Body-ID)']== next) & (df[':END_ID(Body-ID)']==source)] #<------ BUG!!!!
        if not end_candidate.empty:
          curr = end_candidate.iloc[0, 0]
          next = end_candidate.iloc[0, 1]
          print("SOURCE IDENTIFIED: ", next)
          continue
      
      for idx in candidates.index:
        curr = df.iloc[idx, 0]
        next = df.iloc[idx, 1]
        if curr not in path and next not in path and next != curr and\
          str(curr)+"_"+str(next)+"_pre" in ss2syn[":START_ID"].values and\
          str(next)+"_"+str(curr)+"_post" in ss2syn[":START_ID"].values and\
          not df.loc[df[':START_ID(Body-ID)']==next].empty:


          df.drop(candidates.index, inplace=True)
          break

    print("PATH LENGTH: ", len(path))

    path_to_synapses = []
    for i in range(len(path)):
      path_to_synapses.append([path[i], [synapses[i-1][1], synapses[i][0]]])
    
    print(path_to_synapses)
    return path_to_synapses
    
    #FORMAT:  neuronID : (postSynID, preSynID)
    path_to_synapses = []
    for i in range(len(path)):
      post_syn = ss2syn.loc[ss2syn[':START_ID']==str(path[i]) + "_" + str(path[i-1]) + "_post", ':END_ID(Syn-ID)'].values
      pre_syn = ss2syn.loc[ss2syn[':START_ID']==str(path[i]) + "_" + str(path[i+1 if i < len(path)-1 else 0]) + "_pre", ':END_ID(Syn-ID)'].values
      #post_syn = ss2syn.loc[ss2syn[':START_ID']==str(path[i-1]) + "_" + str(path[i]) + "_post", ':END_ID(Syn-ID)'].values
      #pre_syn = ss2syn.loc[ss2syn[':START_ID']== str(path[i+1 if i < len(path)-1 else 0]) + "_" + str(path[i]) + "_pre", ':END_ID(Syn-ID)'].values
      # if there are more than one, just choose first synapse
      if len(post_syn) > 0:
        post_syn = post_syn[0]
      else:
        print("NO POST SYNAPSE", i)
      if len(pre_syn) > 0:
        pre_syn = pre_syn[0]
      else:
        print("NO PRE SYNAPSE", i)
      path_to_synapses.append([path[i], [post_syn, pre_syn]])

    print("DICT: " , path_to_synapses)
    

    if pickle_path:
      with open(pickle_path, "wb") as f:
        pickle.dump(path_to_synapses, f)
    
    return path_to_synapses

  #only using the synapset to synapses csv
  @staticmethod
  def getLoop2(threshold, source_edge=(612371421, 986828383), pickle_path=None):
    currdir = "MinimalSurface/hemibrain_v1.0.1_neo4j_inputs/"
    ss2syn = pd.read_csv(currdir + "Neuprint_SynapseSet_to_Synapses_52a133.csv")

    curr = source_edge[0]
    next = source_edge[1]

    source = curr

    path = []

    while next != source:

      if curr in path:
        print("DUPLICATE")
      
      #append the current neuron and the pre-synapse linking it to the next neuron
      pre_id = str(source_edge[0])+"_"+str(source_edge[1])+"_pre"
      path.append(curr)
      #[ss2syn.loc[ss2syn[':START_ID']==pre_id, ":END_ID(Syn-ID)"]]
      print(path)
      
      """
      start_key = ss2syn[':START_ID'].str.startswith(str(next))
      end_key = ss2syn[':START_ID'].str.endswith("pre")
      if True in start_key:
        print("CHECK 1")
      if True in end_key:
        print("CHECK 2") 
      """
      
      #candidates = ss2syn.loc[[(ss2syn[':START_ID'].split("_")[0]==str(next)) & (ss2syn[':START_ID'].split("_")[2]=="pre")]]
      candidates = ss2syn.loc[(ss2syn[':START_ID'].str.startswith(str(next))) & (ss2syn[':START_ID'].str.endswith("pre"))]
      print(candidates)

      if len(path) >= threshold-1:
        end_candidate = candidates.loc[ss2syn[':START_ID']==str(next)+"_"+str(source)+"_pre"]
        if not end_candidate.empty:
          curr = int(end_candidate.iloc[0].split("_")[0])
          if curr != next:
            print("ERROR")
          next = int(end_candidate.iloc[0].split("_")[1])
          print("SOURCE IDENTIFIED: ", next)
      
      else:
        for idx in candidates.index:
          curr = int(ss2syn.iloc[idx, 0].split("_")[0])
          next = int(ss2syn.iloc[idx, 0].split("_")[1])

          next_candidates = ss2syn.loc[(ss2syn[':START_ID'].str.startswith(str(next))) & (ss2syn[':START_ID'].str.endswith("pre"))]
          if not next_candidates.empty:
            ss2syn.drop(candidates.index, inplace=True)
            break

    print("PATH LENGTH: ", len(path))

    if pickle_path:
      with open(pickle_path, "wb") as f:
        pickle.dump(path, f)
    
    return path

  @staticmethod
  def getMatch(n):
    match = ""
    for i in range(n):
      match += "\nMATCH (n{}:Neuron)-[:Contains]->(:SynapseSet)-[:Contains]->(s{}pre:Synapse)-[:`SynapsesTo`]->(s{}post:Synapse)<-[:Contains]-(:SynapseSet)<-[:Contains]-(n{}:Neuron)"\
        .format(i,i,0 if i==n-1 else i+1,0 if i==n-1 else i+1)

    return match

  @staticmethod
  def getSimpleMatch(n):
    match = ""
    for i in range(n):
      match += "\nMATCH (n{}:Neuron)"\
        .format(i)

    return match

  @staticmethod
  def getWhere(path):
    where = "WHERE "
    for i, id in enumerate(path):
      if i > 0:
        where += " AND "
      where += "n{}.bodyId = {}".format(i, id)

    return where

if __name__ == "__main__":
  """

  extractor = MorphologyDistribution(num_neurons=NUM_NEURONS,\
     match=MorphologyDistribution.getMatch(NUM_NEURONS),\
       where=MorphologyDistribution.getWhere(MorphologyDistribution.getLoop(NUM_NEURONS, (612371421, 986828383)).keys()))
  """
  #path_58 = [203935234, 203253253, 205982050, 204962646, 203257652, 204617274, 203939602, 203253072, 203931040, 203930845, 204271867, 203598501, 204272223, 204958320, 204954059, 204608953, 205299679, 203939724, 203598505, 204621561, 204276619, 203248801, 204962684, 204958353, 204954016, 204617827, 204617237, 203257413, 204276278, 204280644, 203598466, 204272174, 203939660, 204276668, 203598485, 204621549, 203598499, 202916528, 203598941, 204276689, 203598504, 204276195, 204617233, 204613356, 204958610, 204613133, 203594175, 203594169, 204271951, 203930785, 203935157, 203935596, 203594163, 203930803, 203598628, 203248725, 203598647, 203594554]
  #path_to_synapses = MorphologyDistribution.getLoop(NUM_NEURONS, source_edge =(550408954, 1173866149))

  cycles = MorphologyDistribution.getCycle(threshold=29, max = 29, coverage = 0.0005, source_node=None, num_cycles=1)

  #cycles = [[203248801, [99001204284, 99000752122]], [203257413, [99000752475, 99000078770]], [203594554, [99000078763, 99000002348]], [203257532, [99000002302, 99000007912]], [203253072, [99000007909, 99000761500]], [203594175, [99000761506, 99010216978]], [203594169, [99010216976, 99002439436]], [203253253, [99002439429, 99000003455]], [203594164, [99000003515, 99000788017]], [203598466, [99000788013, 99000062521]], [203248973, [99000062522, 99001196338]]]
  #cycles = [cycles]
 
  """
  import ast
  with open('MinimalSurface/cycles.txt', 'r') as file:
    cycles = file.read().replace('\n', '')

  cycles = ast.literal_eval(cycles)
  """

  print(cycles)
  knots = {29 : []}
  print(knots)
  for path_to_synapses in cycles:
    degree = len(path_to_synapses)
    try:
        extractor = MorphologyDistribution(num_neurons=len(path_to_synapses),\
        match=MorphologyDistribution.getSimpleMatch(len(path_to_synapses)),\
        where=MorphologyDistribution.getWhere([i[0] for i in path_to_synapses]),\
        data=path_to_synapses)

        path, poly, start_nodes = extractor.getKnot(0)
        print(path)
        print(start_nodes)
        extractor.visSkeletons(0, path, showSkeletons=True, start_nodes=start_nodes)
        extractor.visSkeletons(0, path, showSkeletons=False, start_nodes=start_nodes)
        if poly is None:
            print("UNIDENTIFIED")
            poly = 0
        
        knots[degree].append(poly)
    except:
        print("COULD NOT USE", path_to_synapses)
        knots[degree].append("?")

    print(knots)
  print(knots)
  
  
  
"""extractor = MorphologyDistribution(num_neurons=4,\
     match=MorphologyDistribution.getMatch(4),\
       where="WHERE n0.bodyId = 203598466 AND n1.bodyId = 203248973 AND n2.bodyId = 203248725 AND n3.bodyId = 203257652")

path, poly, starts = extractor.getKnot(0)
extractor.visSkeletons(0, path, showSkeletons=True)
extractor.visSkeletons(0, path, showSkeletons=False, start_nodes=starts)"""
   

  #extractor = MorphologyDistribution(num_neurons=NUM_NEURONS, match=MorphologyDistribution.getMatch(NUM_NEURONS), where=w2)
