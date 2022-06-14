import skeletor as sk
import pyglet
import trimesh

"""
General Pipeline:
1. skeletor.pre.fix_mesh() to fix the mesh
2. skeletor.pre.simplify() to simplify the mesh
3. skeletor.pre.contract() to contract the mesh [1]
(there is also skeletor.pre.remesh())
4. to generate a skeleton:
- skeletor.skeletonize.by_wavefront() (fastest)
- skeletor.skeletonize.by_vertex_clusters() (fastest)
- skeletor.skeletonize.by_edge_collapse() (fastest)
- skeletor.skeletonize.by_teasar() (fastest)
- skeletor.skeletonize.by_tangent_ball() (fastest)
5. skeletor.post.clean_up() to clean up some potential issues with the skeleton
6. skeletor.post.radii() to extract radii either by k-nearest neighbours or ray-casting
"""
#mesh = trimesh.load("/Users/olivertoh/Documents/NETSI Research/2CylinderEngine.glb", force="mesh")
mesh = sk.example_mesh()
print(mesh)
fixed = sk.pre.fix_mesh(mesh, remove_disconnected=5, inplace=False)
skel = sk.skeletonize.by_wavefront(fixed, waves=1, step_size=1)
print(skel)
print("Vertices\n", skel.vertices.size, skel.vertices)
print("Edges\n", skel.edges.size, skel.edges)
print("Map\n", skel.mesh_map.size, skel.mesh_map)
skel.save_swc("/Users/olivertoh/Documents/NETSI Research/fruit_fly.swc") #saves the skeleton as an swc file
print(skel.swc.head())
skel.show(mesh=True)

sk.post.radii(skel, method='knn')
sk.post.clean_up(skel, inplace=True)
skel.show(mesh=True)