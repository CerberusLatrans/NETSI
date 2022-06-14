from neuprint import Client
from neuprint import fetch_neurons, fetch_synapses, NeuronCriteria as NC, SynapseCriteria as SC

myToken = """eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6InRvaC5vQGh1c2t5Lm5ldS5lZHUiLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hL0FBVFhBSnd6TUtWbHk2ZnhvbTloRDl1UXBXMUlrXzBWWjkxXzdJblF5ajBhPXM5Ni1jP3N6PTUwP3N6PTUwIiwiZXhwIjoxODM0MTE2OTE3fQ.ZJPJsTgsFRwv-obvkDrAhRjRRKfC2WmhYxWmc3gwhb8"""
c = Client('neuprint.janelia.org', dataset='hemibrain:v1.2.1', token=myToken)
c.fetch_version()

q = """\
    MATCH (n :Neuron {`AB(R)`: true})
    WHERE n.pre > 10
    RETURN n.bodyId AS bodyId, n.type as type, n.instance AS instance, n.pre AS numpre, n.post AS numpost
    ORDER BY n.pre + n.post DESC
"""

results = c.fetch_custom(q)
print(f"Found {len(results)} results")
print(results.head())

criteria = 387023620
neuron_df, roi_counts_df = fetch_neurons(criteria)
print("NEURON DF\n", neuron_df.shape, neuron_df)
print("ROI COUNT DF\n", roi_counts_df.shape, roi_counts_df)