from src.load_graphs import load_graphs
from src.predict import predict_colors
from src.load_model import load_model
from src.model import DeepGCN
import networkx as nx

model_path = "/Users/ACER/Desktop/basicMIS_multiplesolns/gcn_model.pth"
graphs_dict = load_graphs("/Users/ACER/PycharmProjects/Mixed-Graph-Coloring-Algorithms/quantum_circuit/qasm_graphs_original.pkl")
output_file = 'output.txt'
hidden_dim = 32
num_layers = 20
model = load_model(model_path, DeepGCN, hidden_dim, num_layers)

"""G = nx.DiGraph()

# Add nodes
G.add_node("G1")
G.add_node("G2")
G.add_node("G3")
G.add_node("G4")
G.add_node("G5")

# Add directed edge A -> B
G.add_edge("G1", "G3", directed=False)
G.add_edge("G1", "G4", directed=True)
G.add_edge("G2", "G3", directed=False)
G.add_edge("G2", "G4", directed=True)
G.add_edge("G2", "G5", directed=True)
G.add_edge("G3", "G4", directed=True)

# Print edges with attributes
print("Edges in the graph:")
for u, v, d in G.edges(data=True):
    print(f"{u} -> {v}, directed = {d.get('directed')}")

colors,coloring=predict_colors(G,model)
print(colors) """

with open(output_file, 'a', buffering=1) as file:
    for i, (name, g) in enumerate(graphs_dict.items()):
          colors, coloring = predict_colors(g,model)
          print(colors)
          file.write(name+"\n")
          file.write("predicted depth = "+str(colors)+"\n")