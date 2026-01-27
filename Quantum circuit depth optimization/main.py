from src.load_graphs import load_graphs
from src.predict import predict_colors
from src.load_model import load_model
from src.model import DeepGCN
import networkx as nx

model_path = "gcn_model.pth"
graphs_dict = load_graphs("qasm_graphs_original.pkl")
output_file = 'output.txt'
hidden_dim = 32
num_layers = 20
model = load_model(model_path, DeepGCN, hidden_dim, num_layers)


with open(output_file, 'a', buffering=1) as file:
    for i, (name, g) in enumerate(graphs_dict.items()):
          colors, coloring = predict_colors(g,model)
          print(colors)
          file.write(name+"\n")

          file.write("predicted depth = "+str(colors)+"\n")
