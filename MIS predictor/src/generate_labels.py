from load_graphs import load_graphs_from_file
import pickle
from gurobi_mis import gurobi_multiple_mis

output_file = 'graph_labels.pickle'
graph_file = 'networkx_graphs.pkl'
names_file = 'graph_names.pkl'
colors_file = 'graph_colors.pkl'
graphs,names,colors = load_graphs_from_file(graph_file,names_file,colors_file)

graph_mis_dict = {}
with open('output.txt', 'w', buffering=1) as file:
    for i, graph in enumerate(graphs):
        file.write("Graph "+str(i+1)+"\n")
        gurobi_mis = gurobi_multiple_mis(graph)
        graph_mis_dict[i] = gurobi_mis
        for mis in gurobi_mis:
             file.write(str(mis))
             file.write("\n")

with open(output_file, 'wb') as f:
        pickle.dump(graph_mis_dict, f)
