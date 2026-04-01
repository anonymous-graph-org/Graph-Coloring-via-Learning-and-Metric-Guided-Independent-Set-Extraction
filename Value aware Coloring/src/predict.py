import torch
import networkx as nx
import time
from queue import Queue
import random
import copy

def predict_value_aware_mis(model, graph, time_budget=30, num_maps=32,
                                  max_solutions=16, device='cpu'):
    """
    Generate multiple value-aware MIS solutions using multiple probability maps.

    Returns:
        list of dicts: Each dict maps node -> {0,1}
    """
    model.eval()

    class GraphState:
        def __init__(self, graph, labels=None):
            self.graph = graph
            self.labels = labels if labels is not None else {}

        def is_completely_labeled(self):
            return len(self.labels) == len(self.graph)

        def get_unlabeled_subgraph(self):
            unlabeled = [n for n in self.graph.nodes() if n not in self.labels]
            return self.graph.subgraph(unlabeled)

        def get_unlabeled_nodes(self):
            """Returns list of unlabeled node indices"""
            return [n for n in range(self.num_nodes) if n not in self.labels]

        def signature(self):
            return tuple(sorted(n for n, v in self.labels.items() if v == 1))

    def get_probability_maps(g):
        adj_matrix = torch.tensor(
            nx.adjacency_matrix(g).todense(),
            dtype=torch.float32,
            device=device
        )
        degree_matrix = torch.diag(torch.sum(adj_matrix, dim=1))

        nodes = sorted(g.nodes())
        select_values = torch.tensor(
            [g.nodes[n]['select_value'] for n in nodes],
            dtype=torch.float32,
            device=device
        )
        nonselect_values = torch.tensor(
            [g.nodes[n]['nonselect_value'] for n in nodes],
            dtype=torch.float32,
            device=device
        )

        features = torch.ones(len(nodes), model.hidden_dim, device=device)
        features[:, 0] = select_values
        features[:, 1] = nonselect_values

        with torch.no_grad():
            prob_maps = model(adj_matrix, degree_matrix, features)

        return prob_maps, nodes

    # Initialize
    start_time = time.time()
    Q = Queue()
    Q.put(GraphState(graph))

    # Track processed states to avoid duplicates
    #seen_signatures = set()

    solutions = []
    seen_signatures = set()

    # Main loop
    while time.time() - start_time < time_budget and not Q.empty():
        current_state = random.choice(list(Q.queue))
        Q.queue.remove(current_state)

        subgraph = current_state.get_unlabeled_subgraph()
        if len(subgraph) == 0:
            continue

        prob_maps, nodes = get_probability_maps(subgraph)

        for m in range(min(num_maps, prob_maps.size(1))):
            new_state = GraphState(graph, copy.deepcopy(current_state.labels))
            probs = prob_maps[:, m]

            vertices = sorted(
                list(subgraph.nodes()),
                key=lambda x: probs[nodes.index(x)].item(),
                reverse=True
            )

            for v in vertices:
                if v in new_state.labels:
                    break
                new_state.labels[v] = 1
                for nbr in graph.neighbors(v):
                    if nbr not in new_state.labels:
                        new_state.labels[nbr] = 0

            if new_state.is_completely_labeled():
                total_value = sum(
                    graph.nodes[n]['select_value'] if l == 1
                    else graph.nodes[n]['nonselect_value']
                    for n, l in new_state.labels.items()
                )

                sig = new_state.signature()
                if sig not in seen_signatures:
                    seen_signatures.add(sig)
                    solutions.append((copy.deepcopy(new_state.labels), total_value))
            else:
                Q.put(new_state)

    # Sort by value (descending)
    solutions.sort(key=lambda x: x[1], reverse=True)

    # Return only label dicts
    return [s[0] for s in solutions[:max_solutions]]
