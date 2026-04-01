import networkx as nx
import copy
import random

class Degree12Reducer:
    """
    Performs degree-1 (pendant) and degree-2 (vertex folding) reductions
    on a NetworkX graph and returns the reduced graph.
    """

    def __init__(self, graph: nx.Graph):
        # Work on a copy so original graph is untouched
        self.graph = copy.deepcopy(graph)

        # Stats
        self.initial_nodes = self.graph.number_of_nodes()
        self.initial_edges = self.graph.number_of_edges()
        self.pendant_reductions = 0
        self.vertex_reductions = 0

    def pendant_reduction(self):
        """
        Remove a degree-1 vertex.
        """
        deg1_nodes = {v for v, d in self.graph.degree() if d == 1}
        if not deg1_nodes:
            return False
        v = random.choice(tuple(deg1_nodes))
        self.graph.remove_node(v)
        self.pendant_reductions += 1
        return True

    def vertex_folding(self):
        """
        Degree-2 vertex folding using v as representative:
        If v has neighbors u and w, and u and w are NOT adjacent,
        remove u and w, and connect v to all neighbors of u and w.
        """

        deg2_nodes = {v for v, d in self.graph.degree() if d == 2}
        if not deg2_nodes:
            return False
        for v in list(deg2_nodes):
           if len(list(self.graph.neighbors(v)))<2:
              deg2_nodes.remove(v)
              continue
           u, w = list(self.graph.neighbors(v))
           if self.graph.has_edge(u, w):
            deg2_nodes.remove(v)

        if not deg2_nodes:
            return False
        v = random.choice(tuple(deg2_nodes))
        u, w = list(self.graph.neighbors(v))
        # Collect external neighbors of u and w
        external_neighbors = set()
        for x in self.graph.neighbors(u):
          if x not in {u, v, w}:
            external_neighbors.add(x)

        for x in self.graph.neighbors(w):
          if x not in {u, v, w}:
            external_neighbors.add(x)

         # Remove u and w
        self.graph.remove_node(u)
        self.graph.remove_node(w)

         # Remove old edges of v (to avoid stale connections)
        self.graph.remove_edges_from(list(self.graph.edges(v)))

          # Connect v to all external neighbors
        for x in external_neighbors:
            self.graph.add_edge(v, x)

        self.vertex_reductions += 1
        return True

    def reduce_graph(self,percent):
        """
        Apply reductions until no more are possible.
        Returns:
            reduced_graph (nx.Graph)
            stats (dict)
        """
        initial_nodes=self.graph.number_of_nodes()
        changed = True
        while changed and (self.graph.number_of_nodes()>percent* initial_nodes):
            changed = False
            if self.pendant_reduction():
                changed = True
                continue

            if self.vertex_folding():
                changed = True
                continue

        stats = {
            "initial_nodes": self.initial_nodes,
            "final_nodes": self.graph.number_of_nodes(),
            "initial_edges": self.initial_edges,
            "final_edges": self.graph.number_of_edges(),
            "pendant_reductions": self.pendant_reductions,
            "vertex_reductions": self.vertex_reductions,
            "node_reduction_ratio":
                1 - self.graph.number_of_nodes() / self.initial_nodes
                if self.initial_nodes > 0 else 0.0
        }

        number_reduced=self.initial_nodes-self.graph.number_of_nodes()
        if number_reduced>0:
            checker=True
        else:
            checker=False

        return self.graph, stats,checker
