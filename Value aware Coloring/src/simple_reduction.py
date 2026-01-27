import networkx as nx
import copy

class SimplifiedValueAwareGraph:
    def __init__(self, graph, logger=None):
        """
        Initialize a simplified value-aware graph for reduction tracking.
        This version doesn't track inclusion status and only uses standard folding rules.
        
        Parameters:
            graph (nx.Graph): Original NetworkX graph
            logger: Optional logger object for logging
        """
        self.graph = copy.deepcopy(graph)
        self.logger = logger  # Store the logger
        
        # Initialize node attributes
        for node in self.graph.nodes():
            self.graph.nodes[node]['select_value'] = 1
            self.graph.nodes[node]['nonselect_value'] = 0
                
        # Keep track of merged nodes
        self.merged_history = {}
        for node in self.graph.nodes():
            self.merged_history[node] = {
                "select": [node],  # include this node if selected
                "nonselect": []  # include nothing if not selected
            }

        """for node in self.graph.nodes():
            self.merged_history[node] = [node]  # Initially, each node only contains itself """

    def pendant_fold(self):
        """
        Perform one pendant vertex folding reduction on the graph.
        Updates the neighbor node with new values and removes the leaf vertex.
        Uses equations (8) and (9) from the specification.

        Returns:
            bool: True if a fold was performed; False otherwise.
        """
        for v in list(self.graph.nodes()):
            if self.graph.degree(v) == 1:
                # Get the neighbor
                nbrs = list(self.graph.neighbors(v))
                u = nbrs[0]

                # Get values
                select_u = self.graph.nodes[u]['select_value']
                nonselect_u = self.graph.nodes[u]['nonselect_value']
                select_v = self.graph.nodes[v]['select_value']
                nonselect_v = self.graph.nodes[v]['nonselect_value']

                # Apply standard pendant folding rule (equations 8 and 9)
                new_select = select_u + nonselect_v
                new_nonselect = nonselect_u + select_v

                # update merged history
                mh_u = self.merged_history.get(u, {"select": [u], "nonselect": []})
                mh_v = self.merged_history.get(v, {"select": [v], "nonselect": []})

                self.merged_history[u] = {"select": mh_u["select"] + mh_v["nonselect"],"nonselect": mh_v["select"] + mh_u["nonselect"]}
                # Update merged history
                # self.merged_history[u] = self.merged_history.get(u, [u]) + self.merged_history.get(v, [v])

                # Update the neighbor node with new values
                self.graph.nodes[u]['select_value'] = new_select
                self.graph.nodes[u]['nonselect_value'] = new_nonselect

                # Remove the leaf vertex
                self.graph.remove_node(v)
                if v in self.merged_history:
                    del self.merged_history[v]

                return True
        return False

    def vertex_fold(self):
        """
        Perform one vertex folding reduction on the graph for degree-2 vertices.
        Creates a representative node with values calculated according to
        equations (6) and (7) from the specification.

        Returns:
            bool: True if a fold was performed; False otherwise.
        """
        for v in list(self.graph.nodes()):
            if self.graph.degree(v) == 2:
                nbrs = list(self.graph.neighbors(v))
                if len(nbrs) != 2:
                    continue  # Safety check
                u, w = nbrs

                # Only fold if the two neighbors are not directly connected
                if self.graph.has_edge(u, w):
                    continue

                # Use the minimum node ID as the representative
                # rep_node = min(u, v, w)
                rep_node = v

                # Get values
                select_u = self.graph.nodes[u]['select_value']
                nonselect_u = self.graph.nodes[u]['nonselect_value']
                select_v = self.graph.nodes[v]['select_value']
                nonselect_v = self.graph.nodes[v]['nonselect_value']
                select_w = self.graph.nodes[w]['select_value']
                nonselect_w = self.graph.nodes[w]['nonselect_value']

                # Apply standard vertex folding rule (equations 6 and 7)
                new_select = select_u + select_w + nonselect_v
                new_nonselect = select_v + nonselect_u + nonselect_w

                # Collect all external neighbors of u, v, and w
                external_nbrs = set()
                for node in set(self.graph.neighbors(u)) | set(self.graph.neighbors(w)):
                    if node not in {u, v, w}:
                        external_nbrs.add(node)

                # Log the values if logger is available
                mh_u = self.merged_history.get(u, {"select": [u], "nonselect": []})
                mh_v = self.merged_history.get(v, {"select": [v], "nonselect": []})
                mh_w = self.merged_history.get(w, {"select": [w], "nonselect": []})

                self.merged_history[rep_node] = {"select": mh_v["nonselect"] + mh_u["select"] + mh_w["select"], "nonselect": mh_v["select"] + mh_u["nonselect"] + mh_w["nonselect"]}
                """# Update merged history
                merged_nodes = []
                for node in [u, v, w]:
                    merged_nodes.extend(self.merged_history.get(node, [node]))
                self.merged_history[rep_node] = merged_nodes"""

                # Remove all original nodes
                self.graph.remove_node(u)
                self.graph.remove_node(v)
                if w in self.graph:  # Check if w still exists
                    self.graph.remove_node(w)
                # Remove merged history of u and w
                if u in self.merged_history:
                    del self.merged_history[u]
                if w in self.merged_history:
                    del self.merged_history[w]

                # Add the representative node with new values
                self.graph.add_node(rep_node,
                                    select_value=new_select,
                                    nonselect_value=new_nonselect)

                # Connect representative to external neighbors
                for nbr in external_nbrs:
                    self.graph.add_edge(rep_node, nbr)

                return True
        return False

    def reduce_graph(self):
        """
        Reduce the graph by repeatedly applying folding operations until
        no more reductions are possible.

        Returns:
            tuple: (reduced graph, merged history)
        """
        reduction_performed = True

        while reduction_performed:
            reduction_performed = False

            # Try pendant folding
            if self.pendant_fold():
                reduction_performed = True
                continue

            # Try vertex folding
            if self.vertex_fold():
                reduction_performed = True
                continue

        return self.graph, self.merged_history

    """def reduce_graph(self):
        
        Reduce the graph by repeatedly applying folding operations until no more reductions are possible.
        Does not remove isolated vertices.
        
        Returns:
            tuple: (reduced graph, merged history)
        
        initial_value = self.calculate_max_value()
        reduction_count = 0
        
        reduction_performed = True
        while reduction_performed:
            # Try pendant folding
            if self.pendant_fold():
                reduction_count += 1
                current_value = self.calculate_max_value()
                continue
                
            # Try vertex folding
            if self.vertex_fold():
                reduction_count += 1
                current_value = self.calculate_max_value()
                continue
                
            # If we reach here, no reductions were performed
            reduction_performed = False
        
        final_value = self.calculate_max_value()

        
        return self.graph, self.merged_history """
    
    def get_node_values(self):
        """
        Get the select and nonselect values for all nodes in the graph.
        
        Returns:
            dict: Dictionary mapping node IDs to (select_value, nonselect_value) tuples
        """
        result = {}
        for node in self.graph.nodes():
            result[node] = (
                self.graph.nodes[node]['select_value'],
                self.graph.nodes[node]['nonselect_value']
            )
        return result
    
    def calculate_max_value(self):
        """
        Calculate the maximum possible value achievable from the current graph state.
        
        Returns:
            int: The maximum value possible from the current graph
        """
        max_value = 0
        for node in self.graph.nodes():
            # Add the maximum between select and nonselect value for each node
            max_value += max(
                self.graph.nodes[node]['select_value'],
                self.graph.nodes[node]['nonselect_value']
            )
        return max_value
        
    def reconstruct_solution(self):
        """
        Reconstruct a solution from the reduced graph by choosing the maximum
        value option (select or nonselect) for each node.
        
        Returns:
            dict: Dictionary mapping node IDs to boolean (True if selected)
        """
        solution = {}
        
        # For each node in the reduced graph, choose the option with higher value
        for node in self.graph.nodes():
            select_val = self.graph.nodes[node]['select_value']
            nonselect_val = self.graph.nodes[node]['nonselect_value']
            
            # Choose to select the node if its select_value is higher
            selected = select_val > nonselect_val
            
            # Map the node and all its merged components to this decision
            for orig_node in self.merged_history[node]:
                solution[orig_node] = selected
                
        return solution

    def lift_solution(self, reduced_solution):
        """
        Lift an MIS solution from the reduced graph back to the original graph.

        Parameters:
            reduced_solution (dict):
                Mapping reduced_node -> 0 or 1

        Returns:
            dict:
                Mapping original_node -> 0 or 1
        """
        original_solution = {}

        for red_node, decision in reduced_solution.items():
            if red_node not in self.merged_history:
                continue

            info = self.merged_history[red_node]

            # Choose correct branch based on decision
            chosen_nodes = info["select"] if decision == 1 else info["nonselect"]

            for v in chosen_nodes:
                original_solution[v] = 1

        return original_solution