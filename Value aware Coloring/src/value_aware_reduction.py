import networkx as nx
import copy

class ValueAwareGraph:
    def __init__(self, graph, mis_solution=None, logger=None):
        """
        Initialize a value-aware graph for reduction tracking.
        
        Parameters:
            graph (nx.Graph): Original NetworkX graph
            mis_solution (dict, optional): A dictionary mapping nodes to 0 or 1, where 1 means included in MIS
            logger: Optional logger object for logging
        """
        self.graph = copy.deepcopy(graph)
        self.logger = logger  # Store the logger
        
        # Initialize node attributes
        for node in self.graph.nodes():
            self.graph.nodes[node]['select_value'] = 1
            self.graph.nodes[node]['nonselect_value'] = 0
            
            # If MIS solution is provided, set 'included' based on solution
            if mis_solution:
                self.graph.nodes[node]['included'] = (mis_solution.get(node, 0) == 1)
            else:
                self.graph.nodes[node]['included'] = False
                
        # Keep track of merged nodes
        self.merged_history = {}
        for node in self.graph.nodes():
            self.merged_history[node] = [node]  # Initially, each node only contains itself
    
    def pendant_fold(self):
        """
        Perform one pendant vertex folding reduction on the graph.
        Updates the neighbor node with new values and removes the leaf vertex.
        Handles cases based on the inclusion status of the nodes.
        
        Returns:
            bool: True if a fold was performed; False otherwise.
        """
        for v in list(self.graph.nodes()):
            if self.graph.degree(v) == 1:
                # Get the neighbor
                nbrs = list(self.graph.neighbors(v))
                u = nbrs[0]
                
                # Get values and inclusion status
                select_u = self.graph.nodes[u]['select_value']
                nonselect_u = self.graph.nodes[u]['nonselect_value']
                select_v = self.graph.nodes[v]['select_value']
                nonselect_v = self.graph.nodes[v]['nonselect_value']
                included_u = self.graph.nodes[u]['included']
                included_v = self.graph.nodes[v]['included']
                
                # Update values based on inclusion status
                if not included_u and not included_v:
                    # Case: Both u and v are not in MIS
                    # Using the special case formula as requested
                    new_select = 0
                    new_nonselect = nonselect_u + nonselect_v
                    new_included = False  # Explicitly set to false when both are not included
                else:
                    # Standard pendant folding rule for cases where at least one node is included
                    new_select = select_u + nonselect_v
                    new_nonselect = nonselect_u + select_v
                    new_included = included_u  # Keep u's inclusion status
                
                # Update merged history
                self.merged_history[u] = self.merged_history.get(u, [u]) + self.merged_history.get(v, [v])
                
                # Update the neighbor node with new values
                self.graph.nodes[u]['select_value'] = new_select
                self.graph.nodes[u]['nonselect_value'] = new_nonselect
                self.graph.nodes[u]['included'] = new_included  # Set the updated inclusion status
                
                # Remove the leaf vertex
                self.graph.remove_node(v)
                
                return True
        return False
        
    def vertex_fold(self):
        """
        Perform one vertex folding reduction on the graph for degree-2 vertices.
        Creates a representative node and handles four different cases based on 
        inclusion status of neighbors.
        
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
                rep_node = min(u, v, w)
                
                # Get values and inclusion status
                select_u = self.graph.nodes[u]['select_value']
                nonselect_u = self.graph.nodes[u]['nonselect_value']
                select_v = self.graph.nodes[v]['select_value']
                nonselect_v = self.graph.nodes[v]['nonselect_value']
                select_w = self.graph.nodes[w]['select_value']
                nonselect_w = self.graph.nodes[w]['nonselect_value']
                included_u = self.graph.nodes[u]['included']
                included_w = self.graph.nodes[w]['included']
                included_v = self.graph.nodes[v]['included']
                
                # Handle different cases based on inclusion status
                
                # Case 1: Both u and w are in MIS
                if included_u and included_w:
                    new_select = select_u + select_w + nonselect_v
                    new_nonselect = 0
                    new_included = True
                    
                    # Connect to all external neighbors
                    external_nbrs = set()
                    for node in set(self.graph.neighbors(u)) | set(self.graph.neighbors(w)):
                        if node not in {u, v, w}:
                            external_nbrs.add(node)
                
                # Case 2: Neither u nor w are in MIS
                elif not included_u and not included_w:
                    # Get v's inclusion status
                    included_v = self.graph.nodes[v]['included']
                    
                    if not included_v:
                        # Subcase 1: None of u, v, w are in the MIS
                        new_select = 0
                        new_nonselect = nonselect_u + nonselect_v + nonselect_w
                        new_included = False
                    else:
                        # Original case: v is in the MIS
                        new_select = 0
                        new_nonselect = select_v + nonselect_u + nonselect_w
                        new_included = False
                    
                    # Connect to all external neighbors (same for both subcases)
                    external_nbrs = set()
                    for node in set(self.graph.neighbors(u)) | set(self.graph.neighbors(w)):
                        if node not in {u, v, w}:
                            external_nbrs.add(node)
                
                # Case 3: u is in MIS, w is not
                elif included_u and not included_w:
                    new_select = select_u + nonselect_v + nonselect_w
                    new_nonselect = 0
                    new_included = True
                    
                    # Connect only to u's neighbors (excluding v)
                    external_nbrs = set()
                    for node in self.graph.neighbors(u):
                        if node not in {v}:
                            external_nbrs.add(node)
                
                # Case 4: w is in MIS, u is not
                elif included_w and not included_u:  # not included_u and included_w
                    new_select = select_w + nonselect_v + nonselect_u
                    new_nonselect = nonselect_w + max(select_v, select_u)
                    new_included = True
                    
                    # Connect only to w's neighbors (excluding v)
                    external_nbrs = set()
                    for node in self.graph.neighbors(w):
                        if node not in {v}:
                            external_nbrs.add(node)
                else:
                    print("NOTA")
                
                # Log the select value if logger is available
                """if self.logger:
                    self.logger.log("Select value = "+str(new_select))"""
                
                # Update merged history
                merged_nodes = []
                for node in [u, v, w]:
                    merged_nodes.extend(self.merged_history.get(node, [node]))
                self.merged_history[rep_node] = merged_nodes
                
                # Remove all original nodes
                self.graph.remove_node(u)
                self.graph.remove_node(w)
                if v in self.graph:  # Check if v still exists
                    self.graph.remove_node(v)
                
                # Add the representative node with new values
                self.graph.add_node(rep_node, 
                                   select_value=new_select,
                                   nonselect_value=new_nonselect,
                                   included=new_included)
                
                # Connect representative to external neighbors
                for nbr in external_nbrs:
                    self.graph.add_edge(rep_node, nbr)
                
                return True
        return False
    
    def reduce_graph(self):
        """
        Reduce the graph by repeatedly applying folding operations until no more reductions are possible.
        Does not remove isolated vertices.
        
        Returns:
            tuple: (reduced graph, merged history)
        """
        initial_value = self.calculate_total_value()
        reduction_count = 0
        
        reduction_performed = True
        while reduction_performed:
            # Try pendant folding
            if self.pendant_fold():
                reduction_count += 1
                current_value = self.calculate_total_value()
                if initial_value != current_value:
                    print(f"Value changed after pendant fold #{reduction_count}: {initial_value} -> {current_value}")
                continue
                
            # Try vertex folding
            if self.vertex_fold():
                reduction_count += 1
                current_value = self.calculate_total_value()
                if initial_value != current_value:
                    print(f"Value changed after vertex fold #{reduction_count}: {initial_value} -> {current_value}")
                continue
                
            # If we reach here, no reductions were performed
            reduction_performed = False
        
        final_value = self.calculate_total_value()
        if initial_value != final_value:
            print(f"WARNING: Total value changed during reduction. Initial: {initial_value}, Final: {final_value}")
        
        return self.graph, self.merged_history
    
    def get_node_values(self):
        """
        Get the select and nonselect values for all nodes in the graph.
        
        Returns:
            dict: Dictionary mapping node IDs to (select_value, nonselect_value, included) tuples
        """
        result = {}
        for node in self.graph.nodes():
            result[node] = (
                self.graph.nodes[node]['select_value'],
                self.graph.nodes[node]['nonselect_value'],
                self.graph.nodes[node]['included']
            )
        return result
    
    def calculate_total_value(self):
        """
        Calculate the total value of the independent set based on current node selections.
        
        Returns:
            int: The total value of the independent set
        """
        total = 0
        for node in self.graph.nodes():
            if self.graph.nodes[node]['included']:
                total += self.graph.nodes[node]['select_value']
            else:
                total += self.graph.nodes[node]['nonselect_value']
        return total