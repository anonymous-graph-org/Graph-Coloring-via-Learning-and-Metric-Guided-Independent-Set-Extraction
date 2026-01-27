import networkx as nx

class GraphReducer:
    def __init__(self, graph):
        """
        Initialize the GraphReducer class.
        
        Args:
            graph: NetworkX graph to be reduced
        """
        self.original_graph = graph.copy()
        self.graph = graph.copy()
        self.original_nodes = len(graph.nodes())
        self.original_edges = len(graph.edges())
        
        
        # Target is to reduce nodes by 50% from original
        self.target_nodes = self.original_nodes * 0.7
        
        # Maximum number of iterations
        self.max_iterations = 2000
        
        # Create a mapping to track merged nodes
        self.merged_nodes = {node: [node] for node in self.graph.nodes()}
        
        # Track tried nodes to avoid revisiting immediately
        self.tried_nodes = set()
    
    def get_nodes_by_degree(self):
        """Get all nodes sorted by degree in descending order."""
        return sorted(self.graph.nodes(), key=lambda x: self.graph.degree(x), reverse=True)
    
    def find_independent_set(self, node_set):
        """
        Find a maximal independent set in the node_set using a greedy approach.
        
        Args:
            node_set: Set of nodes to find independent set from
            
        Returns:
            Set of nodes forming a maximal independent set
        """
        node_set = set(node_set)
        independent_set = set()
        
        # Sort nodes by degree in decreasing order
        nodes_by_degree = sorted(node_set, key=lambda x: self.graph.degree(x), reverse=True)
        
        for v in nodes_by_degree:
            if v not in node_set:
                continue
                
            # Add it to the independent set
            independent_set.add(v)
            
            # Remove v and its neighbors from consideration
            neighbors = set(self.graph.neighbors(v)) & node_set
            node_set.remove(v)
            node_set -= neighbors
            
            if not node_set:
                break
            
        return independent_set
    
    def merge_nodes(self, nodes_to_merge):
        """
        Merge a set of nodes into a single representative node.
        The node with the lowest ID becomes the representative.
        
        Args:
            nodes_to_merge: Set of nodes to merge
            
        Returns:
            The representative node ID
        """
        if not nodes_to_merge:
            return None
            
        nodes_to_merge = list(nodes_to_merge)
        if len(nodes_to_merge) == 1:
            return nodes_to_merge[0]
            
        # Choose the representative node (lowest node ID)
        rep_node = min(nodes_to_merge)
        
        # Create a new merged node list
        for node in nodes_to_merge:
            if node != rep_node:
                # Update the merged nodes mapping
                self.merged_nodes[rep_node].extend(self.merged_nodes[node])
                del self.merged_nodes[node]
                
                # Redirect edges to the representative node
                for neighbor in list(self.graph.neighbors(node)):
                    if neighbor not in nodes_to_merge:
                        self.graph.add_edge(rep_node, neighbor)
                
                # Remove the merged node
                self.graph.remove_node(node)
        
        return rep_node
    
    def reduce_step(self, node=None):
        """
        Perform one step of graph reduction.
        
        Args:
            node: Specific node to use as highest degree node (if None, find it)
            
        Returns:
            True if reduction was performed, False if no further reduction is possible
        """
        # Track initial graph state to detect if any change occurs
        initial_nodes = len(self.graph.nodes())
        initial_edges = len(self.graph.edges())
        
        # 1. Identify node with highest degree or use provided node
        if node is None:
            high_degree_nodes = self.get_nodes_by_degree()
            if not high_degree_nodes:
                return False
            v = high_degree_nodes[0]
        else:
            v = node
        
        # Track that we tried this node
        self.tried_nodes.add(v)
            
        # 2. Identify S1 (neighbors of v) and S2 (neighbors of neighbors, but not in S1)
        if v not in self.graph:
            return False
            
        S1 = set(self.graph.neighbors(v))
        if not S1:
            return False
            
        S2 = set()
        for node in S1:
            S2.update(self.graph.neighbors(node))
        S2 -= S1
        S2.discard(v)
        
        # 3a. Find independent set I1 in S1 and merge
        I1 = self.find_independent_set(S1)
        self.merge_nodes(I1)
        
        # 3b. Find independent set I2 in S2 and merge
        if S2:
            I2 = self.find_independent_set(S2)
            self.merge_nodes(I2)
            
        # Check if any reduction actually happened
        return len(self.graph.nodes()) < initial_nodes or len(self.graph.edges()) < initial_edges
    
    def reduce_graph(self):
        """
        Reduce the graph until node count is reduced by 50% or maximum iterations is reached.
        Systematically tries all nodes by degree order when progress stalls.
        
        Returns:
            The reduced graph
        """
        iteration = 0
        
        # Initialize previous state for tracking progress
        prev_node_count = len(self.graph.nodes())
        stalled_count = 0
        
        while iteration < self.max_iterations:
            # Stop if we've reached the target node reduction (50%)
            current_nodes = len(self.graph.nodes())
            if current_nodes <= self.target_nodes:
                break
                
            # Get list of nodes by degree
            nodes_by_degree = self.get_nodes_by_degree()
            if not nodes_by_degree:
                break
                
            # If we're making progress normally, try the highest degree node
            if stalled_count == 0:
                node = nodes_by_degree[0]
                success = self.reduce_step(node)
                self.tried_nodes.add(node)
            else:
                # We're stalled, try nodes systematically by degree order
                success = False
                for node in nodes_by_degree:
                    if node not in self.tried_nodes:
                        success = self.reduce_step(node)
                        self.tried_nodes.add(node)
                        if success:
                            break
            
            # Check if we made progress
            current_node_count = len(self.graph.nodes())
            if current_node_count < prev_node_count:
                # Progress made
                stalled_count = 0
                self.tried_nodes = set()  # Reset tried nodes since graph has changed
                prev_node_count = current_node_count
            else:
                # No progress
                stalled_count += 1
                
                # If we've tried all nodes without progress, we're truly stuck
                if len(self.tried_nodes) >= len(nodes_by_degree):
                    break
            
            iteration += 1
        
        return self.graph
    
    def get_merged_nodes_mapping(self):
        """
        Returns the mapping of merged nodes.
        
        Returns:
            Dictionary mapping representative nodes to the list of original nodes they represent
        """
        return self.merged_nodes