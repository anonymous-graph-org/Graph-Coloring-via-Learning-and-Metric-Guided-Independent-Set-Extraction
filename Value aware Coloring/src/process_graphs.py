import pickle
import networkx as nx
import time
import os
from load_graphs import load_graphs
from value_aware_reduction import ValueAwareGraph
from gurobi_mis import gurobi_multiple_mis
# from load_model import load_value_aware_model
from predict import predict_value_aware_mis
import config
import torch

class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
    
    def log(self, message):
        print(message, file=self.log_file)
        print(message)  # Also print to console

def process_all_graphs():
    """
    Process all graphs, find multiple MIS solutions, and generate reduced labels.
    Logs all important information to output.txt.
    Checks if the total value of reduced graph MIS matches the original MIS size.
    """
    # Open a log file to record important information
    with open("output.txt", "w") as log_file:
        # Create a logger instance
        logger = Logger(log_file)
        
        logger.log("VALUE-AWARE MIS GENERATION PROCESS")
        logger.log("=" * 50 + "\n")
        
        start_time = time.time()
        logger.log(f"Process started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        logger.log("Loading graphs from graphs.pickle...")
        graphs = load_graphs("/Users/ACER/Desktop/basicMIS_multiplesolns/graphs.pickle")
        logger.log(f"Loaded {len(graphs)} graphs\n")
        
        print(f"Loaded {len(graphs)} graphs")
        
        all_results = []
        total_mis_solutions = 0
        total_reduced_nodes = 0
        total_original_nodes = 0
        total_mismatches = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        """model = load_value_aware_model("/Users/prishajain/Desktop/MTP/ValueAware_MIS/model_parameters/value_aware_gcn.pth", feature_dim=config.FEATURE_DIM,
                                   hidden_dim=config.HIDDEN_DIM, num_layers=config.NUM_LAYERS,
                                   num_maps=config.NUM_MAPS, device=device)"""
        
        for graph_idx, graph in enumerate(graphs):
            logger.log(f"Graph {graph_idx+1}/{len(graphs)}")
            logger.log(f"  Original size: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
            print(f"Processing graph {graph_idx+1}/{len(graphs)} with {graph.number_of_nodes()} nodes...")
            
            # Find multiple MIS solutions using Gurobi
            try:
                logger.log(f"  Finding MIS solutions...")
                mis_start_time = time.time()
                mis_solutions = gurobi_multiple_mis(graph, max_sets=32)  # Find up to 32 MIS solutions
                mis_time = time.time() - mis_start_time
                
                logger.log(f"  Found {len(mis_solutions)} MIS solutions in {mis_time:.2f} seconds")
                print(f"Found {len(mis_solutions)} MIS solutions")
                
                # If no MIS solutions found, skip this graph
                if not mis_solutions:
                    logger.log(f"  No MIS solutions found, skipping graph\n")
                    print(f"No MIS solutions found for graph {graph_idx+1}, skipping...")
                    continue
                
                graph_results = {
                    'graph_idx': graph_idx,
                    'original_graph': graph,
                    'solutions': []
                }
                
                # Process each MIS solution
                for solution_idx, mis_solution in enumerate(mis_solutions):
                    logger.log(f"  Solution {solution_idx+1}/{len(mis_solutions)}:")
                    print(f"Processing MIS solution {solution_idx+1}/{len(mis_solutions)}...")
                    
                    # Calculate original MIS size
                    original_mis_size = sum(1 for value in mis_solution.values() if value == 1)
                    logger.log(f"  Original MIS size: {original_mis_size} nodes")
                    
                    # Initialize value-aware graph with this MIS solution and pass the logger
                    value_graph = ValueAwareGraph(graph, mis_solution, logger)
                    
                    # Apply graph reduction
                    reduction_start_time = time.time()
                    reduced_graph, merged_history = value_graph.reduce_graph()
                    reduction_time = time.time() - reduction_start_time
                    
                    # Get node values from the reduced graph
                    node_values = value_graph.get_node_values()
                    
                    # Calculate total value using the reduced graph
                    # Correctly calculate the total value by iterating through nodes in the reduced graph
                    total_value = 0
                    for node in reduced_graph.nodes():
                        if reduced_graph.nodes[node]['included']:
                            total_value += reduced_graph.nodes[node]['select_value']
                        else:
                            total_value += reduced_graph.nodes[node]['nonselect_value']
                            
                    
                    # Check if total value matches original MIS size
                    is_match = abs(total_value - original_mis_size) == 0  # Ensure exact match
                    
                    if not is_match:
                        error_msg = f"WRONG: Graph {graph_idx+1}, Solution {solution_idx+1}: Original MIS size = {original_mis_size}, Total value = {total_value}"
                        logger.log(f"  {error_msg}")
                        print(error_msg)
                        total_mismatches += 1
                    else:
                        logger.log(f"  CORRECT: Original MIS size = {original_mis_size}, Total value = {total_value}")
                    
                    # Store results for this solution
                    solution_result = {
                        'mis_solution': mis_solution,
                        'reduced_graph': reduced_graph,
                        'node_values': node_values,
                        'merged_history': merged_history,
                        'total_value': total_value,
                        'original_mis_size': original_mis_size,
                        'is_value_correct': is_match
                    }
                    
                    graph_results['solutions'].append(solution_result)
                    
                    # Log reduction details
                    reduction_ratio = 1 - reduced_graph.number_of_nodes() / graph.number_of_nodes()
                    logger.log(f"    Original nodes: {graph.number_of_nodes()}, Reduced nodes: {reduced_graph.number_of_nodes()}")
                    logger.log(f"    Reduction ratio: {reduction_ratio:.2f}")
                    logger.log(f"    Reduction time: {reduction_time:.2f} seconds")
                    
                    total_mis_solutions += 1
                    total_original_nodes += graph.number_of_nodes()
                    total_reduced_nodes += reduced_graph.number_of_nodes()
                
                logger.log("\n")
                all_results.append(graph_results)
                
            except Exception as e:
                logger.log(f"  Error processing graph {graph_idx+1}: {str(e)}\n")
                print(f"Error processing graph {graph_idx+1}: {str(e)}")
                continue
        
        # Calculate and log overall statistics
        avg_reduction_ratio = 1 - (total_reduced_nodes / (total_original_nodes)) if total_original_nodes > 0 else 0
        
        logger.log("\nOVERALL STATISTICS")
        logger.log("=" * 50)
        logger.log(f"Total graphs processed: {len(all_results)}/{len(graphs)}")
        logger.log(f"Total MIS solutions found: {total_mis_solutions}")
        logger.log(f"Total solutions with mismatched values: {total_mismatches}")
        logger.log(f"Percentage of correct solutions: {(total_mis_solutions - total_mismatches) / total_mis_solutions * 100:.2f}% if total_mis_solutions > 0 else 0")
        logger.log(f"Average reduction ratio: {avg_reduction_ratio:.4f}")
        logger.log(f"Total process time: {time.time() - start_time:.2f} seconds")
        
        # Save all results to pickle file
        logger.log("\nSaving results to reduced_labels.pickle...")
        with open("reduced_labels.pickle", 'wb') as f:
            pickle.dump(all_results, f)
        
        logger.log(f"Processed {len(all_results)} graphs successfully")
        logger.log(f"Process completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"Processed {len(all_results)} graphs successfully")
    print(f"Total solutions with mismatched values: {total_mismatches}")
    if total_mis_solutions > 0:
        print(f"Percentage of correct solutions: {(total_mis_solutions - total_mismatches) / total_mis_solutions * 100:.2f}%")
    print(f"Details logged to output.txt")

if __name__ == "__main__":
    process_all_graphs()