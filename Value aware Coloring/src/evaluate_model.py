import pickle
import torch
import networkx as nx
from src.load_model import load_value_aware_model
from src.predict import predict_value_aware_mis
from src.value_aware_reduction import ValueAwareGraph
from src. simple_reduction import SimplifiedValueAwareGraph

def evaluate_model(model, test_data, config, logger, device='cpu'):
    logger.log(f"Evaluating model...")

    total_value_pred = 0
    total_value_gt = 0
    total_graphs = len(test_data)

    per_graph_value_ratios = []

    for idx, data in enumerate(test_data):
        # Get the original graph and MIS solution
        original_graph = data['original_graph']
        ground_truth = data['mis_solutions'][0]  # Use first MIS as ground-truth
        
        # Log graph information
        logger.log(f"\nGraph {idx+1}/{total_graphs}: Original size: {original_graph.number_of_nodes()} nodes, {original_graph.number_of_edges()} edges")
        
        # Create a value-aware graph with the ground truth MIS solution - pass the logger
        value_graph_gt = ValueAwareGraph(original_graph, ground_truth, logger)
        
        # Calculate ground truth value using the value-aware graph
        gt_value = value_graph_gt.calculate_total_value()
        
        # Create a value-aware graph for prediction (without providing MIS solution) - pass the logger
        value_graph_pred = SimplifiedValueAwareGraph(original_graph, logger=logger)
        
        # Reduce the graph
        red_graph, merged_history = value_graph_pred.reduce_graph()
        
        # Make prediction using the reduced graph
        prediction = predict_value_aware_mis(model, red_graph, time_budget=config.TIME_BUDGET, 
                                           num_maps=config.NUM_MAPS, device=device)
        
        # Calculate prediction value directly
        pred_value = 0
        mis_size = 0
        
        for node in red_graph.nodes():
            if prediction.get(node, 0) == 1:
                pred_value += red_graph.nodes[node]['select_value']
                mis_size += 1 
            else:
                pred_value += red_graph.nodes[node]['nonselect_value']
                
        # Add to totals
        total_value_pred += pred_value
        total_value_gt += gt_value

        # Calculate ratio
        value_ratio_graph = pred_value / gt_value if gt_value > 0 else 0
        per_graph_value_ratios.append(value_ratio_graph)

        # Log prediction results
        logger.log(f"Prediction results:")
        logger.log(f"  MIS size: {mis_size} nodes")
        logger.log(f"  Prediction value: {pred_value:.2f}")
        logger.log(f"  Ground truth value: {gt_value:.2f}")
        logger.log(f"  Value ratio: {value_ratio_graph:.4f}")
        
       
    # Calculate overall value ratio
    value_ratio_total = total_value_pred / total_value_gt if total_value_gt > 0 else 0
    logger.log(f"\nOverall Results:")
    logger.log(f"Total Value Ratio (predicted/gt): {value_ratio_total:.4f}")

    # Save results
    results = {
        'value_ratio_total': value_ratio_total,
        'per_graph_value_ratios': per_graph_value_ratios,
    }
    with open(config.RESULTS_PATH, 'wb') as f:
        pickle.dump(results, f)

    logger.log(f"Saved evaluation results to {config.RESULTS_PATH}")