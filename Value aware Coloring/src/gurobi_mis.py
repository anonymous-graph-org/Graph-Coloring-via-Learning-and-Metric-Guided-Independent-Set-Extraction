import gurobipy as gp
import networkx as nx

def gurobi_multiple_mis(graph, max_sets=32, time_limit=3600):
    # Create a new model (local license will be used automatically)
    model = gp.Model("MIS")
    model.Params.OutputFlag = 0  # <--- Disable Gurobi console output
    
    # Create variables for each node
    x = model.addVars(graph.nodes(), vtype=gp.GRB.BINARY, name="x")
    
    # Set objective: maximize the sum of selected nodes
    model.setObjective(gp.quicksum(x[i] for i in graph.nodes()), gp.GRB.MAXIMIZE)
    
    # Add constraints for adjacent nodes
    for (i, j) in graph.edges():
        model.addConstr(x[i] + x[j] <= 1, f"edge_{i}_{j}")
    
    # Add maximality constraints for pendant vertices
    for v in graph.nodes():
        if graph.degree(v) == 1:
            # v is a pendant vertex
            u = list(graph.neighbors(v))[0]  # The single neighbor
            # Either v or u must be in the MIS
            model.addConstr(x[v] + x[u] >= 1, f"pendant_{v}_{u}")
    
    # Add maximality constraints for degree-2 vertices with non-adjacent neighbors
    for v in graph.nodes():
        if graph.degree(v) == 2:
            neighbors = list(graph.neighbors(v))
            u, w = neighbors
            # If u and w are not adjacent, then at least one of u, v, w must be in the MIS
            if not graph.has_edge(u, w):
                model.addConstr(x[u] + x[v] + x[w] >= 1, f"deg2_{u}_{v}_{w}")
    
    # Set time limit
    model.setParam('TimeLimit', time_limit)
    
    # Store all solutions
    solutions = []
    
    # First, find the maximum independent set size
    model.optimize()
    if model.status != gp.GRB.OPTIMAL:
        return solutions
        
    max_size = model.objVal
    
    # Add constraint to ensure we only get solutions of the maximum size
    model.addConstr(gp.quicksum(x[i] for i in graph.nodes()) == max_size, "max_size_constraint")
    
    # Find multiple solutions
    for k in range(max_sets):
        # Solve the model
        model.optimize()
        
        # Check if a solution was found
        if model.status != gp.GRB.OPTIMAL:
            break
            
        # Get the current solution
        current_solution = {i: int(x[i].x + 0.5) for i in graph.nodes()}  # Adding 0.5 to handle floating-point precision issues
        solutions.append(current_solution)
        # Add constraint to exclude this solution
        model.addConstr(
            gp.quicksum(x[i] for i in graph.nodes() if current_solution[i] == 1) <= 
            max_size - 1,
            f"exclude_solution_{k}"
        )
        
        # Update the model
        model.update()
    
    # Close the Gurobi environment
    # env.close()
    return solutions


def calculate_mis_with_gurobi(graph, time_limit=3600):
    """
    Calculate the Maximum Independent Set (MIS) for a given graph using Gurobi solver.
    """
    model = gp.Model("MIS")
    model.Params.OutputFlag = 0  # <--- Disable Gurobi console output
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 0)

    node_vars = {node: model.addVar(vtype=gp.GRB.BINARY, name=f"x_{node}") for node in graph.nodes}
    model.setObjective(gp.quicksum(node_vars[node] for node in graph.nodes), gp.GRB.MAXIMIZE)

    # Add constraints for adjacent nodes
    for u, v in graph.edges:
        model.addConstr(node_vars[u] + node_vars[v] <= 1, f"edge_{u}_{v}")

    # Add maximality constraints for pendant vertices
    for v in graph.nodes:
        if graph.degree(v) == 1:
            # v is a pendant vertex
            u = list(graph.neighbors(v))[0]  # The single neighbor
            # Either v or u must be in the MIS
            model.addConstr(node_vars[v] + node_vars[u] >= 1, f"pendant_{v}_{u}")

    # Add maximality constraints for degree-2 vertices with non-adjacent neighbors
    for v in graph.nodes:
        if graph.degree(v) == 2:
            neighbors = list(graph.neighbors(v))
            u, w = neighbors
            # If u and w are not adjacent, then at least one of u, v, w must be in the MIS
            if not graph.has_edge(u, w):
                model.addConstr(node_vars[u] + node_vars[v] + node_vars[w] >= 1, f"deg2_{u}_{v}_{w}")

    model.optimize()

    return {node: int(node_vars[node].X > 0.5) for node in graph.nodes}

