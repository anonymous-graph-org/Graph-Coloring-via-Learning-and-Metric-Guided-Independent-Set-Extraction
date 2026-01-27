"""
import gurobipy as gp
import networkx as nx

def gurobi_multiple_mis(graph, max_sets=32, time_limit=3600):

    Calculate multiple Maximum Independent Sets (MIS) for a given graph using Gurobi solver.
    Only returns solutions that match the size of the maximum independent set.
    Find multiple different maximum independent sets (all of the same maximum size).
    Parameters:
    - graph (networkx.Graph): The graph for which to compute the MIS.
    - max_sets (int): Maximum number of MIS to find.
    - time_limit (int): Time limit in seconds for Gurobi to find each solution.

    Returns:
    - list: A list of dictionaries, where each dictionary represents an MIS 
            (keys are node IDs, values are binary values: 1 for in MIS, 0 for not in MIS).

    options = {
        "WLSACCESSID": "e6e3aa6a-d667-48f2-8648-12da0a56a082",
        "WLSSECRET": "386f3716-8968-4355-9d5a-9514d4e65fa1",
        "LICENSEID": 2588402,
    }
    
    # Create a new model
    # env = gp.Env(params=options)
    #model = gp.Model(env=env)
    model=gp.Model()
    
    # Create variables for each node
    x = model.addVars(graph.nodes(), vtype=gp.GRB.BINARY, name="x")
    
    # Set objective: maximize the sum of selected nodes
    model.setObjective(gp.quicksum(x[i] for i in graph.nodes()), gp.GRB.MAXIMIZE)
    
    # Add constraints for adjacent nodes
    for (i, j) in graph.edges():
        model.addConstr(x[i] + x[j] <= 1, f"edge_{i}_{j}")
    
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
        current_solution = {i: int(x[i].x) for i in graph.nodes()}
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
    env.close()
    return solutions

def calculate_mis_with_gurobi(graph, time_limit=3600):

    Calculate the Maximum Independent Set (MIS) for a given graph using Gurobi solver.

    Parameters:
    - graph (networkx.Graph): The graph for which to compute the MIS.
    - time_limit (int): Time limit in seconds for Gurobi to find the solution.

    Returns:
    - dict: A dictionary where keys are node IDs, and values are binary values (1 for in MIS, 0 for not in MIS).


    options = {
        "WLSACCESSID": "e6e3aa6a-d667-48f2-8648-12da0a56a082",
        "WLSSECRET": "386f3716-8968-4355-9d5a-9514d4e65fa1",
        "LICENSEID": 2588402,
    }

    
    
    with gp.Env(params=options) as env, gp.Model("MIS", env=env) as model:
  
    
        
        model.setParam("TimeLimit", time_limit)  
        model.setParam("OutputFlag", 0)         


        node_vars = {node: model.addVar(vtype=gp.GRB.BINARY, name=f"x_{node}") for node in graph.nodes}
        
        
        model.setObjective(gp.quicksum(node_vars[node] for node in graph.nodes), gp.GRB.MAXIMIZE)
        

        for u, v in graph.edges:
            model.addConstr(node_vars[u] + node_vars[v] <= 1, f"edge_{u}_{v}")
        

        model.optimize()


        d = {node: int(node_vars[node].X > 0.5) for node in graph.nodes}

        return d
"""
import gurobipy as gp
import networkx as nx


def gurobi_multiple_mis(graph, max_sets=32, time_limit=3600):
    """
    Calculate multiple Maximum Independent Sets (MIS) for a given graph using Gurobi solver.
    Only returns solutions that match the size of the maximum independent set.
    """
    # Create a new model (local license will be used automatically)
    model = gp.Model("MIS")

    # Create variables for each node
    x = model.addVars(graph.nodes(), vtype=gp.GRB.BINARY, name="x")

    # Set objective: maximize the sum of selected nodes
    model.setObjective(gp.quicksum(x[i] for i in graph.nodes()), gp.GRB.MAXIMIZE)

    # Add constraints for adjacent nodes
    for (i, j) in graph.edges():
        model.addConstr(x[i] + x[j] <= 1, f"edge_{i}_{j}")

    # Set time limit
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 0)  # suppress solver output if desired

    solutions = []

    # First optimization to find MIS size
    model.optimize()
    if model.status != gp.GRB.OPTIMAL:
        return solutions

    max_size = int(model.objVal)

    # Add constraint that solutions must have max_size
    model.addConstr(gp.quicksum(x[i] for i in graph.nodes()) == max_size, "max_size_constraint")

    for k in range(max_sets):
        model.optimize()
        if model.status != gp.GRB.OPTIMAL:
            break

        current_solution = {i: int(x[i].X > 0.5) for i in graph.nodes()}
        solutions.append(current_solution)

        # Exclude this solution
        model.addConstr(
            gp.quicksum(x[i] for i in graph.nodes() if current_solution[i] == 1) <= max_size - 1,
            f"exclude_solution_{k}"
        )
        model.update()

    return solutions


def calculate_mis_with_gurobi(graph, time_limit=3600):
    """
    Calculate the Maximum Independent Set (MIS) for a given graph using Gurobi solver.
    """
    model = gp.Model("MIS")
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 0)

    node_vars = {node: model.addVar(vtype=gp.GRB.BINARY, name=f"x_{node}") for node in graph.nodes}
    model.setObjective(gp.quicksum(node_vars[node] for node in graph.nodes), gp.GRB.MAXIMIZE)

    for u, v in graph.edges:
        model.addConstr(node_vars[u] + node_vars[v] <= 1, f"edge_{u}_{v}")

    model.optimize()

    return {node: int(node_vars[node].X > 0.5) for node in graph.nodes}
