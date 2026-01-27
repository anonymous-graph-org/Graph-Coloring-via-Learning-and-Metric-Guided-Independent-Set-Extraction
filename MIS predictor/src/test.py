from src.load_model import load_model  # Import the load_model function
from src.gurobi_mis import calculate_mis_with_gurobi
from src.model import DeepGCN
from src.predict import predict_mis

def test_model(graphs,model_path,hidden_dim,num_layers):
    print("Testing...")
    avg_accuracy = 0
    count = 0
    model = load_model(model_path, DeepGCN, hidden_dim, num_layers)
    for i,graph in enumerate(graphs):
        print("Graph "+str(i+1))
        predicted_mis = predict_mis(model,graph)
        gurobi_mis = calculate_mis_with_gurobi(graph)

        print("predicted mis: ")
        print(predicted_mis)
        print("gurobi mis")
        print(gurobi_mis)
        
        len_gurobi = 0
        len_predicted = 0
        for nd,val in predicted_mis.items():
            len_predicted+=val
        for nd,val in gurobi_mis.items():
            len_gurobi+=val
        accuracy = len_predicted/len_gurobi
        print("len gurobi = "+str(len_gurobi))
        print("len predicted = "+str(len_predicted))
        print("Accuracy = "+str(accuracy))
        avg_accuracy+=accuracy
        count+=1
        print("Average accuracy = "+str(avg_accuracy/count))
        print("____________________________________________________________________________")


