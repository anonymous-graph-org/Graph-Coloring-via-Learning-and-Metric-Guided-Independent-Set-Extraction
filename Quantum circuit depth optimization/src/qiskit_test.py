import os
import glob
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

def calculate_circuit_depths(folder_path, output_file="qiskit.txt"):
    # calculates the depth of each QASM file using Qiskit's standard method and save results to a text file.
   
    # Get all .qasm files in the folder
    qasm_files = glob.glob(os.path.join(folder_path, "*.qasm"))
    
    # Open the output file for writing
    with open(output_file, "w") as f:
        f.write("Circuit Name\tDepth\n")
        f.write("-" * 30 + "\n")
        
        # Process each QASM file
        for qasm_file in qasm_files:
            try:
                # Get the filename without path
                filename = os.path.basename(qasm_file)
                
                # Load the QASM file
                circuit = QuantumCircuit.from_qasm_file(qasm_file)
                dag = circuit_to_dag(circuit)
                # Calculate depth
                depth = dag.depth()
                
                # Write results to the file
                f.write(f"{filename}\t{depth}\n")
                
                print(f"Processed {filename}: Depth = {depth}")
            except Exception as e:
                print(f"Error processing {qasm_file}: {e}")
    
    print(f"\nResults saved to {output_file}")

def main():
    # Path to the folder containing QASM files
    folder_path = "quantum_circuit_instances" 
    
    # Output file for results
    output_file = "qiskit.txt"
    
    # Calculate depths and save results
    calculate_circuit_depths(folder_path, output_file)

if __name__ == "__main__":
    main()
