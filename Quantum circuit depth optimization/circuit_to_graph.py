#!/usr/bin/env python3
"""
Improved QASM Parser and Graph Builder for Quantum Circuit Optimization

This module parses QASM files and builds dependency graphs for quantum circuit optimization.
It correctly handles quantum-specific dependencies including entanglement tracking and
properly models commutativity relationships between quantum gates.
"""

import re
import os
import pickle
import networkx as nx
from collections import defaultdict

def is_parallel(prev_gate,current_gate):
  # Check if parallelization possible
  q1 = prev_gate['qubits']
  q2 = current_gate['qubits']

  # Disjoint qubits → parallel
  if set(q1).isdisjoint(q2):
    return True

  # CNOT–CNOT special case
  if prev_gate['op'] == 'cx' and current_gate['op'] == 'cx':
    ctrl1, tgt1 = q1[0], q1[1]
    ctrl2, tgt2 = q2[0], q2[1]

    # Same control, different targets → parallel
    if ctrl1 == ctrl2 and tgt1 != tgt2:
        return True

  return False

class QASMParser:
    def __init__(self, qasm_file):
        """Initialize the parser with a QASM file."""
        self.gates = []
        self.qubits = set()
        self.entanglement_graph = nx.Graph()
        self.parse_qasm(qasm_file)
        self.build_entanglement_graph()

    def parse_qasm(self, filename):
        """Parse QASM file into gate list with qubit information."""
        # Enhanced pattern to handle gates with parameters, measurements and reset operations
        gate_pattern = re.compile(r'(\w+)(?:\([^)]*\))?\s+((?:q\d*\[\d+\](?:,\s*q\d*\[\d+\])*))(?:\s*->.*)?;')
        measurement_pattern = re.compile(r'measure\s+(q\d*\[\d+\])\s*->\s*(c\d*\[\d+\]);')
        reset_pattern = re.compile(r'reset\s+(q\d*\[\d+\]);')

        # Track if we're within a gate definition block
        in_gate_def = False

        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                # Remove comments and strip whitespace
                line = line.split('//')[0].strip()
                if not line:
                    continue

                # Skip header and register declarations
                if line.startswith(('OPENQASM', 'include')):
                    continue

                # Handle register declarations
                if line.startswith('qreg') or line.startswith('creg'):
                    continue

                # Skip custom gate definitions
                if line.startswith('gate'):
                    in_gate_def = True
                    continue

                if in_gate_def:
                    if line == '}':
                        in_gate_def = False
                    continue

                # Skip barrier instructions - they don't contribute to execution dependencies
                # but are helpful for visualization
                if line.startswith('barrier'):
                    continue

                # Handle measurement operations
                measure_match = measurement_pattern.match(line)
                if measure_match:
                    qubit_str = measure_match.group(1)
                    classical_bit = measure_match.group(2)

                    q_match = re.search(r'q\d*\[(\d+)\]', qubit_str)
                    if q_match:
                        qubit = int(q_match.group(1))
                        gate_idx = len(self.gates)
                        self.gates.append({
                            'op': 'measure',
                            'qubits': [qubit],
                            'classical_bit': classical_bit,
                            'index': gate_idx
                        })
                        self.qubits.add(qubit)
                    continue

                # Handle reset operations
                reset_match = reset_pattern.match(line)
                if reset_match:
                    qubit_str = reset_match.group(1)

                    q_match = re.search(r'q\d*\[(\d+)\]', qubit_str)
                    if q_match:
                        qubit = int(q_match.group(1))
                        gate_idx = len(self.gates)
                        self.gates.append({
                            'op': 'reset',
                            'qubits': [qubit],
                            'index': gate_idx
                        })
                        self.qubits.add(qubit)
                    continue

                # Handle normal gate operations
                match = gate_pattern.match(line)
                if match:
                    op = match.group(1)
                    qubits_str = match.group(2)
                    qubits = []
                    for q in qubits_str.split(','):
                        q_clean = q.strip()
                        if '[' in q_clean:
                            # Extract number from q[XX] or q0[XX], etc.
                            q_match = re.search(r'q\d*\[(\d+)\]', q_clean)
                            if q_match:
                                qubit = int(q_match.group(1))
                                qubits.append(qubit)
                                self.qubits.add(qubit)

                    gate_idx = len(self.gates)
                    self.gates.append({
                        'op': op.lower(),  # Normalize to lowercase
                        'qubits': qubits,
                        'index': gate_idx
                    })
                else:
                    # Print debug info for lines that don't match
                    print(f"Warning: Line {line_num} did not match pattern: {line}")

    def build_entanglement_graph(self):
        """
        Build a graph tracking qubit entanglement throughout the circuit.
        This is important for correctly identifying dependencies between gates.
        """
        # Initialize graph with all qubits as nodes
        for qubit in self.qubits:
            self.entanglement_graph.add_node(qubit)

        # Process gates in order to track entanglement
        for gate in self.gates:
            op = gate['op']
            qubits = gate['qubits']

            # Two-qubit gates generally create entanglement
            entangling_gates = {'cx', 'cz', 'cp', 'cu1', 'cu2', 'cu3', 'swap', 'iswap', 'rxx', 'ryy', 'rzz', 'ccx',
                                'cswap'}
            if op in entangling_gates and len(qubits) >= 2:
                # Add edges between all pairs of qubits involved
                for i in range(len(qubits)):
                    for j in range(i + 1, len(qubits)):
                        self.entanglement_graph.add_edge(qubits[i], qubits[j])

            # Special case for gates that might break entanglement
            # This is a simplification - in reality, measuring or resetting qubits
            # can break entanglement but depends on the specific quantum state
            if op in {'measure', 'reset'} and len(qubits) == 1:
                # In a fully accurate simulation, we'd need to track the quantum state
                # For now, we'll assume measurements don't break entanglement to be conservative
                # A more sophisticated implementation could model measurement-induced collapse
                pass

    def are_qubits_entangled(self, q1, q2, gate_index):
        """
        Check if two qubits are entangled at a specific point in the circuit

        Args:
            q1, q2: Qubit indices to check
            gate_index: Index of gate where we want to check entanglement

        Returns:
            Boolean indicating if qubits are entangled
        """
        # If there's a path in the entanglement graph, they're entangled
        return q1 in self.entanglement_graph and q2 in self.entanglement_graph and \
            nx.has_path(self.entanglement_graph, q1, q2)

    def commute_on_entangled_qubits(self, gate1, gate2):
        """
        Check commutativity for gates on entangled but non-overlapping qubits.

        Args:
            gate1, gate2: Gate dictionaries to check

        Returns:
            Boolean indicating if gates commute despite entanglement
        """
        # Most gates on entangled qubits don't commute
        # unless they're from compatible bases

        # Z-basis operations commute with each other even on entangled qubits
        z_basis = {'z', 's', 't', 'rz', 'p', 'sdg', 'tdg'}
        if gate1['op'] in z_basis and gate2['op'] in z_basis:
            return True

        # X-basis operations generally don't commute with Z-basis on entangled qubits
        x_basis = {'x', 'rx'}
        if (gate1['op'] in z_basis and gate2['op'] in x_basis) or \
                (gate1['op'] in x_basis and gate2['op'] in z_basis):
            return False

        # By default, assume gates on entangled qubits don't commute
        return False

    def gates_commute(self, gate1, gate2):
        """
        Determine if two gates commute based on quantum gate properties.
        This function properly handles both direct qubit sharing and entanglement.
        """
        q1, q2 = set(gate1['qubits']), set(gate2['qubits'])
        shared = q1 & q2

        # No shared qubits - check for entanglement
        if not shared:
            # Check if any qubit in gate1 is entangled with any qubit in gate2
            for qubit1 in q1:
                for qubit2 in q2:
                    # Use the index of the earlier gate to check entanglement state
                    check_index = min(gate1['index'], gate2['index'])
                    if self.are_qubits_entangled(qubit1, qubit2, check_index):
                        # For entangled qubits, check specific commutation rules
                        return self.commute_on_entangled_qubits(gate1, gate2)
            # If no entanglement found, gates on different qubits commute
            return True

        # Single-qubit gate commutativity
        if len(q1) == 1 and len(q2) == 1:
            # Same operation type on same qubit (like multiple Z gates) commute
            if gate1['op'] == gate2['op'] and gate1['op'] in {'z', 's', 't', 'rz', 'p'}:
                return True

            # Different operations from same basis commute
            z_basis = {'z', 's', 't', 'sdg', 'tdg', 'rz', 'p'}
            x_basis = {'x', 'rx'}
            y_basis = {'y', 'ry'}

            if gate1['op'] in z_basis and gate2['op'] in z_basis:
                return True
            if gate1['op'] in x_basis and gate2['op'] in x_basis:
                return True
            if gate1['op'] in y_basis and gate2['op'] in y_basis:
                return True

        # CNOT gate special cases
        if gate1['op'] == 'cx' and gate2['op'] == 'cx':
            # Extract control and target qubits
            if len(gate1['qubits']) >= 2 and len(gate2['qubits']) >= 2:
                ctrl1, tgt1 = gate1['qubits'][0], gate1['qubits'][1]
                ctrl2, tgt2 = gate2['qubits'][0], gate2['qubits'][1]

                # CX gates with same control but different targets commute
                if ctrl1 == ctrl2 and tgt1 != tgt2:
                    return True

                # CX gates with same target but different controls commute
                if tgt1 == tgt2 and ctrl1 != ctrl2:
                    return True
                # control of one is target of the other → do NOT commute
                if ctrl1 == tgt2 or ctrl2 == tgt1:
                    return False

        # CNOT and Z-rotation on control commute
        if gate1['op'] == 'cx' and gate2['op'] in {'z', 'rz', 's', 't', 'p'}:
            ctrl = gate1['qubits'][0]
            if gate2['qubits'] == [ctrl]:
                return True

        if gate2['op'] == 'cx' and gate1['op'] in {'z', 'rz', 's', 't', 'p'}:
            ctrl = gate2['qubits'][0]
            if gate1['qubits'] == [ctrl]:
                return True

        # Special cases for controlled operations
        if 'cx' in {gate1['op'], gate2['op']}:
            cx_gate = gate1 if gate1['op'] == 'cx' else gate2
            other = gate2 if cx_gate == gate1 else gate1

            # Only process if other gate has qubits defined and cx gate has at least 2 qubits
            if len(other['qubits']) > 0 and len(cx_gate['qubits']) >= 2:
                ctrl, tgt = cx_gate['qubits'][0], cx_gate['qubits'][1]

                # Z-rotation on control commutes with CX
                if other['op'] in {'z', 's', 't', 'rz', 'p'} and len(other['qubits']) == 1:
                    if other['qubits'][0] == ctrl:
                        return True

                # X-rotation on target commutes with CX
                if other['op'] in {'x', 'rx'} and len(other['qubits']) == 1:
                    if other['qubits'][0] == tgt:
                        return True

        # Measurement operations don't commute with anything on the same qubit
        if gate1['op'] == 'measure' or gate2['op'] == 'measure':
            return False

        # Reset operations don't commute with anything on the same qubit
        if gate1['op'] == 'reset' or gate2['op'] == 'reset':
            return False

        # By default, gates sharing qubits don't commute
        return False

    def build_dependency_graph(self):
        """
        Build a dependency graph compatible with predict_colors.py
        - Directed edges for non-commuting gates (order matters)
        - Undirected edges for commuting gates that share qubits (can't execute simultaneously)
        This enhanced version properly handles entanglement-based dependencies.
        """
        # Use DiGraph as base, with special handling for undirected edges
        G = nx.DiGraph()

        # Add all gates as nodes
        for gate in self.gates:
            G.add_node(gate['index'], **gate)

        # For each gate, establish dependencies with previous gates
        for i, current_gate in enumerate(self.gates):
            current_qubits = set(current_gate['qubits'])
            current_idx = current_gate['index']

            # Check dependencies with all previous gates
            for j in range(i):
                prev_gate = self.gates[j]
                prev_qubits = set(prev_gate['qubits'])
                prev_idx = prev_gate['index']

                # Check for shared qubits or entanglement
                should_check_commutativity = False

                shared_qubits = current_qubits.intersection(prev_qubits)

                # If no shared qubits → NO EDGE, skip entirely
                if not shared_qubits:
                    continue
                # Direct qubit sharing
                if current_qubits.intersection(prev_qubits):
                    should_check_commutativity = True
                else:
                    # Check for entanglement between qubits
                    for q1 in current_qubits:
                        for q2 in prev_qubits:
                            if self.are_qubits_entangled(q1, q2, min(current_idx, prev_idx)):
                                should_check_commutativity = True
                                break
                        if should_check_commutativity:
                            break


                # If qubits interact (shared or entangled), check commutativity
                if should_check_commutativity:
                    try:
                        if self.gates_commute(current_gate, prev_gate):
                            # For commuting gates, add bidirectional edges with directed=False
                            # Note: This attribute is used by predict_colors.py to identify undirected edges
                            if not is_parallel(prev_gate, current_gate):

                              G.add_edge(prev_idx, current_idx, directed=False)
                            #G.add_edge(current_idx, prev_idx, directed=False)
                        else:
                            # For non-commuting gates, add a directed edge (j must execute before i)
                            G.add_edge(prev_idx, current_idx, directed=True)
                    except Exception as e:
                        print(f"Error in dependency check between gates {prev_idx} and {current_idx}: {e}")
                        # Add a directed edge by default to ensure correctness
                        G.add_edge(prev_idx, current_idx, directed=True)
        # Print edges with attributes
        print("Edges in the graph:")
        for u, v, d in G.edges(data=True):
            print(f"{u} -> {v}, directed = {d.get('directed')}")

        return G


def process_qasm_folder(folder_path, output_pkl="qasm_graphs.pkl"):
    """Process all QASM files in a folder and save graphs to a pickle file"""
    # Just save the graphs directly without nesting in another dictionary
    graphs_dict = {}

    # Find all .qasm files in the folder
    qasm_files = [f for f in os.listdir(folder_path) if f.endswith('.qasm')]

    print(f"Found {len(qasm_files)} QASM files")

    for qasm_file in sorted(qasm_files):
        file_name = os.path.splitext(qasm_file)[0]
        file_path = os.path.join(folder_path, qasm_file)

        try:
            parser = QASMParser(file_path)
            print(f"Found {len(parser.gates)} gates in {file_name}")

            if len(parser.gates) == 0:
                print(f"Warning: No gates found in {file_name}. Skipping.")
                continue

            graph = parser.build_dependency_graph()

            # Verify graph consistency
            missing_nodes = set(range(len(parser.gates))) - set(graph.nodes())
            if missing_nodes:
                print(f"Warning: Missing nodes in graph: {missing_nodes}")

            # Repair the graph if needed
            for i in range(len(parser.gates)):
                if i not in graph:
                    print(f"Repairing graph: Adding missing node {i}")
                    graph.add_node(i, **parser.gates[i])

            # Store metadata for better analysis
            graph.graph['circuit_name'] = file_name
            graph.graph['num_qubits'] = len(parser.qubits)
            graph.graph['num_gates'] = len(parser.gates)

            # Store just the graph object with circuit name as key
            graphs_dict[file_name] = graph

            print(f"Processed {file_name}: {len(parser.gates)} gates, "
                  f"{graph.number_of_edges()} edges, {len(parser.qubits)} qubits")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save only the graphs dictionary to the pickle file
    with open(output_pkl, 'wb') as f:
        pickle.dump(graphs_dict, f)

    print(f"Saved {len(graphs_dict)} graphs to {output_pkl}")
    return graphs_dict


def analyze_graph_properties(graph_dict):
    """Analyze properties of the dependency graphs for reporting in paper"""
    results = {}

    for name, graph in graph_dict.items():
        # Calculate directed and undirected edge counts
        directed_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get('directed', True)]
        undirected_edges = [(u, v) for u, v, d in graph.edges(data=True) if not d.get('directed', True)]

        # Count unique undirected edges (avoid double counting)
        unique_undirected = set()
        for u, v in undirected_edges:
            if (v, u) not in unique_undirected:
                unique_undirected.add((u, v))

        # Gather statistics
        results[name] = {
            'nodes': graph.number_of_nodes(),
            'directed_edges': len(directed_edges),
            'undirected_edges': len(unique_undirected),
            'total_unique_edges': len(directed_edges) + len(unique_undirected),
            'density': nx.density(graph)
        }

        # Calculate potential parallelism (theoretical)
        """try:
            longest_path = nx.dag_longest_path_length(graph)
            results[name]['min_depth'] = longest_path + 1  # +1 because path length counts edges
            results[name]['parallelism_ratio'] = graph.number_of_nodes() / (longest_path + 1)
        except nx.NetworkXError:
            # The graph might have cycles due to undirected edges
            results[name]['min_depth'] = "Unknown (cycles)"
            results[name]['parallelism_ratio'] = "Unknown" """

    return results


# Example usage - using the same paths as the original code
if __name__ == "__main__":
    folder_path = "quantum_circuit_instances"  # Look in the QASM_graphs directory by default
    output_file = "qasm_graphs_original.pkl"  # Save to current directory by default
    graphs_dict = process_qasm_folder(folder_path, output_file)

    # Optionally analyze and print graph properties for the paper
    analysis = analyze_graph_properties(graphs_dict)
    print("\nDependency Graph Analysis (useful for your paper):")
    print("==================================================")
    for circuit, stats in analysis.items():
        print(f"\nCircuit: {circuit}")
        for key, value in stats.items():
            print(f"  {key}: {value}")

