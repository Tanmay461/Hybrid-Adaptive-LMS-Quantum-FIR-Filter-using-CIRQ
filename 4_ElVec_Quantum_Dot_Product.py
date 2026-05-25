import cirq
import numpy as np
import os
from cirq.contrib.svg import circuit_to_svg

# ==========================================
# 1. HARDWARE GATE DECOMPOSITIONS
# ==========================================

def manual_toffoli(c1, c2, target):
    """The 15-gate combination lock for a Toffoli gate."""
    T = cirq.T
    T_inv = cirq.T**-1 

    yield cirq.H(target)
    yield cirq.CNOT(c2, target)
    yield T_inv(target)
    yield cirq.CNOT(c1, target)
    yield T(target)
    yield cirq.CNOT(c2, target)
    yield T_inv(target)
    yield cirq.CNOT(c1, target)
    yield T(target)
    yield T(c2)
    yield cirq.CNOT(c1, c2)
    yield T_inv(c2)
    yield T(c1)
    yield cirq.CNOT(c1, c2)
    yield cirq.H(target)

def hardware_cswap(ancilla, target_a, target_b):
    """The 3-step physical SWAP operation."""
    yield cirq.CNOT(target_a, target_b)
    yield from manual_toffoli(ancilla, target_b, target_a)
    yield cirq.CNOT(target_a, target_b)

def manual_controlled_ry(theta, control, target):
    """
    Hardware decomposition of a Controlled-Ry gate.
    Sandwiches a reversed rotation between two CNOTs.
    """
    yield cirq.ry(theta / 2.0)(target)
    yield cirq.CNOT(control, target)
    yield cirq.ry(-theta / 2.0)(target)
    yield cirq.CNOT(control, target)


# ==========================================
# 2. DATA PREPARATION (4x1 VECTOR ALGEBRA)
# ==========================================

def calculate_angles_4d(vector):
    """Calculates normalization and rotation angles for a 4-element vector."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return 0.0, 0.0, 0.0, 0.0
        
    v = vector / norm
    prob_0 = v[0]**2 + v[1]**2
    
    alpha = np.arccos(np.clip(np.sqrt(prob_0), -1.0, 1.0))
    beta = np.arctan2(v[1], v[0]) if prob_0 > 0 else 0.0
    
    prob_1 = v[2]**2 + v[3]**2
    gamma = np.arctan2(v[3], v[2]) if prob_1 > 0 else 0.0
    
    return alpha, beta, gamma, norm

def hardware_state_prep_4d(q0, q1, alpha, beta, gamma):
    """Prepares the 2-qubit state using only hardware-native gates."""
    # Rotate Qubit 0
    yield cirq.ry(2 * alpha)(q0)
    
    # Conditionally rotate Qubit 1 if Q0 is |0> (Beta angle)
    yield cirq.X(q0)
    yield from manual_controlled_ry(2 * beta, q0, q1)
    yield cirq.X(q0)
    
    # Conditionally rotate Qubit 1 if Q0 is |1> (Gamma angle)
    yield from manual_controlled_ry(2 * gamma, q0, q1)


# ==========================================
# 3. THE MAIN QUANTUM ALGORITHM
# ==========================================

def execute_4d_quantum_dot_product(vec_x, vec_w, shots=20000):
    
    # 1. Math Prep
    alpha_x, beta_x, gamma_x, norm_x = calculate_angles_4d(vec_x)
    alpha_w, beta_w, gamma_w, norm_w = calculate_angles_4d(vec_w)
    
    # 2. Qubit Setup (5 wires total)
    ancilla = cirq.NamedQubit('Ancilla(C)')
    qx = [cirq.NamedQubit(f'VecX_Q{i}') for i in range(2)]
    qw = [cirq.NamedQubit(f'VecW_Q{i}') for i in range(2)]
    
    circuit = cirq.Circuit()
    
    # --- PHASE 1: Load the 4x1 Data ---
    circuit.append(hardware_state_prep_4d(qx[0], qx[1], alpha_x, beta_x, gamma_x))
    circuit.append(hardware_state_prep_4d(qw[0], qw[1], alpha_w, beta_w, gamma_w))
    
    # --- PHASE 2: Parallel Realities ---
    circuit.append(cirq.H(ancilla))
    
    # --- PHASE 3: The Hardware Swaps (Two pairs to swap!) ---
    circuit.append(hardware_cswap(ancilla, qx[0], qw[0]))
    circuit.append(hardware_cswap(ancilla, qx[1], qw[1]))
    
    # --- PHASE 4: Interference & Measurement ---
    circuit.append(cirq.H(ancilla))
    circuit.append(cirq.measure(ancilla, key='result'))

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    svg_path = os.path.join(BASE_DIR, "dot_product_circuit.svg")
    svg = circuit_to_svg(circuit)
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg)
    
    # 3. Execute the Simulation
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=shots)
    
    # 4. Reverse Engineer the Math
    measurements = result.measurements['result']
    prob_0 = np.sum(measurements == 0) / shots
    
    overlap = np.sqrt(max(0, 2 * prob_0 - 1))
    quantum_dot_product = overlap * norm_x * norm_w
    
    return quantum_dot_product, circuit

# ==========================================
# EXECUTION SCRIPT
# ==========================================
if __name__ == "__main__":
    # Your original 4x1 vectors!
    x = [0.4, 1.2, 4.6, 3.7]
    w = [0.6, 4.0, 5.2, 6.1]
    
    print(f"Vector X: {x}")
    print(f"Vector W: {w}")
    print("-" * 50)
    
    # Classical Baseline
    classical_dot = np.dot(x, w)
    print(f"Classical Target (Perfect Math): {classical_dot:.4f}")
    
    # Quantum Hardware Simulation
    # We use 50,000 shots because a 5-qubit circuit has more statistical variance
    print("\nExecuting 5-Qubit Hardware Simulation...")
    quantum_dot, full_circuit = execute_4d_quantum_dot_product(x, w, shots=50000)
    
    print(f"Quantum Result (Statistical):  {quantum_dot:.4f}")
