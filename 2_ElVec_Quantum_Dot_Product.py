import cirq
import numpy as np
import os
from cirq.contrib.svg import circuit_to_svg

# ==========================================
# PHASE 3: THE HARDWARE GATE DECOMPOSITIONS
# ==========================================

def manual_toffoli(c1, c2, target):
    """
    The 15-gate 'Combination Lock' that flips the target 
    ONLY if c1 and c2 are both 1.
    Uses only H, T, T-dagger, and CNOT.
    """
    T = cirq.T
    T_inv = cirq.T**-1  # This is the negative 45-degree rotation

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
    """
    The 3-step physical SWAP operation using our scratchpad logic.
    """
    # 1. Entangle (Store difference in Target B)
    yield cirq.CNOT(target_a, target_b)
    
    # 2. Conditional Swap (using the Combination Lock)
    # Notice: ancilla and target_b act as the controls!
    yield from manual_toffoli(ancilla, target_b, target_a)
    
    # 3. Un-entangle (Clean up the garbage data in Target B)
    yield cirq.CNOT(target_a, target_b)


# ==========================================
# PHASE 1: DATA PREPARATION
# ==========================================

def encode_2d_vector(vector):
    """
    Calculates the magnitude and the Y-axis rotation angle
    needed to store a 2-element array in a single qubit.
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return 0.0, 0.0
    
    v = vector / norm
    
    # For a 1-qubit state: cos(theta/2)|0> + sin(theta/2)|1>
    # We multiply by 2 because the Ry gate rotates by theta/2
    theta = 2 * np.arctan2(v[1], v[0])
    
    return theta, norm


# ==========================================
# THE MAIN QUANTUM ALGORITHM
# ==========================================

def execute_quantum_dot_product(vec_x, vec_w, shots=20000):
    
    # 1. Math Prep
    theta_x, norm_x = encode_2d_vector(vec_x)
    theta_w, norm_w = encode_2d_vector(vec_w)

    print(theta_x)
    print(theta_w)
    
    # 2. Qubit Setup (The 3 wires)
    ancilla = cirq.NamedQubit('Ancilla(C)')
    ta = cirq.NamedQubit('Target_A(Ta)')
    tb = cirq.NamedQubit('Target_B(Tb)')
    
    circuit = cirq.Circuit()
    
    # --- PHASE 1: Load the Data ---
    circuit.append(cirq.ry(theta_x)(ta))
    circuit.append(cirq.ry(theta_w)(tb))
    
    # --- PHASE 2: Parallel Realities ---
    circuit.append(cirq.H(ancilla))
    
    # --- PHASE 3: The Hardware Swap ---
    circuit.append(hardware_cswap(ancilla, ta, tb))
    
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
    
    # Extract overlap from P(0)
    overlap = np.sqrt(max(0, 2 * prob_0 - 1))
    
    # Re-apply classical magnitudes
    quantum_dot_product = overlap * norm_x * norm_w
    
    return quantum_dot_product, circuit

# ==========================================
# EXECUTION SCRIPT
# ==========================================
if __name__ == "__main__":
    # Generate two random 2-element vectors (positive numbers for simplicity)
    np.random.seed() # Ensure true randomness on each run
##    x_vec = np.round(np.random.rand(2) * 5, 2) # Random values up to 5.0
##    w_vec = np.round(np.random.rand(2) * 5, 2)

    x_vec = [2, 4] # Random values up to 5.0
    w_vec = [1, 3]
    
    print(f"Random Vector X: {x_vec}")
    print(f"Random Vector W: {w_vec}")
    print("-" * 50)
    
    # Calculate Classical Baseline
    classical_dot = np.dot(x_vec, w_vec)
    print(f"Classical Target (Perfect Math): {classical_dot:.4f}")
    
    # Calculate Quantum Physics Result
    print("\nExecuting Quantum Hardware Simulation...")
    quantum_dot, full_circuit = execute_quantum_dot_product(x_vec, w_vec, shots=50000)
    
    print(f"Quantum Result (Statistical):  {quantum_dot:.4f}")

