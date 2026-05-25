import numpy as np
import cirq
import matplotlib.pyplot as plt

# ============================================================
# Reproducibility
# ============================================================
np.random.seed(42)

# ============================================================
# 1. HARDWARE GATE DECOMPOSITIONS (From your code!)
# ============================================================
def manual_toffoli(c1, c2, target):
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
    yield cirq.CNOT(target_a, target_b)
    yield from manual_toffoli(ancilla, target_b, target_a)
    yield cirq.CNOT(target_a, target_b)

def manual_controlled_ry(theta, control, target):
    yield cirq.ry(theta / 2.0)(target)
    yield cirq.CNOT(control, target)
    yield cirq.ry(-theta / 2.0)(target)
    yield cirq.CNOT(control, target)

# ============================================================
# 2. DATA PREPARATION (4x1 VECTOR ALGEBRA)
# ============================================================
def calculate_angles_4d(vector):
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
    yield cirq.ry(2 * alpha)(q0)
    yield cirq.X(q0)
    yield from manual_controlled_ry(2 * beta, q0, q1)
    yield cirq.X(q0)
    yield from manual_controlled_ry(2 * gamma, q0, q1)

# ============================================================
# 3. DOT PRODUCT FUNCTIONS
# ============================================================
# ============================================================
# 3A. ADAPTIVE SHOT ESTIMATION (Added - minimal integration)
# ============================================================
_LAST_ADAPTIVE = {'shots_used': 0, 'history': None}
_LAST_LMS_SHOTS_HIST = None

def get_quantum_overlap_adaptive(circuit, ancilla, max_shots=20000, batch=200, eps_p=0.005, z=1.96, min_shots=400):
    """Adaptive-shot estimator for the SWAP-test ancilla probability P(0).

    Stops when the (normal-approx) 95% CI half-width for P(0) is <= eps_p.
    Returns (overlap, shots_used, history_dict).
    This does NOT require knowing the true dot product (real-hardware valid).
    """
    sim = cirq.Simulator()
    shots_used = 0
    zeros = 0
    hist_shots = []
    hist_p = []
    hist_overlap = []
    hist_ci = []

    # Stream shots in batches and update confidence interval.
    while shots_used < max_shots:
        cur = min(batch, max_shots - shots_used)
        result = sim.run(circuit, repetitions=cur)
        m = result.measurements['result'].reshape(-1)
        zeros += int(np.sum(m == 0))
        shots_used += cur

        p_hat = zeros / shots_used
        # Standard error of Bernoulli proportion (normal approximation)
        se = np.sqrt(max(1e-12, p_hat * (1.0 - p_hat)) / shots_used)
        ci_half = z * se
        overlap_hat = np.sqrt(max(0.0, 2.0 * p_hat - 1.0))

        hist_shots.append(shots_used)
        hist_p.append(p_hat)
        hist_overlap.append(overlap_hat)
        hist_ci.append(ci_half)

        if shots_used >= min_shots and ci_half <= eps_p:
            break

    history = {'shots': np.array(hist_shots), 'p0': np.array(hist_p), 'overlap': np.array(hist_overlap), 'ci_half_p0': np.array(hist_ci)}
    return float(hist_overlap[-1]), int(shots_used), history

def get_quantum_overlap(circuit, ancilla, shots):
    """Executes the circuit and extracts the positive overlap magnitude.

    Backward compatible: if 'shots' is an int, behaves exactly like before.
    If 'shots' is a dict with shots['adaptive']=True, uses adaptive batches.
    """
    # --- Adaptive mode (minimal change: accept a dict config) ---
    if isinstance(shots, dict) and shots.get('adaptive', False):
        overlap, used, history = get_quantum_overlap_adaptive(
            circuit, ancilla,
            max_shots=shots.get('max_shots', 20000),
            batch=shots.get('batch', 200),
            eps_p=shots.get('eps_p', 0.005),
            z=shots.get('z', 1.96),
            min_shots=shots.get('min_shots', 400),
        )
        _LAST_ADAPTIVE['shots_used'] = used
        _LAST_ADAPTIVE['history'] = history
        return overlap

    # --- Fixed-shot mode (original behavior) ---
    sim = cirq.Simulator()
    result = sim.run(circuit, repetitions=shots)
    prob_0 = np.sum(result.measurements['result'] == 0) / shots
    overlap = np.sqrt(max(0, 2 * prob_0 - 1))
    _LAST_ADAPTIVE['shots_used'] = int(shots)
    _LAST_ADAPTIVE['history'] = None
    return overlap

def inbuilt_cswap_dot(vec_x, vec_w, shots=1000):
    """Dot product using Cirq's built-in CSWAP gate."""
    a_x, b_x, g_x, n_x = calculate_angles_4d(vec_x)
    a_w, b_w, g_w, n_w = calculate_angles_4d(vec_w)
    
    if n_x == 0 or n_w == 0: return 0.0
    
    ancilla = cirq.NamedQubit('C')
    qx = [cirq.NamedQubit(f'X{i}') for i in range(2)]
    qw = [cirq.NamedQubit(f'W{i}') for i in range(2)]
    
    circuit = cirq.Circuit()
    circuit.append(hardware_state_prep_4d(qx[0], qx[1], a_x, b_x, g_x))
    circuit.append(hardware_state_prep_4d(qw[0], qw[1], a_w, b_w, g_w))
    
    circuit.append(cirq.H(ancilla))
    circuit.append(cirq.CSWAP(ancilla, qx[0], qw[0]))
    circuit.append(cirq.CSWAP(ancilla, qx[1], qw[1]))
    circuit.append(cirq.H(ancilla))
    circuit.append(cirq.measure(ancilla, key='result'))
    
    overlap = get_quantum_overlap(circuit, ancilla, shots)
    
    # Sign Correction for LMS Stability
    sign = np.sign(np.dot(vec_x, vec_w))
    sign = 1 if sign == 0 else sign 
    
    return overlap * n_x * n_w * sign

def custom_cswap_dot(vec_x, vec_w, shots=1000):
    """Dot product using your manual 15-gate Toffoli hardware CSWAP."""
    a_x, b_x, g_x, n_x = calculate_angles_4d(vec_x)
    a_w, b_w, g_w, n_w = calculate_angles_4d(vec_w)
    
    if n_x == 0 or n_w == 0: return 0.0
    
    ancilla = cirq.NamedQubit('C')
    qx = [cirq.NamedQubit(f'X{i}') for i in range(2)]
    qw = [cirq.NamedQubit(f'W{i}') for i in range(2)]
    
    circuit = cirq.Circuit()
    circuit.append(hardware_state_prep_4d(qx[0], qx[1], a_x, b_x, g_x))
    circuit.append(hardware_state_prep_4d(qw[0], qw[1], a_w, b_w, g_w))
    
    circuit.append(cirq.H(ancilla))
    circuit.append(hardware_cswap(ancilla, qx[0], qw[0]))
    circuit.append(hardware_cswap(ancilla, qx[1], qw[1]))
    circuit.append(cirq.H(ancilla))
    circuit.append(cirq.measure(ancilla, key='result'))
    
    overlap = get_quantum_overlap(circuit, ancilla, shots)
    
    # Sign Correction for LMS Stability
    sign = np.sign(np.dot(vec_x, vec_w))
    sign = 1 if sign == 0 else sign 
    
    return overlap * n_x * n_w * sign

# ============================================================
# 4. ADAPTIVE LMS FILTER INTEGRATION
# ============================================================
def get_x_vec(x, n, L):
    idx = np.arange(n, n - L, -1)
    return x[idx].astype(float)

def hybrid_lms_filter(x, d, L=4, mu=0.05, mode='classical', shots=1000):
    N = len(x)
    w = np.zeros(L, dtype=float)
    y = np.zeros(N, dtype=float)
    e = np.zeros(N, dtype=float)
    w_hist = np.zeros((N, L), dtype=float)

    # Track how many shots were actually used (only meaningful in adaptive mode)
    track_shots = isinstance(shots, dict) and shots.get('adaptive', False)
    shots_hist = np.zeros(N, dtype=int) if track_shots else None


    for n in range(L - 1, N):
        xv = get_x_vec(x, n, L)

        # Calculate FIR Filter Output using selected Dot Product
        if mode == 'classical':
            y[n] = np.dot(w, xv)
        elif mode == 'inbuilt':
            y[n] = inbuilt_cswap_dot(w, xv, shots=shots)
            if track_shots:
                shots_hist[n] = _LAST_ADAPTIVE.get('shots_used', 0)
        elif mode == 'custom':
            y[n] = custom_cswap_dot(w, xv, shots=shots)
            if track_shots:
                shots_hist[n] = _LAST_ADAPTIVE.get('shots_used', 0)

        # Standard LMS Update Rule
        e[n] = d[n] - y[n]
        w = w + mu * e[n] * xv
        w_hist[n] = w


    global _LAST_LMS_SHOTS_HIST
    _LAST_LMS_SHOTS_HIST = shots_hist

    return w, y, e, w_hist

# ============================================================
# 5. EXECUTION & GRAPHING SCRIPT
# ============================================================
if __name__ == "__main__":
    
    # -- A. SHOT CONVERGENCE GRAPH --
    print("Generating Shot Convergence Graph...")
    x_test = np.array([0.4, 1.2, 4.6, 2.4])
    w_test = np.array([4.8, 4.0, 1, 3.2])
    true_dot = np.dot(x_test, w_test)
    
    shots_list = [100, 500, 1000, 2500, 5000, 10000, 20000]
    inbuilt_res = []
    custom_res = []
    
    for s in shots_list:
        inbuilt_res.append(inbuilt_cswap_dot(x_test, w_test, shots=s))
        custom_res.append(custom_cswap_dot(x_test, w_test, shots=s))
        
    plt.figure(figsize=(10, 5))
    plt.axhline(true_dot, color='black', linestyle='--', label=f'Classical True Dot ({true_dot:.4f})')
    plt.plot(shots_list, inbuilt_res, marker='o', label='Inbuilt CSWAP')
    plt.plot(shots_list, custom_res, marker='x', label='Custom Hardware CSWAP')
    plt.title("Dot Product Convergence vs Number of Shots")
    plt.xlabel("Shots")
    plt.ylabel("Calculated Dot Product")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.show()


    # -- A2. ADAPTIVE SHOT DEMO GRAPH (Added) --
    print("Generating Adaptive-Shot Convergence Graph (batch=200)...")
    adaptive_cfg = {'adaptive': True, 'batch': 200, 'max_shots': 20000, 'min_shots': 400, 'eps_p': 0.01, 'z': 1.96}
    # Run one adaptive estimate using the custom circuit (you can switch to inbuilt as needed)
    adaptive_dot = custom_cswap_dot(x_test, w_test, shots=adaptive_cfg)
    hist = _LAST_ADAPTIVE.get('history', None)
    if hist is not None:
        # Scale overlap history to dot-product history using the same normalization used inside custom_cswap_dot
        _, _, _, n_x = calculate_angles_4d(x_test)
        _, _, _, n_w = calculate_angles_4d(w_test)
        sign = np.sign(np.dot(x_test, w_test)); sign = 1 if sign == 0 else sign
        dot_hist = hist['overlap'] * n_x * n_w * sign
        plt.figure(figsize=(10, 5))
        plt.axhline(true_dot, color='black', linestyle='--', label=f'Classical True Dot ({true_dot:.4f})')
        plt.plot(hist['shots'], dot_hist, marker='o', label='Adaptive (Custom CSWAP)')
        plt.title('Adaptive Shot Dot Product Estimate (Stops when CI is small)')
        plt.xlabel('Cumulative shots')
        plt.ylabel('Calculated Dot Product')
        plt.grid(True, linestyle=':')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # -- B. LMS FILTER SETUP --
    print("\nRunning LMS Filters... (This may take a minute due to Quantum Simulations)")
    
    # Using N=80 to keep execution time reasonable for standard computers. 
    # Change to 1500 if you want to run the full overnight simulation!
    N = 80 
    L = 4
    noise_std = 0.05
    mu = 0.05
    lms_shots = 1000 

    x = np.random.randn(N)
    w_true = np.array([0.8, -0.4, 0.25, 0.1], dtype=float)
    d = np.zeros(N, dtype=float)
    
    for n in range(L - 1, N):
        xv = get_x_vec(x, n, L)
        d[n] = np.dot(w_true, xv) + noise_std * np.random.randn()

    # Execute all 3 methods
    print("1/3 Running Classical LMS...")
    w_cls, y_cls, e_cls, w_hist_cls = hybrid_lms_filter(x, d, L, mu, 'classical')
    
    print("2/3 Running Inbuilt CSWAP LMS...")
    w_inb, y_inb, e_inb, w_hist_inb = hybrid_lms_filter(x, d, L, mu, 'inbuilt', lms_shots)
    
    print("3/3 Running Custom Hardware CSWAP LMS...")
    w_cus, y_cus, e_cus, w_hist_cus = hybrid_lms_filter(x, d, L, mu, 'custom', lms_shots)

    # Optional: Adaptive-shot LMS run (Added).
    # Note: Uses a confidence-interval stopping rule on ancilla P(0), does NOT need ground truth.
    # Optional: Adaptive-shot LMS runs at different confidence (tolerance) levels (Added).
    # z values correspond approximately to two-sided normal CI levels: 1.64→90%, 1.96→95%, 2.58→99%.
    print("4/4 Running Custom Hardware CSWAP LMS (Adaptive shots: 90%, 95%, 99%)...")
    lms_adaptive_specs = [(1.64, '90%'), (1.96, '95%'), (2.58, '99%')]
    shots_hist_ad = {}  # tolerance_label -> per-iteration shots used
    # (Keep the last adaptive run outputs in w_cus_ad/y_cus_ad/e_cus_ad/w_hist_cus_ad)
    for z, tol in lms_adaptive_specs:
        lms_adaptive_cfg = {'adaptive': True, 'batch': 200, 'max_shots': 20000, 'min_shots': 400, 'eps_p': 0.01, 'z': z}
        print(f"   -> Adaptive run for {tol} tolerance (z={z})")
        w_cus_ad, y_cus_ad, e_cus_ad, w_hist_cus_ad = hybrid_lms_filter(x, d, L, mu, 'custom', lms_adaptive_cfg)
        # _LAST_LMS_SHOTS_HIST is updated by hybrid_lms_filter; copy to freeze this run's history
        shots_hist_ad[tol] = None if _LAST_LMS_SHOTS_HIST is None else _LAST_LMS_SHOTS_HIST.copy()


    # -- C. GRAPHING RESULTS --
    
    # 1. Side-by-side FIR outputs
    n_show = N - L + 1
    start = L - 1
    end = start + n_show
    plt.figure(figsize=(12, 5))
    plt.plot(np.arange(start, end), d[start:end], label="Desired output d[n]", linewidth=2)
    plt.plot(np.arange(start, end), y_cls[start:end], label="Classical FIR", linewidth=2)
    plt.plot(np.arange(start, end), y_inb[start:end], label="Inbuilt CSWAP FIR", linestyle="--")
    plt.plot(np.arange(start, end), y_cus[start:end], label="Custom CSWAP FIR", linestyle=":")
    plt.title("Desired vs FIR Outputs (Side by Side)")
    plt.xlabel("Sample index")
    plt.ylabel("Output")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2. Learning Curves
    plt.figure(figsize=(12, 5))
    plt.plot(e_cls**2, label="Classical LMS Error^2")
    plt.plot(e_inb**2, label="Inbuilt CSWAP LMS Error^2")
    plt.plot(e_cus**2, label="Custom CSWAP LMS Error^2")
    plt.title("Learning Curves")
    plt.xlabel("Sample index")
    plt.ylabel("Squared error")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2b. Adaptive shot usage per iteration (Added)
    # Plots three curves (z = 1.64, 1.96, 2.58) for 90%, 95%, 99% tolerance levels.
    if isinstance(shots_hist_ad, dict) and len(shots_hist_ad) > 0:
        plt.figure(figsize=(12, 4))
        for z, tol in lms_adaptive_specs:
            hist = shots_hist_ad.get(tol, None)
            if hist is None:
                continue
            plt.plot(hist, label=f'Shots used ({tol}, z={z})')
        plt.title('Adaptive Shots per LMS Iteration (Custom CSWAP)')
        plt.xlabel('Sample index')
        plt.ylabel('Shots used')
        plt.grid(True, linestyle=':')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # 3. Coefficient Convergence
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    for i in range(L):
        axs[0].plot(w_hist_cls[:, i], label=f"w{i}")
        axs[0].axhline(w_true[i], linestyle=":", alpha=0.8)
    axs[0].set_title("Classical LMS Coefficient Convergence")
    axs[0].grid(True, linestyle=":")
    axs[0].legend(loc="upper right")

    for i in range(L):
        axs[1].plot(w_hist_inb[:, i], label=f"w{i}")
        axs[1].axhline(w_true[i], linestyle=":", alpha=0.8)
    axs[1].set_title("Inbuilt CSWAP Coefficient Convergence")
    axs[1].grid(True, linestyle=":")

    for i in range(L):
        axs[2].plot(w_hist_cus[:, i], label=f"w{i}")
        axs[2].axhline(w_true[i], linestyle=":", alpha=0.8)
    axs[2].set_title("Custom CSWAP Coefficient Convergence")
    axs[2].grid(True, linestyle=":")
    plt.show()

    # 4. Final Coefficient Bar Chart
    tap_idx = np.arange(L)
    bar_w = 0.2
    plt.figure(figsize=(10, 5))
    plt.bar(tap_idx - 1.5*bar_w, w_true, width=bar_w, label="True")
    plt.bar(tap_idx - 0.5*bar_w, w_cls, width=bar_w, label="Classical")
    plt.bar(tap_idx + 0.5*bar_w, w_inb, width=bar_w, label="Inbuilt CSWAP")
    plt.bar(tap_idx + 1.5*bar_w, w_cus, width=bar_w, label="Custom CSWAP")
    plt.xticks(tap_idx, [f"Tap {i}" for i in range(L)])
    plt.ylabel("Coefficient value")
    plt.title("Final FIR Coefficients")
    plt.grid(True, axis="y", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.show()
