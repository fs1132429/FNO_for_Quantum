import numpy as np
import matplotlib.pyplot as plt
import torch
import cudaq
from typing import List
from functions.functions_pauli_strings import *


def generate_random_input_states_wavefunction(n, num_states):
    # Generate random complex amplitudes for each spin state
    real_part = torch.rand(num_states, 2**n) * 2 - 1  # Random real part between -1 and 1
    imag_part = torch.rand(num_states, 2**n) * 2 - 1  # Random imag part between -1 and 1
    amplitudes = real_part + 1j * imag_part
    # Calculate normalization factor
    norm = torch.sqrt(torch.sum(torch.square(real_part), dim=1) + torch.sum(torch.square(imag_part), dim=1))
    # Normalize the amplitudes 
    amplitudes /= norm.view(-1, 1)
    return amplitudes.to(torch.complex64)

def generate_wavefunctions_shared_sparsity(n, num_states, sparsity=0.1, epsilon=1e-4):
    dim = 2**n
    num_significant = max(1, int(sparsity * dim))

    # Select same sparsity mask for all wavefunctions
    indices = torch.randperm(dim)
    significant_idx = indices[:num_significant]
    near_zero_idx = indices[num_significant:]

    # Create random complex amplitudes
    real_significant = torch.rand(num_states, num_significant) * 2 - 1
    imag_significant = torch.rand(num_states, num_significant) * 2 - 1
    values_significant = real_significant + 1j * imag_significant

    real_small = torch.rand(num_states, dim - num_significant) * 2 - 1
    imag_small = torch.rand(num_states, dim - num_significant) * 2 - 1
    values_small = epsilon * (real_small + 1j * imag_small)

    # Fill into the full amplitude tensor
    amplitudes = torch.zeros((num_states, dim), dtype=torch.complex64)
    amplitudes[:, significant_idx] = values_significant
    amplitudes[:, near_zero_idx] = values_small

    # Normalize
    norms = torch.linalg.norm(amplitudes, dim=1, keepdim=True)
    amplitudes /= norms

    return amplitudes


# Extract term coefficients from the Hamiltonian
def termCoefficients(op: cudaq.SpinOperator) -> List[complex]:
    result = []
    op.for_each_term(lambda term: result.append(term.get_coefficient()))
    return result

# Extract term words from the Hamiltonian
def termWords(op: cudaq.SpinOperator) -> List[str]:
    result = []
    op.for_each_term(lambda term: result.append(term.to_string(False)))
    return result

@cudaq.kernel
def trotter(state: cudaq.State, coefficients: List[complex],
            words: List[cudaq.pauli_word], dt: float):
    q = cudaq.qvector(state)
    for i in range(len(coefficients)):
        exp_pauli(coefficients[i].real * dt, q, words[i]) 

def create_time_data_set_cudaq2(n_spins, num_states, hamiltonian, dt, n_steps):
    input_states = generate_random_input_states_wavefunction(n_spins, num_states)
    
    # Initialize the output tensor
    output_tensor = torch.zeros(
        num_states, 2**n_spins, n_steps + 1, dtype=torch.complex64
    )
    coefficients = termCoefficients(hamiltonian)
    words = termWords(hamiltonian)
    for state_idx, random_state in enumerate(input_states):
        # Convert the current random state to a CUDA Quantum State
        initial_state = cudaq.State.from_data(random_state.numpy())
        # Perform the simulation
        state = initial_state
        for i in range(n_steps):
            state = cudaq.get_state(trotter, state, coefficients, words, dt)
            sv = np.array(state)
            output_tensor[state_idx, :, i] = torch.tensor(sv, dtype=torch.complex64)
    return output_tensor

            
    
def create_time_data_set_cudaq(n, num_states, hamiltonian, time, steps):
    # Generate all random input states as a list of PyTorch tensors
    input_states = generate_random_input_states_wavefunction(n, num_states)

    # Initialize the output tensor
    output_tensor = torch.zeros(
        num_states, 2**n, steps + 1, dtype=torch.complex64
    )
    dimensions = {i: 2 for i in range(n)}
    # Process each state individually
    for state_idx, random_state in enumerate(input_states):
        # Convert the current random state to a CUDA Quantum State
        initial_state = cudaq.State.from_data(random_state.numpy())
        
        # Time steps for evolution
        time_steps = np.linspace(0, time, steps + 1)
        schedule = cudaq.Schedule(time_steps, ["time"])
        
        # Perform time evolution
        evolution_result = cudaq.evolve(
            hamiltonian=hamiltonian,
            dimensions=dimensions,
            schedule=schedule,
            initial_state=initial_state,
            store_intermediate_results=True,
        )

        # Extract and rearrange intermediate states
        for step_idx, state in enumerate(evolution_result.intermediate_states()):
            # Convert state vector to complex numpy array
            sv = np.array(state)
            # Assign to output tensor
            output_tensor[state_idx, :, step_idx] = torch.tensor(sv, dtype=torch.complex64)
    return output_tensor


#here added 1/2 as it is in our def of spin operators but not in cuda quantum's
def construct_hamiltonian_cudaq(
    n_particles, Jx=None, Jy=None, Jz=None, h=None, interactions=None):
    """
    Construct a generalized Hamiltonian using CUDA Quantum.

    Parameters:
    - n_particles: Number of particles (qubits) in the system.
    - Jx, Jy, Jz: Coupling strengths for X, Y, and Z interactions. Can be scalars or lists.
    - h: Local field strengths. Can be a scalar or a list.
    - interactions: Custom list of qubit pairs defining interactions.

    Returns:
    - cudaq.SpinOperator: The constructed Hamiltonian.
    """
    # Define default interactions if None
    if interactions is None:
        interactions = [(i, (i + 1) % n_particles) for i in range(n_particles)]
    
    # Ensure Jx, Jy, and Jz are numpy arrays of the correct length
    if Jx is None:
        Jx = -1 * np.ones(len(interactions), dtype=complex)
    elif isinstance(Jx, (int, float)):
        Jx = np.array([Jx] * len(interactions), dtype=complex)

    if Jy is None:
        Jy = -1 * np.ones(len(interactions), dtype=complex)
    elif isinstance(Jy, (int, float)):
        Jy = np.array([Jy] * len(interactions), dtype=complex)

    if Jz is None:
        Jz = -1 * np.ones(len(interactions), dtype=complex)
    elif isinstance(Jz, (int, float)):
        Jz = np.array([Jz] * len(interactions), dtype=complex)

    if h is None:
        h = np.zeros(n_particles, dtype=complex)
    elif isinstance(h, (int, float)):
        h = np.array([h] * n_particles, dtype=complex)

    # Initialize the Hamiltonian
    hamiltonian = 0
    # Add interaction terms
    for idx, (i, j) in enumerate(interactions):
        hamiltonian += Jx[idx] * (1/2)*cudaq.spin.x(i) * (1/2)*cudaq.spin.x(j)
        hamiltonian += Jy[idx] * (1/2)*cudaq.spin.y(i) * (1/2)*cudaq.spin.y(j)
        hamiltonian += Jz[idx] * (1/2)*cudaq.spin.z(i) * (1/2)*cudaq.spin.z(j)

    # Add local field terms
    for i in range(n_particles):
        hamiltonian += h[i] * (1/2)*cudaq.spin.z(i)

    return hamiltonian

def construct_hamiltonian_ising_cudaq(n_particles, Jz=None, hx=None, interactions=None):
    """
    Construct an Ising Hamiltonian with ZZ interactions and X field terms using CUDA Quantum.

    Parameters:
    - n_particles: Number of qubits in the system.
    - Jz: ZZ coupling strengths. Can be a scalar or list of length equal to number of interactions.
    - hx: Local transverse field strengths (along X). Can be a scalar or list of length n_particles.
    - interactions: List of qubit index pairs defining ZZ interactions.

    Returns:
    - cudaq.SpinOperator: The constructed Hamiltonian.
    """
    import numpy as np
    import cudaq

    # Define default interactions (nearest neighbor with periodic boundary)
    if interactions is None:
        interactions = [(i, (i + 1) % n_particles) for i in range(n_particles)]

    # Ensure Jz and hx are numpy arrays of the correct length
    if Jz is None:
        Jz = -1 * np.ones(len(interactions), dtype=complex)
    elif isinstance(Jz, (int, float, complex)):
        Jz = np.array([Jz] * len(interactions), dtype=complex)

    if hx is None:
        hx = np.zeros(n_particles, dtype=complex)
    elif isinstance(hx, (int, float, complex)):
        hx = np.array([hx] * n_particles, dtype=complex)

    # Initialize the Hamiltonian
    hamiltonian = 0

    # Add ZZ interaction terms
    for idx, (i, j) in enumerate(interactions):
        hamiltonian += Jz[idx] * (1/2) * cudaq.spin.z(i) * (1/2) * cudaq.spin.z(j)

    # Add local X field terms
    for i in range(n_particles):
        hamiltonian += hx[i] * (1/2) * cudaq.spin.x(i)

    return hamiltonian


@cudaq.kernel
def trotter(state: cudaq.State, coefficients: List[complex],
            words: List[cudaq.pauli_word], dt: float):
    q = cudaq.qvector(state)
    for i in range(len(coefficients)):
        #flip the sign as exp_pauli implements e^{iHt}
        exp_pauli(-coefficients[i].real * dt, q, words[i]) 
        
# Extract term coefficients from the Hamiltonian
def termCoefficients(op: cudaq.SpinOperator) -> List[complex]:
    result = []
    op.for_each_term(lambda term: result.append(term.get_coefficient()))
    return result

# Extract term words from the Hamiltonian
def termWords(op: cudaq.SpinOperator) -> List[str]:
    result = []
    op.for_each_term(lambda term: result.append(term.to_string(False)))
    return result

def create_trotter_time_data_set_cudaq(n, num_states, hamiltonian, dt, n_steps):
    # Generate all random input states as a list of PyTorch tensors
    input_states = generate_random_input_states_wavefunction(n, num_states)

    # Initialize the output tensor
    output_tensor = torch.zeros(
        num_states, 2**n, n_steps, dtype=torch.complex64
    )
    coefficients = termCoefficients(hamiltonian)
    words = termWords(hamiltonian)
    # Process each state individually
    for state_idx, random_state in enumerate(input_states):
        print(f"Running for state {state_idx}")
        # Convert the current random state to a CUDA Quantum State
        initial_state = cudaq.State.from_data(random_state.numpy())
        # Perform the simulation
        state = initial_state
        for i in range(n_steps):
            # Trotter step to evolve the state
            state = cudaq.get_state(trotter, state, coefficients, words, dt)
            sv = np.array(state)
            output_tensor[state_idx, :, i] = torch.tensor(sv, dtype=torch.complex64)
    return output_tensor


def create_trotter_discretized_time_data_set_cudaq(n, num_states, hamiltonian, dt, n_steps,step_interval):
    # Generate all random input states as a list of PyTorch tensors
    input_states = generate_random_input_states_wavefunction(n, num_states)
    
    selected_steps = n_steps // step_interval
    
    # Initialize the output tensor
    output_tensor = torch.zeros(
    num_states, 2**n, selected_steps, dtype=torch.complex64
    )
    coefficients = termCoefficients(hamiltonian)
    words = termWords(hamiltonian)
    
    # Process each state individually
    for state_idx, random_state in enumerate(input_states):
        # Convert to CUDA Quantum state
        initial_state = cudaq.State.from_data(random_state.numpy())
        
        # Perform simulation
        state = initial_state
        store_idx = 0  # Index to store in output_tensor
        for i in range(n_steps):
            # Apply Trotter step
            state = cudaq.get_state(trotter, state, coefficients, words, dt)

            # Store only if i is divisible by 100
            if i % step_interval == 0:
                sv = np.array(state)
                output_tensor[state_idx, :, store_idx] = torch.tensor(sv, dtype=torch.complex64)
                store_idx += 1  # Move to the next storage index
    return output_tensor


def create_trotter_discretized_time_data_set_with_expectations(n, num_states, hamiltonian, dt, n_steps, step_interval, pauli_strings, sparsity=0.1):
    """
    Generate Trotter-evolved states and compute expectation values for given Pauli strings.
    """

    selected_steps = n_steps // step_interval

    # Generate input states
    input_states = generate_wavefunctions_shared_sparsity(n, num_states,sparsity=sparsity)

    # Initialize output tensors
    output_tensor = torch.zeros((num_states, 2**n, selected_steps), dtype=torch.complex64)
    expectation_tensor = torch.zeros((num_states, len(pauli_strings), selected_steps), dtype=torch.float32)

    coefficients = termCoefficients(hamiltonian)
    words = termWords(hamiltonian)

    # Precompute Pauli operators
    pauli_ops = [generate_pauli_matrix(''.join(ps)) for ps in pauli_strings]

    for state_idx, random_state in enumerate(input_states):
        # Convert to CUDA Quantum state
        state = cudaq.State.from_data(random_state.numpy())
        store_idx = 0

        for i in range(n_steps):
            state = cudaq.get_state(trotter, state, coefficients, words, dt)

            if i % step_interval == 0:
                sv = np.array(state)
                psi = torch.tensor(sv, dtype=torch.complex64)
                output_tensor[state_idx, :, store_idx] = psi

                # Expectation values for each Pauli string
                for p_idx, op in enumerate(pauli_ops):
                    expectation_tensor[state_idx, p_idx, store_idx] = torch.vdot(psi, torch.matmul(op, psi)).real

                store_idx += 1

    return expectation_tensor

def create_trotter_discretized_time_data_set_with_expectations2(
    n, num_states, hamiltonian, dt, n_steps, step_interval, pauli_strings, sparsity=0.1
):
    """
    Generate Trotter-evolved states and compute expectation values for given Pauli strings.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_steps = n_steps // step_interval
    # Generate input states
    input_states = generate_wavefunctions_shared_sparsity(n, num_states, sparsity=sparsity)
    # Initialize output tensors
    expectation_tensor = torch.zeros((num_states, len(pauli_strings), selected_steps), dtype=torch.float32, device=device)
    coefficients = termCoefficients(hamiltonian)
    words = termWords(hamiltonian)
    # Precompute Pauli operators and stack them
    pauli_ops = [generate_pauli_matrix(''.join(ps)).to(torch.complex64).to(device) for ps in pauli_strings]
    pauli_stack = torch.stack(pauli_ops)  # Shape: (P, 2^n, 2^n) 
    #pauli_stack = torch.empty((len(pauli_strings), 2**n, 2**n), dtype=torch.complex64, device=device)
    #for i in range(pauli_stack.shape[0]):
        #for j,ps in enumerate(pauli_strings):
            #pauli_stack[i,j] = generate_pauli_matrix(''.join(ps))
    for state_idx, random_state in enumerate(input_states):
        # Convert initial state to CUDAQ state
        state = cudaq.State.from_data(random_state.numpy())
        store_idx = 0
        for i in range(n_steps):
            state = cudaq.get_state(trotter, state, coefficients, words, dt)
            if i % step_interval == 0:
                psi = torch.tensor(np.array(state), dtype=torch.complex64, device=device).view(-1)
                # Batched expectation values: (psi† @ A @ psi)
                expectation_tensor[state_idx, :, store_idx] = torch.einsum(
                    'i,bij,j->b', psi.conj(), pauli_stack, psi
                ).real
                store_idx += 1
    return expectation_tensor

def create_trotter_discretized_time_data_set_with_expectations2_(
    n, num_states, hamiltonian, dt, n_steps, step_interval, pauli_strings, sparsity=0.1
):
    """
    Generate Trotter-evolved states and compute expectation values for given Pauli strings.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_steps = n_steps // step_interval
    print("generating input")
    # Generate input states
    input_states = generate_wavefunctions_shared_sparsity(n, num_states, sparsity=sparsity)
    # Initialize output tensors
    print("input generated")
    expectation_tensor = torch.zeros((num_states, len(pauli_strings), selected_steps), dtype=torch.float32, device=device)
    coefficients = termCoefficients(hamiltonian)
    words = termWords(hamiltonian)
    # Precompute Pauli operators and stack them
    pauli_ops = [generate_pauli_matrix(''.join(ps)).to(torch.complex64).to(device) for ps in pauli_strings]
    pauli_stack = torch.stack(pauli_ops)  # Shape: (P, 2^n, 2^n) 
    print("pauli stack generated")
    for state_idx, random_state in enumerate(input_states):
        print("starting loop")
        # Convert initial state to CUDAQ state
        state = cudaq.State.from_data(random_state.numpy())
        print("got state")
        store_idx = 0
        for i in range(n_steps):
            print("starting second loop")
            state = cudaq.get_state(trotter, state, coefficients, words, dt)
            print("ran get_state")
            if i % step_interval == 0:
                print("getting psi")
                psi = torch.tensor(np.array(state), dtype=torch.complex64, device=device).view(-1)
                print("psi stored")
                # Batched expectation values: (psi† @ A @ psi)
                # Compute in batches to reduce memory pressure
                batch_size = 10  # You can adjust this depending on your GPU
                num_paulis = pauli_stack.shape[0]
                expectations = []

                for b_start in range(0, num_paulis, batch_size):
                    b_end = min(b_start + batch_size, num_paulis)
                    pauli_batch = pauli_stack[b_start:b_end]  # shape: (B, 2^n, 2^n)
                    with torch.no_grad():
                        exp_vals = torch.einsum('i,bij,j->b', psi.conj(), pauli_batch, psi).real
                    expectations.append(exp_vals)

                expectations = torch.cat(expectations)  # shape: (num_paulis,)
                expectation_tensor[state_idx, :, store_idx] = expectations
                print("exp val generated")
                store_idx += 1
        torch.cuda.empty_cache()
    return expectation_tensor

I = {(0, 0): 1, (1, 1): 1}
X = {(0, 1): 0.5, (1, 0): 0.5}
Y = {(0, 1): -1j*0.5, (1, 0): 1j*0.5}
Z = {(0, 0): 0.5, (1, 1): -0.5}

pauli_map = {'I': I, 'X': X, 'Y': Y, 'Z': Z}


def expectation_sparse_torch(sparse_matrix, vector):
    """
    Compute ⟨ψ|A|ψ⟩ where A is a sparse COO tensor and ψ is a normalized complex torch vector.
    Supports 1D vector input (shape: [N])
    """
    if vector.dim() != 1:
        raise ValueError(f"Expected vector to be 1D, but got shape {vector.shape}")

    # reshape for sparse mm
    vec_col = vector.unsqueeze(-1)  # (N, 1)
    result = torch.sparse.mm(sparse_matrix, vec_col).squeeze(-1)  # (N,)
    return torch.vdot(vector, result)




def kronecker_sparse_indices_vals(A_indices, A_values, dim_A, B_indices, B_values, dim_B):
    new_indices = []
    new_values = []

    for idx_A, val_A in zip(A_indices, A_values):
        i, j = idx_A
        for idx_B, val_B in zip(B_indices, B_values):
            k, l = idx_B
            row = i * dim_B + k
            col = j * dim_B + l
            new_indices.append((row, col))
            new_values.append(val_A * val_B)

    return new_indices, new_values

def pauli_string_to_sparse_tensor(pauli_str, device='cuda'):
    # Start with the first Pauli operator
    base = pauli_map[pauli_str[0]]
    indices = list(base.keys())
    values = list(base.values())
    dim = 2

    for p in pauli_str[1:]:
        next_op = pauli_map[p]
        next_indices = list(next_op.keys())
        next_values = list(next_op.values())

        indices, values = kronecker_sparse_indices_vals(
            indices, values, dim,
            next_indices, next_values, 2
        )
        dim *= 2

    # Convert to torch.sparse_coo_tensor
    if len(indices) == 0:
        coo_indices = torch.empty((2, 0), dtype=torch.long)
        coo_values = torch.empty((0,), dtype=torch.complex64)
    else:
        coo_indices = torch.tensor(indices, dtype=torch.long).T  # shape (2, nnz)
        coo_values = torch.tensor(values, dtype=torch.complex64)

    return torch.sparse_coo_tensor(coo_indices, coo_values, (dim, dim), device=device)


def compare_all_pauli_operators(N):
    device = torch.device('cuda')
    pauli_strings = generate_pauli_strings_ising_all(N)

    matches = 0
    total = len(pauli_strings)
    
    #import psutil

    #process = psutil.Process(os.getpid())
    #mem_before = process.memory_info().rss 
    t1=t.time()
    #pauli_ops=[]
    for ps in pauli_strings:
        sparse = pauli_string_to_sparse_tensor(ps)

        dense_sparse= sparse.to_dense()
        #t3=t.time()
        dense_torch = generate_pauli_matrix(ps).to(torch.complex64).to(device).cpu().numpy()
        #t4=t.time()
        #print(f"Time taken to generate pauli matrix(full matrix): {t4-t3} seconds")
    #mem_after = process.memory_info().rss
    #print(f"Memory used: {(mem_after - mem_before) / (1024**2):.2f} MB")
    #t2=t.time()
    #print(f"Time taken to generate sparse matrix: {t2-t1} seconds")
        if np.allclose(dense_sparse, dense_torch):
            matches += 1
        else:
            print(f"❌ Mismatch: {ps}")
            print("Sparse:")
            print(dense_sparse)
            print("Torch:")
            print(dense_torch)
    print(f"\n Matched {matches}/{total} Pauli matrices (element-wise)")
    
#compare_all_pauli_operators(N=8)


def check_expectation_match( N, device="cuda", atol=1e-5):
    """
    Check that the expectation value computed using a sparse Pauli operator
    matches the one from a dense operator, for a random normalized complex statevector.

    Args:
        ps (str): Pauli string like "XIZY"
        sparse (torch.sparse_coo_tensor): Sparse Pauli operator
        N (int): Dimension of the statevector (should be 2^n)
        device (str or torch.device): Target device
        atol (float): Tolerance for matching

    Returns:
        matched (bool): True if expectations match within tolerance
    """
    # Generate random normalized complex statevector
    pauli_strings = generate_pauli_strings_ising_all(N)
    match=0
    total=len(pauli_strings)
    for ps in pauli_strings:
        real = torch.randn(2**N, device=device)
        imag = torch.randn(2**N, device=device)
        vec = real + 1j * imag
        vec = vec / torch.linalg.norm(vec)
        sparse = pauli_string_to_sparse_tensor(ps)
        # Compute expectation values
        exp_sparse = expectation_sparse_torch(sparse, vec)
        dense_torch_op = generate_pauli_matrix(ps).to(torch.complex64).to(device)
        exp_dense = torch.vdot(vec, dense_torch_op @ vec)

        # Compare
        if not torch.allclose(exp_sparse, exp_dense, atol=atol):
            print(f"❌ Expectation mismatch for {ps}")
            print(f"   Sparse: {exp_sparse}")
            print(f"   Dense:  {exp_dense}")
            return False
        else:
            match+=1
    print(f"\n Matched {match}/{total} Pauli matrices (element-wise)")
    
#check_expectation_match(N=10)
    
    

def create_trotter_discretized_time_data_set_with_expectations_sparse(
    n, num_states, hamiltonian, dt, n_steps, step_interval, pauli_strings, sparsity=0.1
):
    """
    Generate Trotter-evolved states and compute expectation values using sparse Pauli operators.
    """
    import gc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_steps = n_steps // step_interval

    # Generate initial statevectors with shared sparsity
    input_states = generate_wavefunctions_shared_sparsity(n, num_states, sparsity=sparsity)

    # Allocate expectation output tensor
    expectation_tensor = torch.zeros(
        (num_states, len(pauli_strings), selected_steps), dtype=torch.float32, device=device
    )
    
    #expectation_tensor2 = torch.zeros(
        #(num_states, len(pauli_strings), selected_steps), dtype=torch.float32, device=device
    #)

    # Prepare Hamiltonian terms
    coefficients = termCoefficients(hamiltonian)
    words = termWords(hamiltonian)

    # Precompute all sparse Pauli operators as sparse COO tensors
    pauli_ops_sparse = [pauli_string_to_sparse_tensor(''.join(ps)).to(device) for ps in pauli_strings]
    
    #pauli_ops = [generate_pauli_matrix(''.join(ps)).to(torch.complex64).to(device) for ps in pauli_strings]
    #pauli_stack = torch.stack(pauli_ops)

    for state_idx, random_state in enumerate(input_states):
        # Convert NumPy vector to CUDAq state
        state = cudaq.State.from_data(random_state.numpy())
        store_idx = 0

        for i in range(n_steps):
            state = cudaq.get_state(trotter, state, coefficients, words, dt)

            if i % step_interval == 0:
                psi = torch.tensor(np.array(state), dtype=torch.complex64, device=device).view(-1)
                # Store result
                expectation_tensor[state_idx, :, store_idx] = torch.stack([
                    expectation_sparse_torch(sparse_op, psi).real
                    for sparse_op in pauli_ops_sparse
                ])
                
                #expectation_tensor2[state_idx, :, store_idx] = torch.einsum(
                    #'i,bij,j->b', psi.conj(), pauli_stack, psi
                #).real
                
                store_idx += 1

        del state, psi
        torch.cuda.empty_cache()
        gc.collect()
    
    #if not torch.allclose(expectation_tensor, expectation_tensor2, atol=1e-3):
        #print(f" Expectation mismatch ")
    #else:
        #print(f"Match ")
    return expectation_tensor

