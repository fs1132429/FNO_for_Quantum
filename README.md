# FNO_for_Quantum

This repository contains code for reproducing the results from the paper: [https://arxiv.org/abs/2509.07084](https://arxiv.org/abs/2509.07084)

Fourier Neural Operators (FNOs) are applied to simulate the time evolution of quantum wavefunctions and Hamiltonian observables for quantum spin systems. This codebase includes data generation, model training, evaluation, and example workflows.

---

## Folder Structure

- **`functions/`**  
  Contains the main functions for data generation and FNO architectures.  
  - Supports different input types: **wavefunctions** and **Hamiltonian observables**.  
  - Handles different input states: **random** or **low-energy** wavefunctions.

- **`hamiltonian_observable_examples/`**  
  Example workflow demonstrating how a trained FNO for a **20-qubit Hamiltonian observable** is tested on new data and used to extrapolate future time steps.

- **`scripts/`**  
  Ready-to-use scripts for **data generation** and **model training**. These scripts were used to generate datasets and train the FNO models in the project.

- **`tests/`**  
  Contains test files to verify the functions in the `functions/` folder.

- **`wavefunction_input_examples/`**  
  Example files for training the FNO on **4- and 8-qubit wavefunction inputs**.  
  Includes both **random** and **low-energy** input cases.

---

## Installation

Clone the repository and install the required Python packages:

```bash
git clone https://github.com/your-username/FNO_for_Quantum.git
cd FNO_for_Quantum
pip install -r requirements.txt


