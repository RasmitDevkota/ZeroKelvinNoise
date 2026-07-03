from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

def generate_ising_benchmark(n_qubits, trotter_steps):
    qc = QuantumCircuit(n_qubits)
    
    for i in range(n_qubits):
        qc.h(i)
    
    for _ in range(trotter_steps):
        for i in range(n_qubits - 1):
            qc.cx(i, i+1)
            qc.rz(0.5, i+1)
            qc.cx(i, i+1)
        for i in range(n_qubits):
            qc.rx(0.2, i)
            
    observable = SparsePauliOp.from_list([("Z" * n_qubits, 1.0)])
    
    return qc, observable
