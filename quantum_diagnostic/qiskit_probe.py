import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantum_engine import (
    build_circuit,
    build_noise_model,
    get_noisy_probabilities,
    get_statevector_data,
    run_shot_experiment,
)

st.title("Quantum Simulation Test")

try:
    circuit = build_circuit(2, [{"gate": "h", "qubits": [0]}, {"gate": "cx", "qubits": [0, 1]}])
    st.success("1. Circuit construction passed")

    amplitudes, probabilities = get_statevector_data(circuit)
    st.success("2. Statevector simulation passed")
    st.write(probabilities)

    counts = run_shot_experiment(circuit, shots=100)
    st.success("3. Shot simulation passed")
    st.write(counts)

    noise = build_noise_model(bit_flip_probability=0.01)
    st.success("4. Noise model creation passed")

    noisy = get_noisy_probabilities(circuit, noise)
    st.success("5. Noisy simulation passed")
    st.write(noisy)

except Exception as error:
    st.error(f"Test failed: {error}")
