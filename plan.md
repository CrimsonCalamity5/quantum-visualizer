# Quantum Circuit Visualizer Architecture

## Module responsibilities

- `quantum_engine.py` owns gate and circuit validation, Qiskit circuit construction, ideal and noisy simulation, shot experiments, circuit serialization, algorithm builders, and state metrics. It does not import Streamlit or render UI.
- `app.py` owns Streamlit controls and session state, builds engine-compatible gate sequences, invokes engine operations, and renders circuit diagrams, probability comparisons, state inspection, and parameter tools.
- `tests/` exercises the engine independently and the Streamlit interface through `AppTest`.

## Data contracts

- An ordered gate has the form `{"gate": "cx", "qubits": [0, 1]}`. Single-qubit operations take one zero-based qubit index. Rotation and phase gates also take a numeric angle or parameter name.
- Qubit outcomes use Qiskit's little-endian convention: a fixed-width string `q[n-1] ... q1 q0`, with q0 at the right. UI charts and tables keep these outcomes as text.
- Multi-qubit gate operands must be in range and distinct. Empty gate lists are valid identity circuits.
- The engine limits statevector simulation to `MAX_QUBITS = 10`, shot experiments to `MAX_SHOTS = 100_000`, and circuits to `MAX_GATES = 50`.
- Exact noisy probability simulation is limited to `MAX_NOISY_EXACT_QUBITS = 8`; noisy shot simulation remains available above that limit.

## Implemented engine capabilities

- Gates: H, X, Y, Z, S, T, P, RX, RY, RZ, CX, CZ, controlled phase, SWAP, and CCX.
- Ideal statevector amplitudes/probabilities, prefix circuits, per-qubit reduced-state Bloch vectors, purity and entropy.
- Aer shot counts with reproducible seeds, standard-error calculations, and optional gate/readout noise. Pauli-type gate-noise channels are merged analytically; errors are applied after supported gates, and idle-time noise is not modeled.
- Exact noisy measurement distributions using density-matrix evolution, including separately applied symmetric readout-error probabilities.
- Named-parameter binding and multi-parameter sweeps (`run_parameter_sweep(circuit, parameter_name, values, fixed_values=None)`), a teaching VQE example, cached QFT and Grover sequence builders, validated JSON import/export, and OpenQASM 3 export.

## Implemented UI capabilities

- Editable presets for Bell, GHZ, coin toss, uniform superposition, QFT, Grover, balanced and constant Deutsch–Jozsa, and Bernstein–Vazirani examples.
- Circuit preview, clear-all, JSON import/export, QASM download, gate-prefix step-through, and bounded qubit/shot controls.
- Separate workspace, simulation comparison, state inspector, and variational tabs.
- Grouped Altair bars for ideal/noisy exact probabilities and shots; labels remain nominal binary strings, finite-shot standard errors are shown, and a shared top-outcome limit filters both the chart and state table.
- Complex-amplitude and phase inspection, Plotly Bloch vectors, reduced-state metrics, parameter sweeps, and the toy VQE panel.

## Validation and extension notes

- Install the direct dependencies with `pip install -r requirements.txt`; `requirements-lock.txt` records the resolved environment.
- Run `python -m unittest discover -s tests -v` and `streamlit run app.py` from the project directory.
- Run `ruff check .` and `ruff format .`; GitHub Actions runs Ruff and the tests on Python 3.12 for pushes and pull requests.
- Keep Streamlit-specific logic in `app.py`; keep simulation contracts and validation in `quantum_engine.py`.
- Potential future extensions include per-gate editing/reordering, additional noise channels and calibrated device profiles, QASM import, and larger algorithm examples. Exact statevector limits should remain explicit if increased because memory and computation scale exponentially with qubit count.
