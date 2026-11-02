# Quantum Circuit Visualizer

A Streamlit learning app for building editable Qiskit circuits, exploring their statevectors, and comparing ideal predictions with finite-shot or noisy simulation.

The workspace provides live Matplotlib circuit diagrams; the inspector includes a Plotly Bloch-sphere view and reduced-state metrics.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints local and network URLs in the terminal after startup. Use the local URL on the same computer; the network URL is for devices that can reach the computer over the same network.

## Test

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Stress tests

The optional stress suite exercises maximum-size simulations, randomized
validation and UI actions, statistical sampling, repeated resource use, and
the largest supported Streamlit configuration. It is skipped during normal
test discovery. Run it explicitly with:

```powershell
$env:RUN_STRESS="1"; .\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_stress.py" -v
```

## Structure

- `app.py`: Streamlit controls, session state, charts, circuit preview, and state inspection.
- `quantum_engine.py`: Qiskit circuit validation/construction, exact and noisy simulations, shot experiments, exports, and quantum-state metrics.
- `tests/`: Engine and Streamlit regression tests.
- `requirements.txt`: Top-level runtime dependencies.
- `requirements-lock.txt`: Resolved package versions for reproducible local installs.
- `pyproject.toml`: Project metadata and Ruff lint/format configuration.
- `plan.md`: Architecture and implementation contracts.

The engine uses Qiskit's little-endian bitstring convention (`q[n-1] ... q1 q0`, q0 at the right), limits exact simulation to 10 qubits, and caps circuits at 50 gates and experiments at 100,000 shots. Exact noisy probability simulation is limited to 8 qubits; noisy shot sampling remains available above that limit. The optional noise model affects supported gate operations and/or measurement; it does not model idle-time noise.

## Code quality and CI

Install development tools with `pip install -r requirements-dev.txt`. Run Ruff locally with `ruff check .` and `ruff format .`. GitHub Actions runs Ruff and the unittest suite on pushes and pull requests using Python 3.12.

**Python version:** Python 3.12 is recommended and matches CI. On one Windows
machine, Python 3.13.9 with Qiskit 2.6.1 repeatedly crashed with an access
violation inside Qiskit's native extension (`_accelerate.pyd`), while the same
package versions on Python 3.12.16 ran stably. The cause was not identified.
