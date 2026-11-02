"""Quantum circuit construction and simulation helpers.

Gate specifications use ``{"gate": ..., "qubits": [...]}``. Single-qubit
gates take one qubit index, while ``cx`` takes ``[control, target]``. Rotation
gates also require an ``angle`` in radians.

Every returned outcome bitstring follows Qiskit's native little-endian
convention: for n qubits, the string is ``q[n-1] ... q1 q0``, with q0 on the
right.
"""

from __future__ import annotations

import json
import math
import operator
import re
from collections.abc import Mapping, Sequence
from functools import lru_cache
from numbers import Real
from typing import Any, TypeAlias

import numpy as np
from qiskit import QuantumCircuit, qasm3
from qiskit.circuit import Parameter
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import DensityMatrix, Statevector, entropy, partial_trace
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    amplitude_damping_error,
    pauli_error,
)

MAX_QUBITS = 10
MAX_SHOTS = 100_000
MAX_GATES = 50
MAX_NOISY_EXACT_QUBITS = 8

GateSpecification: TypeAlias = Mapping[str, object]
_ANGLE_GATES = frozenset({"rx", "ry", "rz", "p", "cp"})
_SUPPORTED_GATES = frozenset(
    {"h", "x", "y", "z", "s", "t", "swap", "cx", "cz", "ccx", *_ANGLE_GATES}
)
_GATE_ARITY = {
    "h": 1,
    "x": 1,
    "y": 1,
    "z": 1,
    "s": 1,
    "t": 1,
    "cx": 2,
    "cz": 2,
    "swap": 2,
    "ccx": 3,
    "p": 1,
    "cp": 2,
    "rx": 1,
    "ry": 1,
    "rz": 1,
}
_NOISY_SINGLE_QUBIT_GATES = ("h", "x", "y", "z", "s", "t", "p", "rx", "ry", "rz")
_MAX_NOISE_PROBABILITY = 0.2


def _validate_qubit_count(qubit_count: int) -> None:
    if isinstance(qubit_count, bool):
        raise ValueError("qubit_count must be an integer.")
    try:
        qubit_count = operator.index(qubit_count)
    except TypeError as error:
        raise ValueError("qubit_count must be an integer.") from error
    if qubit_count < 1:
        raise ValueError("qubit_count must be at least 1.")
    if qubit_count > MAX_QUBITS:
        raise ValueError(f"A circuit may have at most {MAX_QUBITS} qubits; received {qubit_count}.")


def _convolve_pauli_probabilities(first: Sequence[float], second: Sequence[float]) -> list[float]:
    result = [0.0] * len(first)
    for first_index, first_probability in enumerate(first):
        for second_index, second_probability in enumerate(second):
            result[first_index ^ second_index] += first_probability * second_probability
    return result


def _pauli_error_from_probabilities(probabilities: Sequence[float]):
    pauli_labels = "IXZY"
    qubit_count = (len(probabilities).bit_length() - 1) // 2
    terms = []
    for index, probability in enumerate(probabilities):
        if probability == 0.0:
            continue
        label = "".join(
            pauli_labels[(index >> (2 * qubit)) & 0b11] for qubit in reversed(range(qubit_count))
        )
        terms.append((label, probability))
    return pauli_error(terms)


def _parse_gate_operation(
    operation: GateSpecification, operation_index: int, qubit_count: int
) -> tuple[str, tuple[int, ...], float | str | None]:
    if not isinstance(operation, Mapping):
        raise ValueError(
            f"Gate at position {operation_index} must be a mapping with 'gate' and 'qubits' fields."
        )

    gate_value = operation.get("gate")
    if not isinstance(gate_value, str) or not gate_value.strip():
        raise ValueError(
            f"Gate at position {operation_index} must have a non-empty string 'gate' name."
        )
    gate = gate_value.strip().lower()
    if gate not in _SUPPORTED_GATES:
        supported = ", ".join(sorted(_SUPPORTED_GATES))
        raise ValueError(
            f"Unsupported gate {gate_value!r} at position {operation_index}. "
            f"Supported gates are: {supported}."
        )

    expected_keys = {"gate", "qubits"}
    if gate in _ANGLE_GATES:
        expected_keys.add("angle")
    unexpected_keys = set(operation) - expected_keys
    missing_keys = expected_keys - set(operation)
    if missing_keys:
        fields = ", ".join(sorted(missing_keys))
        raise ValueError(
            f"Gate {gate!r} at position {operation_index} is missing required field(s): {fields}."
        )
    if unexpected_keys:
        fields = ", ".join(sorted(map(str, unexpected_keys)))
        raise ValueError(
            f"Gate {gate!r} at position {operation_index} has unexpected field(s): {fields}."
        )

    qubits_value = operation["qubits"]
    if not isinstance(qubits_value, Sequence) or isinstance(qubits_value, str | bytes | bytearray):
        raise ValueError(
            f"Field 'qubits' for gate {gate!r} at position {operation_index} "
            "must be a sequence of qubit indices."
        )

    expected_arity = _GATE_ARITY[gate]
    if len(qubits_value) != expected_arity:
        raise ValueError(
            f"Gate {gate!r} at position {operation_index} requires exactly "
            f"{expected_arity} qubit index/indices in 'qubits'; "
            f"received {len(qubits_value)}."
        )

    qubits: list[int] = []
    for operand_index, qubit in enumerate(qubits_value):
        if isinstance(qubit, bool):
            raise ValueError(
                f"Qubit index {operand_index} for gate {gate!r} at position "
                f"{operation_index} must be an integer."
            )
        try:
            normalized_qubit = operator.index(qubit)
        except TypeError as error:
            raise ValueError(
                f"Qubit index {operand_index} for gate {gate!r} at position "
                f"{operation_index} must be an integer."
            ) from error
        if not 0 <= normalized_qubit < qubit_count:
            raise ValueError(
                f"Qubit index {normalized_qubit} for gate {gate!r} at position "
                f"{operation_index} is out of bounds; valid indices are "
                f"0 through {qubit_count - 1}."
            )
        qubits.append(normalized_qubit)

    if len(set(qubits)) != len(qubits):
        raise ValueError(
            f"Gate {gate!r} at position {operation_index} cannot use the "
            "same qubit for multiple operands; control and target qubits "
            "must be distinct."
        )

    angle: float | str | None = None
    if gate in _ANGLE_GATES:
        angle_value = operation["angle"]
        if isinstance(angle_value, str):
            angle = angle_value.strip()
            if not re.fullmatch(r"[A-Za-z_]\w*", angle):
                raise ValueError(
                    f"Gate {gate!r} at position {operation_index} requires "
                    "a finite numeric angle or a valid parameter name."
                )
            return gate, tuple(qubits), angle
        if isinstance(angle_value, bool) or not isinstance(angle_value, Real):
            raise ValueError(
                f"Gate {gate!r} at position {operation_index} requires a "
                "numeric angle or a valid parameter name."
            )
        try:
            angle = float(angle_value)
        except OverflowError as error:
            raise ValueError(
                f"Gate {gate!r} at position {operation_index} requires a finite 'angle' in radians."
            ) from error
        if not math.isfinite(angle):
            raise ValueError(
                f"Gate {gate!r} at position {operation_index} requires a finite 'angle' in radians."
            )

    return gate, tuple(qubits), angle


def validate_circuit_parameters(
    qubit_count: int, gate_sequence: Sequence[GateSpecification]
) -> None:
    """Validate circuit dimensions and an ordered sequence of gate specs.

    Each gate specification has ``gate`` and ``qubits`` fields. Qubit indices
    are zero-based; ``cx`` orders its operands as ``[control, target]``.
    Rotation gates require an ``angle`` in radians. An empty sequence is valid
    and describes the identity circuit.

    Raises:
        ValueError: If the qubit count, sequence, gate, operands, or angle is
            invalid, including when a gate's operands are out of range or
            repeat the same qubit.
    """
    _validate_qubit_count(qubit_count)
    if not isinstance(gate_sequence, Sequence) or isinstance(
        gate_sequence, str | bytes | bytearray
    ):
        raise ValueError("gate_sequence must be an ordered sequence of gate mappings.")
    if len(gate_sequence) > MAX_GATES:
        raise ValueError(
            f"A circuit may contain at most {MAX_GATES} gates; received {len(gate_sequence)}."
        )

    for operation_index, operation in enumerate(gate_sequence):
        _parse_gate_operation(operation, operation_index, qubit_count)


def build_circuit(qubit_count: int, gate_sequence: Sequence[GateSpecification]) -> QuantumCircuit:
    """Build an unmeasured circuit from an ordered sequence of gate specs.

    Gate specs use ``{"gate": ..., "qubits": [...]}``; ``cx`` uses
    ``[control, target]`` and rotations additionally require ``angle`` in
    radians. The returned circuit's qubit indices follow Qiskit's convention.
    An empty gate sequence returns an identity circuit with ``qubit_count``
    qubits.

    Raises:
        ValueError: If circuit parameters fail validation.
    """
    validate_circuit_parameters(qubit_count, gate_sequence)
    circuit = QuantumCircuit(qubit_count)
    parameters: dict[str, Any] = {}

    for operation_index, operation in enumerate(gate_sequence):
        gate, qubits, angle = _parse_gate_operation(operation, operation_index, qubit_count)
        if gate in _ANGLE_GATES:
            if angle is None:
                raise RuntimeError("Validated parameterized gate is missing its angle.")
            if isinstance(angle, str):
                angle = parameters.setdefault(angle, Parameter(angle))
            getattr(circuit, gate)(angle, *qubits)
        else:
            getattr(circuit, gate)(*qubits)

    return circuit


def _validate_unmeasured_circuit(circuit: QuantumCircuit) -> None:
    if not isinstance(circuit, QuantumCircuit):
        raise ValueError("circuit must be a Qiskit QuantumCircuit.")
    _validate_qubit_count(circuit.num_qubits)
    if circuit.num_clbits:
        raise ValueError("circuit must be unmeasured and must not contain classical bits.")
    if circuit.count_ops().get("measure", 0):
        raise ValueError("circuit must be unmeasured.")


def get_statevector_data(
    circuit: QuantumCircuit,
) -> tuple[dict[str, complex], dict[str, float]]:
    """Return ideal statevector amplitudes and probabilities for every outcome.

    Both dictionaries use fixed-width Qiskit little-endian bitstrings,
    ``q[n-1] ... q1 q0``, with q0 on the right. Amplitudes are complex values;
    probabilities are their squared magnitudes. This is an ideal pure-state
    calculation and does not include simulator noise.

    Raises:
        ValueError: If ``circuit`` is not an unmeasured QuantumCircuit with
            between 1 and ``MAX_QUBITS`` qubits.
        QiskitError: If the circuit contains an operation that cannot be
            evaluated as a statevector.
    """
    _validate_unmeasured_circuit(circuit)
    statevector = Statevector.from_instruction(circuit)
    width = circuit.num_qubits
    amplitudes: dict[str, complex] = {}
    probabilities: dict[str, float] = {}
    for index, amplitude in enumerate(statevector.data):
        bitstring = format(index, f"0{width}b")
        amplitudes[bitstring] = complex(amplitude)
        probabilities[bitstring] = float(abs(amplitude) ** 2)
    return amplitudes, probabilities


def get_exact_probabilities(circuit: QuantumCircuit) -> dict[str, float]:
    """Return exact basis-state probabilities for an unmeasured circuit.

    Keys are fixed-width bitstrings in Qiskit's native little-endian
    convention: ``q[n-1] ... q1 q0``, with q0 on the right. All ``2**n``
    basis states are included, including outcomes with zero probability. For
    an identity circuit, the all-zero outcome has probability 1.0.

    Raises:
        ValueError: If ``circuit`` is not an unmeasured QuantumCircuit with
            between 1 and ``MAX_QUBITS`` qubits.
        QiskitError: If the circuit contains an operation that cannot be
            evaluated as a statevector.
    """
    return get_statevector_data(circuit)[1]


def build_noise_model(
    bit_flip_probability: float = 0.0,
    phase_flip_probability: float = 0.0,
    depolarizing_probability: float = 0.0,
    amplitude_damping_probability: float = 0.0,
    readout_error_probability: float = 0.0,
) -> NoiseModel | None:
    """Build an Aer gate and readout noise model.

    Bit-flip, phase-flip, and depolarizing channels are merged analytically
    into one Pauli error. Amplitude damping is composed afterwards, so order
    matters. Supported single-qubit gates receive the single-qubit error.
    ``cx``, ``cz``, ``swap``, and ``cp`` receive the tensor of two
    single-qubit errors plus an optional two-qubit depolarizing error; ``ccx``
    receives a three-way tensor. Readout errors are symmetric. Idle noise is
    not modeled.

    Args:
        bit_flip_probability: X-error probability in the inclusive range
            0.0–0.2.
        phase_flip_probability: Z-error probability in the inclusive range
            0.0–0.2.
        depolarizing_probability: Depolarizing-channel probability in 0.0–0.2.
        amplitude_damping_probability: Amplitude-damping gamma in 0.0–0.2.
        readout_error_probability: Symmetric measurement bit-flip probability
            in 0.0–0.2.

    Raises:
        ValueError: If any probability is non-numeric, non-finite, or
            outside the supported range.
    """
    probabilities: list[float] = []
    named_values = (
        ("bit_flip_probability", bit_flip_probability),
        ("phase_flip_probability", phase_flip_probability),
        ("depolarizing_probability", depolarizing_probability),
        ("amplitude_damping_probability", amplitude_damping_probability),
        ("readout_error_probability", readout_error_probability),
    )
    for name, value in named_values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a number in the range 0.0–0.2.")
        try:
            probability = float(value)
        except OverflowError as error:
            raise ValueError(f"{name} must be a finite number in the range 0.0–0.2.") from error
        if not math.isfinite(probability) or not 0.0 <= probability <= _MAX_NOISE_PROBABILITY:
            raise ValueError(
                f"{name} must be finite and in the range 0.0–{_MAX_NOISE_PROBABILITY}."
            )
        probabilities.append(probability)

    (
        bit_flip,
        phase_flip,
        depolarizing_probability,
        amplitude_damping_probability,
        readout_error_probability,
    ) = probabilities
    if not any(probabilities):
        return None

    one_qubit_paulis = [1.0 - bit_flip, bit_flip, 0.0, 0.0]
    one_qubit_paulis = _convolve_pauli_probabilities(
        one_qubit_paulis,
        [1.0 - phase_flip, 0.0, phase_flip, 0.0],
    )
    if depolarizing_probability:
        one_qubit_paulis = _convolve_pauli_probabilities(
            one_qubit_paulis,
            [
                1.0 - 3.0 * depolarizing_probability / 4.0,
                depolarizing_probability / 4.0,
                depolarizing_probability / 4.0,
                depolarizing_probability / 4.0,
            ],
        )
    single_qubit_error = _pauli_error_from_probabilities(one_qubit_paulis)
    if amplitude_damping_probability:
        single_qubit_error = single_qubit_error.compose(
            amplitude_damping_error(amplitude_damping_probability)
        )

    noise_model = NoiseModel()
    if bit_flip or phase_flip or depolarizing_probability or amplitude_damping_probability:
        noise_model.add_all_qubit_quantum_error(single_qubit_error, list(_NOISY_SINGLE_QUBIT_GATES))
        two_qubit_error = single_qubit_error.tensor(single_qubit_error)
        if depolarizing_probability:
            two_qubit_paulis = [
                1.0 - 15.0 * depolarizing_probability / 16.0,
                *([depolarizing_probability / 16.0] * 15),
            ]
            two_qubit_error = two_qubit_error.compose(
                _pauli_error_from_probabilities(two_qubit_paulis)
            )
        noise_model.add_all_qubit_quantum_error(two_qubit_error, ["cx", "cz", "swap", "cp"])
        noise_model.add_all_qubit_quantum_error(two_qubit_error.tensor(single_qubit_error), ["ccx"])
    if readout_error_probability:
        noise_model.add_all_qubit_readout_error(
            ReadoutError(
                [
                    [1.0 - readout_error_probability, readout_error_probability],
                    [readout_error_probability, 1.0 - readout_error_probability],
                ]
            )
        )
    return noise_model


def run_shot_experiment(
    circuit: QuantumCircuit,
    shots: int,
    seed_simulator: int | None = None,
    noise_model: NoiseModel | None = None,
    max_shots: int = MAX_SHOTS,
) -> dict[str, int]:
    """Run a measurement experiment and return counts by bitstring.

    Measurements are added to a circuit copy, mapping qubit ``i`` to classical
    bit ``i``. Returned keys therefore use Qiskit's native little-endian
    convention, ``q[n-1] ... q1 q0``, with q0 on the right. The original
    circuit is not modified. For an identity circuit, every shot returns the
    all-zero bitstring.

    Args:
        circuit: An unmeasured QuantumCircuit with 1 to ``MAX_QUBITS`` qubits.
        shots: Number of measurements, between 1 and ``MAX_SHOTS``.
        seed_simulator: Optional non-negative integer seed for repeatability.
        noise_model: Optional Aer NoiseModel applied to gate execution and/or
            readout during sampling only.
        max_shots: Configurable upper shot bound, capped at ``MAX_SHOTS``.

    Raises:
        ValueError: If the circuit, shot count, seed, or noise model is invalid.
    """
    _validate_unmeasured_circuit(circuit)
    if isinstance(shots, bool):
        raise ValueError("shots must be an integer.")
    try:
        shots = operator.index(shots)
    except TypeError as error:
        raise ValueError("shots must be an integer.") from error
    if isinstance(max_shots, bool):
        raise ValueError("max_shots must be an integer.")
    try:
        max_shots = operator.index(max_shots)
    except TypeError as error:
        raise ValueError("max_shots must be an integer.") from error
    max_shots = min(max_shots, MAX_SHOTS)
    if max_shots < 1:
        raise ValueError("max_shots must be a positive integer.")
    if shots < 1:
        raise ValueError("shots must be a positive integer.")
    if shots > max_shots:
        raise ValueError(f"shots cannot exceed the configured maximum ({max_shots}).")
    if seed_simulator is not None:
        if isinstance(seed_simulator, bool):
            raise ValueError("seed_simulator must be a non-negative integer or None.")
        try:
            seed_simulator = operator.index(seed_simulator)
        except TypeError as error:
            raise ValueError("seed_simulator must be a non-negative integer or None.") from error
        if seed_simulator < 0:
            raise ValueError("seed_simulator must be a non-negative integer or None.")
    if noise_model is not None and not isinstance(noise_model, NoiseModel):
        raise ValueError("noise_model must be a Qiskit Aer NoiseModel or None.")

    measured_circuit = circuit.copy()
    measured_circuit.measure_all()
    result = (
        AerSimulator(noise_model=noise_model)
        .run(measured_circuit, shots=shots, seed_simulator=seed_simulator)
        .result()
    )
    raw_counts = result.get_counts(measured_circuit)

    width = circuit.num_qubits
    counts: dict[str, int] = {}
    for raw_bitstring, count in raw_counts.items():
        bitstring = raw_bitstring.replace(" ", "")
        if len(bitstring) > width:
            raise RuntimeError("AerSimulator returned an outcome wider than the circuit register.")
        normalized_bitstring = bitstring.zfill(width)
        counts[normalized_bitstring] = counts.get(normalized_bitstring, 0) + count
    return counts


def draw_circuit_text(circuit: QuantumCircuit) -> str:
    """Return the unmeasured circuit's Qiskit text drawing.

    Raises:
        ValueError: If ``circuit`` is not an unmeasured QuantumCircuit with
            between 1 and ``MAX_QUBITS`` qubits.
    """
    _validate_unmeasured_circuit(circuit)
    return str(circuit.draw(output="text"))


def draw_circuit_mpl(circuit: QuantumCircuit) -> Any:
    """Return a Matplotlib drawing of the unmeasured circuit."""
    _validate_unmeasured_circuit(circuit)
    import matplotlib

    matplotlib.use("Agg", force=True)
    return circuit.draw(output="mpl")


def build_prefix_circuit(
    qubit_count: int,
    gate_sequence: Sequence[GateSpecification],
    gate_count: int,
) -> QuantumCircuit:
    """Build the circuit after its first ``gate_count`` gates."""
    validate_circuit_parameters(qubit_count, gate_sequence)
    if isinstance(gate_count, bool):
        raise ValueError("gate_count must be a non-negative integer.")
    try:
        gate_count = operator.index(gate_count)
    except TypeError as error:
        raise ValueError("gate_count must be a non-negative integer.") from error
    if not 0 <= gate_count <= len(gate_sequence):
        raise ValueError(f"gate_count must be between 0 and {len(gate_sequence)}.")
    return build_circuit(qubit_count, gate_sequence[:gate_count])


def bind_circuit_parameters(
    circuit: QuantumCircuit, parameter_values: Mapping[str, float]
) -> QuantumCircuit:
    """Return a copy of a parameterized circuit with named values assigned."""
    unknown_names = set(parameter_values) - {parameter.name for parameter in circuit.parameters}
    if unknown_names:
        raise ValueError(f"Unknown circuit parameter(s): {', '.join(sorted(unknown_names))}.")
    bindings: dict[Any, float] = {}
    for parameter in circuit.parameters:
        if parameter.name not in parameter_values:
            continue
        value = parameter_values[parameter.name]
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"Parameter {parameter.name!r} must have a finite numeric value.")
        bindings[parameter] = float(value)
    return circuit.assign_parameters(bindings, inplace=False)


def run_parameter_sweep(
    circuit: QuantumCircuit,
    parameter_name: str,
    values: Sequence[float],
    fixed_values: Mapping[str, float] | None = None,
) -> list[dict[str, object]]:
    """Calculate ideal probabilities across values of a named circuit parameter.

    Every other circuit parameter must be supplied in ``fixed_values``.
    """
    parameter_names = {parameter.name for parameter in circuit.parameters}
    if parameter_name not in parameter_names:
        raise ValueError(f"Unknown circuit parameter {parameter_name!r}.")
    if len(values) == 0:
        raise ValueError("values must contain at least one parameter value.")
    fixed_values = fixed_values or {}
    missing_names = parameter_names - {parameter_name} - set(fixed_values)
    if missing_names:
        raise ValueError(
            f"Missing fixed value(s) for circuit parameter(s): {', '.join(sorted(missing_names))}."
        )

    results: list[dict[str, object]] = []
    for value in values:
        bindings = {**fixed_values, parameter_name: value}
        bound = bind_circuit_parameters(circuit, bindings)
        _, probabilities = get_statevector_data(bound)
        results.append({"value": float(value), "probabilities": probabilities})
    return results


def get_qubit_state_metrics(circuit: QuantumCircuit) -> list[dict[str, float | int]]:
    """Return each qubit's Bloch vector, reduced-state purity, and entropy."""
    _validate_unmeasured_circuit(circuit)
    statevector = Statevector.from_instruction(circuit)
    metrics: list[dict[str, float | int]] = []
    for qubit_index in range(circuit.num_qubits):
        traced_out = [index for index in range(circuit.num_qubits) if index != qubit_index]
        reduced_state = partial_trace(statevector, traced_out)
        matrix = np.asarray(reduced_state.data, dtype=complex)
        bloch_x = float(2.0 * matrix[0, 1].real)
        bloch_y = float(-2.0 * matrix[0, 1].imag)
        bloch_z = float((matrix[0, 0] - matrix[1, 1]).real)
        purity = float(np.trace(matrix @ matrix).real)
        von_neumann_entropy = float(entropy(reduced_state, base=2))
        metrics.append(
            {
                "qubit": qubit_index,
                "x": bloch_x,
                "y": bloch_y,
                "z": bloch_z,
                "purity": purity,
                "entropy": von_neumann_entropy,
            }
        )
    return metrics


def get_noisy_probabilities(
    circuit: QuantumCircuit,
    noise_model: NoiseModel,
    readout_error_probability: float = 0.0,
) -> dict[str, float]:
    """Return exact noisy measurement probabilities using Aer density matrices.

    Gate noise is simulated by Aer with the supplied noise model. If the
    model includes symmetric readout errors, the same probability must be
    passed as ``readout_error_probability`` to transform the final diagonal
    distribution into observed measurement probabilities.
    """
    _validate_unmeasured_circuit(circuit)
    if circuit.num_qubits > MAX_NOISY_EXACT_QUBITS:
        raise ValueError(
            "Exact noisy probability simulation is limited to "
            f"{MAX_NOISY_EXACT_QUBITS} qubits; received {circuit.num_qubits}."
        )
    if not isinstance(noise_model, NoiseModel):
        raise ValueError("noise_model must be a Qiskit Aer NoiseModel.")
    if (
        isinstance(readout_error_probability, bool)
        or not isinstance(readout_error_probability, Real)
        or not math.isfinite(float(readout_error_probability))
        or not 0.0 <= float(readout_error_probability) <= _MAX_NOISE_PROBABILITY
    ):
        raise ValueError("readout_error_probability must be in the range 0.0–0.2.")

    noisy_circuit = circuit.copy()
    noisy_circuit.save_density_matrix()
    result = (
        AerSimulator(method="density_matrix", noise_model=noise_model).run(noisy_circuit).result()
    )
    density_matrix = DensityMatrix(result.data(0)["density_matrix"])
    diagonal = np.real(np.diag(np.asarray(density_matrix.data)))
    width = circuit.num_qubits
    readout_probability = float(readout_error_probability)
    if readout_probability:
        distribution = diagonal.reshape((2,) * width)
        readout_matrix = np.array(
            [
                [1.0 - readout_probability, readout_probability],
                [readout_probability, 1.0 - readout_probability],
            ]
        )
        for axis in range(width):
            distribution = np.moveaxis(
                np.tensordot(readout_matrix, distribution, axes=([1], [axis])),
                0,
                axis,
            )
        diagonal = distribution.reshape(-1)
    exact_noisy = {
        format(index, f"0{width}b"): max(0.0, float(probability))
        for index, probability in enumerate(diagonal)
    }
    total_probability = sum(exact_noisy.values())
    if not math.isfinite(total_probability) or total_probability <= 0.0:
        raise RuntimeError("Noisy simulation produced an invalid probability distribution.")
    return {
        outcome: probability / total_probability for outcome, probability in exact_noisy.items()
    }


def export_circuit_json(qubit_count: int, gate_sequence: Sequence[GateSpecification]) -> str:
    """Serialize a validated ordered circuit description as JSON."""
    validate_circuit_parameters(qubit_count, gate_sequence)
    return json.dumps(
        {"qubit_count": qubit_count, "gates": list(gate_sequence)},
        indent=2,
    )


@lru_cache(maxsize=MAX_QUBITS)
def _build_qft_sequence_json(qubit_count: int) -> str:
    _validate_qubit_count(qubit_count)
    gates: list[dict[str, object]] = []
    decomposed = QFTGate(qubit_count).definition.decompose()
    for instruction in decomposed.data:
        gate = instruction.operation.name
        qubits = [decomposed.find_bit(qubit).index for qubit in instruction.qubits]
        if gate == "u":
            theta, phi, lam = (float(value) for value in instruction.operation.params)
            gates.extend(
                [
                    {"gate": "rz", "qubits": qubits, "angle": lam},
                    {"gate": "ry", "qubits": qubits, "angle": theta},
                    {"gate": "rz", "qubits": qubits, "angle": phi},
                ]
            )
        elif gate in _SUPPORTED_GATES:
            gate_spec: dict[str, object] = {"gate": gate, "qubits": qubits}
            if gate in _ANGLE_GATES:
                gate_spec["angle"] = float(instruction.operation.params[0])
            gates.append(gate_spec)
        else:
            raise ValueError(f"QFT decomposition produced unsupported gate {gate!r}.")
    validate_circuit_parameters(qubit_count, gates)
    return json.dumps(gates)


def build_qft_sequence(qubit_count: int) -> list[dict[str, object]]:
    """Return a fresh supported-gate decomposition of Qiskit's exact QFT unitary.

    The supported decomposition is limited to 4 qubits by ``MAX_GATES``.
    """
    _validate_qubit_count(qubit_count)
    return json.loads(_build_qft_sequence_json(operator.index(qubit_count)))


@lru_cache(maxsize=16)
def _build_grover_sequence_json(qubit_count: int, marked_state: str) -> str:
    _validate_qubit_count(qubit_count)
    if qubit_count not in (2, 3):
        raise ValueError("The teaching Grover sequence supports two or three qubits.")
    if len(marked_state) != qubit_count or any(bit not in "01" for bit in marked_state):
        raise ValueError("marked_state must be a fixed-width binary string.")
    gates: list[dict[str, object]] = [
        {"gate": "h", "qubits": [qubit]} for qubit in range(qubit_count)
    ]

    def append_ccz_oracle() -> None:
        if qubit_count == 2:
            gates.extend(
                [
                    {"gate": "h", "qubits": [1]},
                    {"gate": "cx", "qubits": [0, 1]},
                    {"gate": "h", "qubits": [1]},
                ]
            )
        else:
            gates.extend(
                [
                    {"gate": "h", "qubits": [2]},
                    {"gate": "ccx", "qubits": [0, 1, 2]},
                    {"gate": "h", "qubits": [2]},
                ]
            )

    for qubit, bit in enumerate(reversed(marked_state)):
        if bit == "0":
            gates.append({"gate": "x", "qubits": [qubit]})
    append_ccz_oracle()
    for qubit, bit in enumerate(reversed(marked_state)):
        if bit == "0":
            gates.append({"gate": "x", "qubits": [qubit]})

    gates.extend({"gate": "h", "qubits": [qubit]} for qubit in range(qubit_count))
    gates.extend({"gate": "x", "qubits": [qubit]} for qubit in range(qubit_count))
    append_ccz_oracle()
    gates.extend({"gate": "x", "qubits": [qubit]} for qubit in range(qubit_count))
    gates.extend({"gate": "h", "qubits": [qubit]} for qubit in range(qubit_count))
    validate_circuit_parameters(qubit_count, gates)
    return json.dumps(gates)


def build_grover_sequence(qubit_count: int, marked_state: str) -> list[dict[str, object]]:
    """Return a fresh gate sequence for one Grover iteration."""
    _validate_qubit_count(qubit_count)
    return json.loads(_build_grover_sequence_json(operator.index(qubit_count), marked_state))


def import_circuit_json(content: str) -> tuple[int, list[dict[str, object]]]:
    """Parse and validate a JSON circuit description."""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(f"This file is not valid JSON. (Line {error.lineno}.)") from error
    if not isinstance(payload, dict):
        raise ValueError("Circuit file must be a JSON object.")
    if set(payload) != {"qubit_count", "gates"}:
        raise ValueError("Circuit JSON must contain exactly 'qubit_count' and 'gates'.")
    qubit_count = payload["qubit_count"]
    gates = payload["gates"]
    if not isinstance(gates, list):
        raise ValueError("Circuit JSON 'gates' must be a list.")
    validate_circuit_parameters(qubit_count, gates)
    return qubit_count, gates


def export_circuit_qasm(circuit: QuantumCircuit) -> str:
    """Export an unmeasured circuit as OpenQASM 3 source."""
    _validate_unmeasured_circuit(circuit)
    return qasm3.dumps(circuit)


def calculate_shot_standard_errors(counts: Mapping[str, int], shots: int) -> dict[str, float]:
    """Calculate binomial standard errors for observed outcome frequencies."""
    if isinstance(shots, bool):
        raise ValueError("shots must be a positive integer.")
    try:
        shots = operator.index(shots)
    except TypeError as error:
        raise ValueError("shots must be a positive integer.") from error
    if not 1 <= shots <= MAX_SHOTS:
        raise ValueError(f"shots must be between 1 and {MAX_SHOTS}.")
    return {
        outcome: math.sqrt((count / shots) * (1.0 - count / shots) / shots)
        for outcome, count in counts.items()
    }


def run_toy_vqe() -> dict[str, object]:
    """Minimize a teaching Hamiltonian ``Z`` using a one-parameter Ry ansatz."""
    parameterized = build_circuit(1, [{"gate": "ry", "qubits": [0], "angle": "theta"}])

    def energy(theta: float) -> float:
        bound = bind_circuit_parameters(parameterized, {"theta": theta})
        probabilities = get_exact_probabilities(bound)
        return probabilities["0"] - probabilities["1"]

    grid_count = 128
    grid = [2.0 * math.pi * index / grid_count for index in range(grid_count)]
    best_index = min(range(grid_count), key=lambda index: energy(grid[index]))
    center = grid[best_index]
    left = center - 2.0 * math.pi / grid_count
    right = center + 2.0 * math.pi / grid_count
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = right - ratio * (right - left)
    x2 = left + ratio * (right - left)
    y1 = energy(x1)
    y2 = energy(x2)
    history: list[dict[str, float]] = [
        {"theta": grid[index], "energy": energy(grid[index])} for index in range(grid_count)
    ]
    for _ in range(48):
        if y1 < y2:
            right, x2, y2 = x2, x1, y1
            x1 = right - ratio * (right - left)
            y1 = energy(x1)
            history.append({"theta": x1 % (2 * math.pi), "energy": y1})
        else:
            left, x1, y1 = x1, x2, y2
            x2 = left + ratio * (right - left)
            y2 = energy(x2)
            history.append({"theta": x2 % (2 * math.pi), "energy": y2})
    theta_opt = (left + right) / 2.0
    return {
        "hamiltonian": "Z",
        "theta": theta_opt % (2 * math.pi),
        "energy": energy(theta_opt),
        "history": history,
        "ground_energy": -1.0,
    }
