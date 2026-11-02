import json
import math
import time
import unittest

import numpy as np
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Operator
from qiskit_aer.noise import (
    NoiseModel,
    amplitude_damping_error,
    depolarizing_error,
    pauli_error,
)

from quantum_engine import (
    MAX_GATES,
    MAX_NOISY_EXACT_QUBITS,
    MAX_SHOTS,
    build_circuit,
    build_grover_sequence,
    build_noise_model,
    build_qft_sequence,
    calculate_shot_standard_errors,
    draw_circuit_mpl,
    export_circuit_json,
    export_circuit_qasm,
    get_exact_probabilities,
    get_noisy_probabilities,
    get_qubit_state_metrics,
    get_statevector_data,
    import_circuit_json,
    run_parameter_sweep,
    run_shot_experiment,
    run_toy_vqe,
)


class StatevectorDataTests(unittest.TestCase):
    def test_amplitudes_and_probabilities_use_little_endian_keys(self) -> None:
        circuit = build_circuit(2, [{"gate": "x", "qubits": [0]}])

        amplitudes, probabilities = get_statevector_data(circuit)

        self.assertEqual(list(amplitudes), ["00", "01", "10", "11"])
        self.assertEqual(amplitudes["01"], 1 + 0j)
        self.assertEqual(probabilities["01"], 1.0)
        self.assertEqual(probabilities, get_exact_probabilities(circuit))

    def test_phase_amplitude_is_complex_and_normalized(self) -> None:
        circuit = build_circuit(1, [{"gate": "rx", "qubits": [0], "angle": math.pi / 2}])

        amplitudes, probabilities = get_statevector_data(circuit)

        self.assertAlmostEqual(amplitudes["0"].real, math.sqrt(0.5))
        self.assertAlmostEqual(amplitudes["1"].imag, -math.sqrt(0.5))
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)


class NoiseModelTests(unittest.TestCase):
    def test_zero_noise_returns_no_model(self) -> None:
        self.assertIsNone(build_noise_model(0.0, 0.0))

    def test_noise_probabilities_are_bounded(self) -> None:
        for invalid in (-0.01, 0.21, math.nan, math.inf, True):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    build_noise_model(invalid, 0.0)

    def test_bit_flip_noise_changes_shot_counts_not_exact_state(self) -> None:
        circuit = build_circuit(1, [{"gate": "x", "qubits": [0]}])
        noise_model = build_noise_model(0.2, 0.0)

        exact = get_exact_probabilities(circuit)
        first_counts = run_shot_experiment(
            circuit, 1000, seed_simulator=42, noise_model=noise_model
        )
        repeated_counts = run_shot_experiment(
            circuit, 1000, seed_simulator=42, noise_model=noise_model
        )

        self.assertEqual(exact, {"0": 0.0, "1": 1.0})
        self.assertEqual(first_counts, repeated_counts)
        self.assertGreater(first_counts.get("0", 0), 0)
        self.assertEqual(sum(first_counts.values()), 1000)

    def test_phase_flip_noise_changes_interference_measurement(self) -> None:
        circuit = build_circuit(
            1,
            [
                {"gate": "h", "qubits": [0]},
                {"gate": "h", "qubits": [0]},
            ],
        )
        noise_model = build_noise_model(0.0, 0.2)

        exact = get_exact_probabilities(circuit)
        noisy_counts = run_shot_experiment(
            circuit, 1000, seed_simulator=123, noise_model=noise_model
        )

        self.assertAlmostEqual(exact["0"], 1.0)
        self.assertGreater(noisy_counts.get("1", 0), 0)
        self.assertEqual(sum(noisy_counts.values()), 1000)

    def test_bit_flip_and_depolarizing_paulis_merge_analytically(self) -> None:
        circuit = build_circuit(1, [{"gate": "x", "qubits": [0]}])
        noise_model = build_noise_model(
            bit_flip_probability=0.1,
            depolarizing_probability=0.1,
        )

        probabilities = get_noisy_probabilities(circuit, noise_model)

        self.assertAlmostEqual(probabilities["0"], 0.14, delta=1e-9)

    def test_phase_flip_channel_probability_is_analytical(self) -> None:
        circuit = build_circuit(
            1,
            [
                {"gate": "h", "qubits": [0]},
                {"gate": "h", "qubits": [0]},
            ],
        )
        noise_model = build_noise_model(phase_flip_probability=0.2)

        probabilities = get_noisy_probabilities(circuit, noise_model)

        self.assertAlmostEqual(probabilities["1"], 0.2, delta=1e-9)

    def test_amplitude_damping_remains_after_pauli_channels(self) -> None:
        circuit = build_circuit(1, [{"gate": "x", "qubits": [0]}])
        noise_model = build_noise_model(amplitude_damping_probability=0.1)

        probabilities = get_noisy_probabilities(circuit, noise_model)

        self.assertAlmostEqual(probabilities["0"], 0.1, delta=1e-9)

    def test_all_noise_channels_build_quickly_and_simulate_multi_qubit_gates(self) -> None:
        started = time.perf_counter()
        noise_model = build_noise_model(0.05, 0.05, 0.05, 0.05, 0.05)
        build_duration = time.perf_counter() - started

        self.assertLess(build_duration, 3.0)
        circuit = build_circuit(
            3,
            [
                {"gate": "cx", "qubits": [0, 1]},
                {"gate": "cz", "qubits": [1, 2]},
                {"gate": "swap", "qubits": [0, 2]},
                {"gate": "cp", "qubits": [0, 1], "angle": 0.3},
                {"gate": "ccx", "qubits": [0, 1, 2]},
            ],
        )

        probabilities = get_noisy_probabilities(circuit, noise_model, 0.05)

        self.assertAlmostEqual(sum(probabilities.values()), 1.0, delta=1e-9)

    def test_merged_noise_matches_public_aer_composition_reference(self) -> None:
        noise_model = build_noise_model(0.05, 0.05, 0.05, 0.05)
        reference_error = (
            pauli_error([("I", 0.95), ("X", 0.05)])
            .compose(pauli_error([("I", 0.95), ("Z", 0.05)]))
            .compose(depolarizing_error(0.05, 1))
            .compose(amplitude_damping_error(0.05))
        )
        reference_model = NoiseModel()
        reference_model.add_all_qubit_quantum_error(reference_error, ["h", "x", "ry"])
        circuit = build_circuit(
            1,
            [
                {"gate": "h", "qubits": [0]},
                {"gate": "ry", "qubits": [0], "angle": 0.7},
                {"gate": "x", "qubits": [0]},
                {"gate": "h", "qubits": [0]},
            ],
        )

        actual = get_noisy_probabilities(circuit, noise_model)
        expected = get_noisy_probabilities(circuit, reference_model)

        for outcome in expected:
            self.assertAlmostEqual(actual[outcome], expected[outcome], delta=1e-7)


class ExpandedEngineFeatureTests(unittest.TestCase):
    def test_numpy_integer_qubits_and_distinct_multi_qubit_operands(self) -> None:
        circuit = build_circuit(2, [{"gate": "cx", "qubits": [np.int64(0), np.int64(1)]}])
        self.assertEqual(circuit.num_qubits, 2)
        with self.assertRaisesRegex(ValueError, "same qubit"):
            build_circuit(2, [{"gate": "cx", "qubits": [np.int64(0), np.int64(0)]}])

    def test_circuit_gate_limit_accepts_50_and_rejects_51(self) -> None:
        fifty_gates = [{"gate": "x", "qubits": [0]}] * MAX_GATES
        circuit = build_circuit(1, fifty_gates)
        self.assertEqual(len(circuit.data), MAX_GATES)

        with self.assertRaisesRegex(ValueError, "at most 50 gates; received 51"):
            build_circuit(1, [*fifty_gates, {"gate": "x", "qubits": [0]}])

    def test_json_import_rejects_more_than_max_gates(self) -> None:
        payload = json.dumps(
            {
                "qubit_count": 1,
                "gates": [{"gate": "x", "qubits": [0]}] * (MAX_GATES + 1),
            }
        )

        with self.assertRaisesRegex(ValueError, "at most 50 gates; received 51"):
            import_circuit_json(payload)

    def test_extended_gate_set_and_combined_noise_model(self) -> None:
        gates = [
            {"gate": "h", "qubits": [0]},
            {"gate": "s", "qubits": [1]},
            {"gate": "t", "qubits": [2]},
            {"gate": "p", "qubits": [0], "angle": 0.3},
            {"gate": "rx", "qubits": [1], "angle": 0.2},
            {"gate": "ry", "qubits": [2], "angle": 0.4},
            {"gate": "rz", "qubits": [0], "angle": 0.5},
            {"gate": "cx", "qubits": [0, 1]},
            {"gate": "cz", "qubits": [1, 2]},
            {"gate": "cp", "qubits": [0, 2], "angle": 0.1},
            {"gate": "swap", "qubits": [1, 2]},
            {"gate": "ccx", "qubits": [0, 1, 2]},
        ]
        circuit = build_circuit(3, gates)
        noise_model = build_noise_model(0.01, 0.01, 0.01, 0.01, 0.01)

        noisy_probabilities = get_noisy_probabilities(circuit, noise_model, 0.01)
        counts = run_shot_experiment(circuit, 1000, seed_simulator=42, noise_model=noise_model)

        self.assertAlmostEqual(sum(noisy_probabilities.values()), 1.0)
        self.assertEqual(sum(counts.values()), 1000)

    def test_shot_cap_and_binomial_standard_errors(self) -> None:
        circuit = build_circuit(1, [])
        with self.assertRaisesRegex(ValueError, str(MAX_SHOTS)):
            run_shot_experiment(circuit, MAX_SHOTS + 1)

        errors = calculate_shot_standard_errors({"0": 50, "1": 50}, 100)
        self.assertAlmostEqual(errors["0"], 0.05)
        self.assertAlmostEqual(errors["1"], 0.05)

    def test_shot_seed_accepts_numpy_integer_but_rejects_bool_and_negative(self) -> None:
        circuit = build_circuit(1, [{"gate": "h", "qubits": [0]}])
        numpy_seed_counts = run_shot_experiment(circuit, 100, seed_simulator=np.int64(12))
        integer_seed_counts = run_shot_experiment(circuit, 100, seed_simulator=12)

        self.assertEqual(numpy_seed_counts, integer_seed_counts)
        for invalid_seed in (True, -1):
            with self.subTest(seed=invalid_seed):
                with self.assertRaisesRegex(ValueError, "seed_simulator"):
                    run_shot_experiment(circuit, 10, seed_simulator=invalid_seed)

    def test_qft_and_grover_presets_have_expected_quantum_behavior(self) -> None:
        qft_circuit = build_circuit(3, build_qft_sequence(3))
        self.assertTrue(Operator(qft_circuit).equiv(Operator(QFTGate(3))))

        grover_probabilities = get_exact_probabilities(
            build_circuit(3, build_grover_sequence(3, "111"))
        )
        self.assertGreater(grover_probabilities["111"], 0.78)
        self.assertAlmostEqual(sum(grover_probabilities.values()), 1.0)

    def test_cached_algorithm_sequences_are_returned_as_fresh_copies(self) -> None:
        qft_sequence = build_qft_sequence(3)
        expected_qft_first_gate = {
            key: value.copy() if isinstance(value, list) else value
            for key, value in qft_sequence[0].items()
        }
        qft_sequence[0]["qubits"][0] = 9
        qft_sequence.append({"gate": "x", "qubits": [0]})
        self.assertEqual(build_qft_sequence(3)[0], expected_qft_first_gate)
        self.assertEqual(len(build_qft_sequence(3)), len(qft_sequence) - 1)

        grover_sequence = build_grover_sequence(3, "111")
        expected_grover_first_gate = {
            key: value.copy() if isinstance(value, list) else value
            for key, value in grover_sequence[0].items()
        }
        grover_sequence[0]["qubits"][0] = 9
        grover_sequence.clear()
        next_grover_sequence = build_grover_sequence(3, "111")
        self.assertEqual(next_grover_sequence[0], expected_grover_first_gate)
        self.assertTrue(next_grover_sequence)

    def test_json_round_trip_and_qasm_export(self) -> None:
        gates = [
            {"gate": "h", "qubits": [0]},
            {"gate": "ry", "qubits": [1], "angle": "theta"},
        ]
        encoded = export_circuit_json(2, gates)
        self.assertEqual(import_circuit_json(encoded), (2, gates))

        qasm = export_circuit_qasm(build_circuit(2, gates[:1]))
        self.assertIn("OPENQASM 3", qasm)
        self.assertIn("h q[0];", qasm)

    def test_matplotlib_circuit_drawing_returns_a_figure(self) -> None:
        import matplotlib
        import matplotlib.pyplot as plt

        circuit = build_circuit(2, [{"gate": "h", "qubits": [0]}, {"gate": "cx", "qubits": [0, 1]}])
        figure = draw_circuit_mpl(circuit)
        try:
            self.assertTrue(hasattr(figure, "savefig"))
            self.assertEqual(matplotlib.get_backend().lower(), "agg")
        finally:
            plt.close(figure)

    def test_noisy_density_probabilities_include_readout_errors(self) -> None:
        circuit = build_circuit(1, [{"gate": "x", "qubits": [0]}])
        noise_model = build_noise_model(readout_error_probability=0.1)
        probabilities = get_noisy_probabilities(circuit, noise_model, 0.1)

        self.assertAlmostEqual(probabilities["0"], 0.1)
        self.assertAlmostEqual(probabilities["1"], 0.9)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)

    def test_vectorized_readout_matches_brute_force_reference(self) -> None:
        rng = np.random.default_rng(42)
        gates = [
            {"gate": "ry", "qubits": [qubit], "angle": float(rng.uniform(0, 2 * math.pi))}
            for qubit in range(3)
        ]
        gates.extend(
            [
                {"gate": "cx", "qubits": [0, 1]},
                {"gate": "cx", "qubits": [1, 2]},
                {"gate": "rz", "qubits": [2], "angle": float(rng.uniform(0, 2 * math.pi))},
            ]
        )
        circuit = build_circuit(3, gates)
        initial_probabilities = list(get_exact_probabilities(circuit).values())
        readout_probability = 0.1
        noise_model = build_noise_model(readout_error_probability=readout_probability)

        actual = get_noisy_probabilities(circuit, noise_model, readout_probability)
        expected: dict[str, float] = {}
        for observed_index in range(8):
            observed_bits = format(observed_index, "03b")
            probability = 0.0
            for true_index, true_probability in enumerate(initial_probabilities):
                true_bits = format(true_index, "03b")
                differing_bits = sum(
                    observed != actual_bit
                    for observed, actual_bit in zip(observed_bits, true_bits, strict=True)
                )
                probability += (
                    true_probability
                    * readout_probability**differing_bits
                    * (1.0 - readout_probability) ** (3 - differing_bits)
                )
            expected[observed_bits] = probability

        for outcome, probability in expected.items():
            self.assertAlmostEqual(actual[outcome], probability, delta=1e-12)

    def test_exact_noisy_probabilities_enforce_qubit_limit(self) -> None:
        circuit = build_circuit(MAX_NOISY_EXACT_QUBITS + 1, [])
        noise_model = build_noise_model(bit_flip_probability=0.1)

        with self.assertRaisesRegex(ValueError, "limited to 8 qubits"):
            get_noisy_probabilities(circuit, noise_model)

    def test_parameter_sweep_and_toy_vqe_converge(self) -> None:
        circuit = build_circuit(1, [{"gate": "ry", "qubits": [0], "angle": "theta"}])
        sweep = run_parameter_sweep(circuit, "theta", [0.0, math.pi])

        self.assertAlmostEqual(sweep[0]["probabilities"]["0"], 1.0)
        self.assertAlmostEqual(sweep[1]["probabilities"]["1"], 1.0)
        result = run_toy_vqe()
        self.assertAlmostEqual(result["energy"], -1.0, places=6)
        self.assertAlmostEqual(result["ground_energy"], -1.0)

    def test_multi_parameter_sweep_binds_other_parameters(self) -> None:
        circuit = build_circuit(
            1,
            [
                {"gate": "ry", "qubits": [0], "angle": "a"},
                {"gate": "rx", "qubits": [0], "angle": "b"},
            ],
        )

        sweep = run_parameter_sweep(circuit, "a", [0.0, math.pi], fixed_values={"b": 0.0})

        self.assertAlmostEqual(sweep[0]["probabilities"]["0"], 1.0)
        self.assertAlmostEqual(sweep[0]["probabilities"]["1"], 0.0)
        self.assertAlmostEqual(sweep[1]["probabilities"]["0"], 0.0)
        self.assertAlmostEqual(sweep[1]["probabilities"]["1"], 1.0)

    def test_multi_parameter_sweep_requires_all_other_values(self) -> None:
        circuit = build_circuit(
            1,
            [
                {"gate": "ry", "qubits": [0], "angle": "a"},
                {"gate": "rx", "qubits": [0], "angle": "b"},
                {"gate": "rz", "qubits": [0], "angle": "c"},
            ],
        )

        with self.assertRaisesRegex(ValueError, "b"):
            run_parameter_sweep(circuit, "a", [0.0], fixed_values={"c": 0.0})

    def test_reduced_state_metrics_report_bell_entanglement(self) -> None:
        bell = build_circuit(2, [{"gate": "h", "qubits": [0]}, {"gate": "cx", "qubits": [0, 1]}])
        metrics = get_qubit_state_metrics(bell)

        for qubit_metrics in metrics:
            self.assertAlmostEqual(qubit_metrics["purity"], 0.5)
            self.assertAlmostEqual(qubit_metrics["entropy"], 1.0)


if __name__ == "__main__":
    unittest.main()
