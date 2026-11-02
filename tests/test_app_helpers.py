import matplotlib

matplotlib.use("Agg", force=True)

import json
import math  # noqa: E402
import unittest  # noqa: E402
from copy import deepcopy  # noqa: E402
from itertools import product  # noqa: E402
from unittest.mock import Mock, patch  # noqa: E402

import numpy as np  # noqa: E402
from qiskit.circuit.library import QFTGate  # noqa: E402
from qiskit.quantum_info import Operator  # noqa: E402

import app as app_module  # noqa: E402
from app import (
    _build_comparison_data,
    _build_state_inspector_data,
    _comparison_legend_methods,
    _filter_result_frames,
    _rank_outcomes,
    _select_sweep_outcomes,
)  # noqa: E402
from quantum_engine import (  # noqa: E402
    build_circuit,
    build_grover_sequence,
    build_qft_sequence,
    get_exact_probabilities,
    get_statevector_data,
    run_parameter_sweep,
)


class ComparisonDataTests(unittest.TestCase):
    def test_comparison_aligns_outcomes_and_preserves_zero_padding(self) -> None:
        outcomes, frame = _build_comparison_data(
            {"00": 0.75, "01": 0.25},
            {"00": 3, "10": 1},
            4,
            shot_label="Shots (noisy)",
        )

        self.assertEqual(outcomes, ["00", "01", "10"])
        self.assertEqual(frame["Outcome"].tolist(), ["00", "00", "01", "01", "10", "10"])
        self.assertEqual(frame.columns.tolist(), ["Outcome", "Method", "Probability", "Error"])
        self.assertTrue(all(isinstance(value, str) for value in frame["Outcome"]))
        self.assertEqual(
            frame.loc[frame["Outcome"] == "10", "Method"].tolist(),
            ["Exact", "Shots (noisy)"],
        )
        self.assertAlmostEqual(
            frame.loc[
                (frame["Outcome"] == "10") & (frame["Method"] == "Shots (noisy)"),
                "Probability",
            ].iloc[0],
            0.25,
        )

    def test_top_outcome_filter_is_shared_by_chart_and_inspector_data(self) -> None:
        exact = {"00": 0.5, "01": 0.3, "10": 0.2, "11": 0.0}
        counts = {"00": 50, "01": 30, "10": 20}
        _, comparison = _build_comparison_data(exact, counts, 100)
        state = _build_state_inspector_data(
            {key: complex(value**0.5) for key, value in exact.items()}, exact
        )

        visible = _rank_outcomes(exact, counts, 100, limit=2)
        filtered_state, filtered_comparison = _filter_result_frames(state, comparison, visible)

        self.assertEqual(visible, ["00", "01"])
        self.assertEqual(filtered_state["Bitstring"].tolist(), visible)
        self.assertEqual(list(dict.fromkeys(filtered_comparison["Outcome"])), visible)

    def test_noisy_support_keeps_zero_ideal_outcome_in_filtered_comparison(self) -> None:
        exact = {"00": 0.0, "01": 1.0, "10": 0.0}
        counts = {"00": 10, "01": 80, "10": 10}
        noisy = {"00": 0.2, "01": 0.7, "10": 0.1}
        outcomes, comparison = _build_comparison_data(
            exact,
            counts,
            100,
            noisy_probabilities=noisy,
        )

        visible = _rank_outcomes(exact, counts, 100, limit=2, noisy_probabilities=noisy)
        _, filtered_comparison = _filter_result_frames(
            _build_state_inspector_data({"00": 0j, "01": 1 + 0j, "10": 0j}, exact),
            comparison,
            visible,
        )

        self.assertIn("00", visible)
        self.assertEqual(exact["00"], 0.0)
        self.assertIn("00", filtered_comparison["Outcome"].tolist())
        self.assertTrue(set(visible).issubset(outcomes))

    def test_noisy_exact_legend_entry_tracks_available_series(self) -> None:
        exact = {"0": 1.0}
        counts = {"0": 10}
        for qubit_count in (8, 9):
            with self.subTest(qubit_count=qubit_count):
                noisy = {"0": 1.0} if qubit_count <= 8 else None
                _, comparison = _build_comparison_data(
                    exact,
                    counts,
                    10,
                    noisy_probabilities=noisy,
                )
                legend_methods = _comparison_legend_methods(comparison)
                self.assertEqual(
                    "Noisy Exact" in legend_methods,
                    qubit_count <= 8,
                )


class StateInspectorTests(unittest.TestCase):
    def test_inspector_calculates_phase_and_marks_zero_amplitude_undefined(self) -> None:
        frame = _build_state_inspector_data(
            {"00": 1 + 0j, "01": -1j, "10": 0j, "11": 0j},
            {"00": 1.0, "01": 1.0, "10": 0.0, "11": 0.0},
        )

        self.assertEqual(
            frame.columns.tolist(),
            [
                "Bitstring",
                "Real Component",
                "Imaginary Component",
                "Probability",
                "Phase (Radians)",
                "Phase (Degrees)",
            ],
        )
        self.assertAlmostEqual(frame.loc[1, "Phase (Radians)"], -math.pi / 2)
        self.assertAlmostEqual(frame.loc[1, "Phase (Degrees)"], -90.0)
        self.assertTrue(math.isnan(frame.loc[2, "Phase (Radians)"]))
        self.assertTrue(math.isnan(frame.loc[2, "Phase (Degrees)"]))


class CircuitImportTests(unittest.TestCase):
    def _import_error(self, content: str) -> str:
        state = {"circuit_file": Mock(getvalue=lambda: content.encode("utf-8"))}
        with patch.object(app_module.st, "session_state", state):
            app_module._load_uploaded_circuit()
        self.assertEqual(state["import_message"][0], "error")
        return state["import_message"][1]

    def test_import_errors_use_friendly_messages(self) -> None:
        too_many_qubits = self._import_error('{"qubit_count": 11, "gates": []}')
        self.assertIn("at most 10 qubits", too_many_qubits)
        self.assertNotIn("MAX_QUBITS", too_many_qubits)

        for content in ("[]", "null"):
            with self.subTest(content=content):
                self.assertIn("must be a JSON object", self._import_error(content))

        invalid_json = self._import_error("{")
        self.assertNotIn("Expecting value", invalid_json)
        self.assertIn("This file is not valid JSON.", invalid_json)

    def test_import_error_lists_exact_supported_gate_set(self) -> None:
        message = self._import_error(
            '{"qubit_count": 1, "gates": [{"gate": "unsupported", "qubits": [0]}]}'
        )
        supported_list = message.split("Supported gates are: ", maxsplit=1)[1].rstrip(".")
        supported_gates = supported_list.split(", ")
        self.assertEqual(len(supported_gates), 15)
        self.assertEqual(
            set(supported_gates),
            {
                "h",
                "x",
                "y",
                "z",
                "s",
                "t",
                "swap",
                "cx",
                "cz",
                "ccx",
                "rx",
                "ry",
                "rz",
                "p",
                "cp",
            },
        )

    def test_failed_import_leaves_current_editor_untouched(self) -> None:
        old_editor_state = {
            "gate_count": 1,
            "gate_0_type": "h",
            "preset_name": "Bell State",
            "simulation_result": {"existing": True},
        }
        state = {
            **old_editor_state,
            "circuit_file": Mock(
                getvalue=lambda: json.dumps(
                    {
                        "qubit_count": 1,
                        "gates": [{"gate": "x", "qubits": [0]}] * 51,
                    }
                ).encode("utf-8")
            ),
        }

        with patch.object(app_module.st, "session_state", state):
            app_module._load_uploaded_circuit()

        for key, value in old_editor_state.items():
            self.assertEqual(state[key], value)
        self.assertEqual(state["import_message"][0], "error")
        self.assertIn("at most 50 gates", state["import_message"][1])

    def test_oversized_upload_leaves_current_editor_untouched(self) -> None:
        old_editor_state = {
            "gate_count": 1,
            "gate_0_type": "h",
            "preset_name": "Bell State",
            "simulation_result": {"existing": True},
        }
        state = {
            **old_editor_state,
            "circuit_file": Mock(getvalue=lambda: b" " * 1_000_001),
        }

        with patch.object(app_module.st, "session_state", state):
            app_module._load_uploaded_circuit()

        for key, value in old_editor_state.items():
            self.assertEqual(state[key], value)
        self.assertEqual(state["import_message"][0], "error")
        self.assertIn("1 MB or smaller", state["import_message"][1])


class ParameterSweepCacheTests(unittest.TestCase):
    gate_sequence = (
        ("ry", (0,), None, "alpha"),
        ("rx", (0,), None, "beta"),
    )
    swept_parameters = ("alpha",)
    sweep_range = (0.0, math.pi)
    sweep_points = 5
    fixed_values = (("beta", 0.25),)

    def setUp(self) -> None:
        app_module._cached_parameter_sweep.clear()

    def tearDown(self) -> None:
        app_module._cached_parameter_sweep.clear()

    def _run_cached(
        self,
        gate_sequence: tuple[tuple[str, tuple[int, ...], float | None, str | None], ...]
        | None = None,
        swept_parameters: tuple[str, ...] | None = None,
        sweep_range: tuple[float, float] | None = None,
        sweep_points: int | None = None,
    ) -> list[dict[str, object]]:
        return app_module._cached_parameter_sweep(
            1,
            gate_sequence or self.gate_sequence,
            swept_parameters or self.swept_parameters,
            sweep_range or self.sweep_range,
            sweep_points or self.sweep_points,
            self.fixed_values,
        )

    def test_cached_sweep_matches_direct_engine_results(self) -> None:
        values = [
            self.sweep_range[0]
            + (self.sweep_range[1] - self.sweep_range[0]) * index / (self.sweep_points - 1)
            for index in range(self.sweep_points)
        ]
        engine_gates = [
            {"gate": gate, "qubits": list(operands), "angle": parameter_name}
            for gate, operands, _, parameter_name in self.gate_sequence
        ]
        direct = run_parameter_sweep(
            build_circuit(1, engine_gates),
            "alpha",
            values,
            fixed_values=dict(self.fixed_values),
        )

        self.assertEqual(self._run_cached(), direct)

    def test_identical_inputs_compute_sweep_once(self) -> None:
        original = app_module.run_parameter_sweep
        with patch.object(app_module, "run_parameter_sweep", wraps=original) as engine_sweep:
            first = self._run_cached()
            second = self._run_cached()

        self.assertEqual(first, second)
        engine_sweep.assert_called_once()

    def test_sweep_points_gate_and_parameter_changes_invalidate_cache(self) -> None:
        original = app_module.run_parameter_sweep
        changed_gate = (("x", (0,), None, None), *self.gate_sequence)
        changed_parameter = (
            ("ry", (0,), None, "gamma"),
            ("rx", (0,), None, "beta"),
        )
        with patch.object(app_module, "run_parameter_sweep", wraps=original) as engine_sweep:
            self._run_cached()
            self._run_cached(sweep_points=7)
            self._run_cached(gate_sequence=changed_gate)
            self._run_cached(gate_sequence=changed_parameter, swept_parameters=("gamma",))

        self.assertEqual(engine_sweep.call_count, 4)

    def test_shots_and_seed_do_not_invalidate_sweep_cache(self) -> None:
        original = app_module.run_parameter_sweep
        shots = 1024
        seed = 12
        with patch.object(app_module, "run_parameter_sweep", wraps=original) as engine_sweep:
            expected = self._run_cached()
            shots = 100_000
            seed = 987_654_321
            actual = self._run_cached()

        self.assertEqual((shots, seed), (100_000, 987_654_321))
        self.assertEqual(actual, expected)
        engine_sweep.assert_called_once()


class SweepOutcomeSelectionTests(unittest.TestCase):
    def test_ten_qubit_ry_sweep_displays_exactly_two_outcomes_without_changing_data(
        self,
    ) -> None:
        sweep_results = app_module._cached_parameter_sweep(
            10,
            (("ry", (0,), None, "theta"),),
            ("theta",),
            (0.0, 2 * math.pi),
            5,
            (),
        )
        original_results = deepcopy(sweep_results)

        displayed, caption = _select_sweep_outcomes(sweep_results, 10, 50)

        self.assertEqual(displayed, ["0" * 10, "0" * 9 + "1"])
        self.assertEqual(len(displayed), 2)
        self.assertTrue(caption.startswith("Showing 2 of 1024 outcomes"))
        self.assertEqual(sweep_results, original_results)

    def test_sweep_outcome_display_cap_is_respected(self) -> None:
        sweep_results = [
            {"value": 0.0, "probabilities": {"00": 0.5, "01": 0.3, "10": 0.2, "11": 0.0}}
        ]

        displayed, _ = _select_sweep_outcomes(sweep_results, 2, 2)

        self.assertEqual(displayed, ["00", "01"])

    def test_caption_reports_maximum_hidden_probability(self) -> None:
        sweep_results = [
            {"value": 0.0, "probabilities": {"00": 0.5, "01": 0.3, "10": 0.2}},
            {"value": 1.0, "probabilities": {"00": 0.1, "01": 0.2, "10": 0.7}},
        ]

        displayed, caption = _select_sweep_outcomes(sweep_results, 2, 2)

        self.assertEqual(displayed, ["10", "00"])
        self.assertEqual(
            caption,
            "Showing 2 of 4 outcomes (peak probability >= 0.1%). "
            "Hidden outcomes total at most 30.00% at any sweep point",
        )

    def test_sweep_outcome_fallback_selects_highest_peak(self) -> None:
        sweep_results = [
            {"value": 0.0, "probabilities": {"00": 0.0004, "01": 0.0008, "10": 0.0002}}
        ]

        displayed, caption = _select_sweep_outcomes(sweep_results, 2, 4)

        self.assertEqual(displayed, ["01"])
        self.assertTrue(caption.startswith("Showing 1 of 4 outcomes"))


class PresetMathematicsTests(unittest.TestCase):
    def test_qft_matrices_match_analytic_dft_and_qiskit_gate(self) -> None:
        for qubit_count in range(1, 5):
            with self.subTest(qubit_count=qubit_count):
                actual = Operator(build_circuit(qubit_count, build_qft_sequence(qubit_count))).data
                dimension = 2**qubit_count
                expected = np.array(
                    [
                        [
                            np.exp(2j * math.pi * row * column / dimension) / math.sqrt(dimension)
                            for column in range(dimension)
                        ]
                        for row in range(dimension)
                    ]
                )
                pivot = np.unravel_index(np.argmax(np.abs(actual)), actual.shape)
                global_phase = actual[pivot] / expected[pivot]
                global_phase /= abs(global_phase)

                np.testing.assert_allclose(actual, global_phase * expected, atol=1e-9, rtol=0)
                self.assertTrue(Operator(actual).equiv(Operator(QFTGate(qubit_count))))

    def test_three_qubit_qft_state_probabilities_and_phases(self) -> None:
        qft_sequence = build_qft_sequence(3)
        amplitudes, probabilities = get_statevector_data(build_circuit(3, qft_sequence))

        self.assertEqual(len(probabilities), 8)
        for outcome, probability in probabilities.items():
            with self.subTest(outcome=outcome):
                self.assertAlmostEqual(probability, 0.125, delta=1e-9)
        first_amplitude = amplitudes["000"]
        for amplitude in amplitudes.values():
            self.assertAlmostEqual(abs(amplitude - first_amplitude), 0.0, delta=1e-9)

        input_one_gates = [{"gate": "x", "qubits": [0]}, *qft_sequence]
        input_one_amplitudes, _ = get_statevector_data(build_circuit(3, input_one_gates))
        expected = np.array([np.exp(2j * math.pi * index / 8) / math.sqrt(8) for index in range(8)])
        actual = np.array([input_one_amplitudes[f"{index:03b}"] for index in range(8)])
        pivot = int(np.argmax(np.abs(actual)))
        global_phase = actual[pivot] / expected[pivot]
        global_phase /= abs(global_phase)
        np.testing.assert_allclose(actual, global_phase * expected, atol=1e-9, rtol=0)

    def test_grover_success_and_other_outcome_probabilities_for_each_mark(self) -> None:
        for qubit_count in (2, 3):
            dimension = 2**qubit_count
            expected_success = math.sin(3 * math.asin(1 / math.sqrt(dimension))) ** 2
            expected_other = (1 - expected_success) / (dimension - 1)
            for bits in product("01", repeat=qubit_count):
                marked_state = "".join(bits)
                with self.subTest(qubit_count=qubit_count, marked_state=marked_state):
                    gates = build_grover_sequence(qubit_count, marked_state)
                    initial_layer = [
                        {"gate": "h", "qubits": [qubit]} for qubit in range(qubit_count)
                    ]
                    self.assertEqual(gates[:qubit_count], initial_layer)
                    probabilities = get_exact_probabilities(build_circuit(qubit_count, gates))
                    self.assertAlmostEqual(
                        probabilities[marked_state], expected_success, delta=1e-9
                    )
                    self.assertEqual(len(probabilities), dimension)
                    for outcome, probability in probabilities.items():
                        if outcome != marked_state:
                            self.assertAlmostEqual(probability, expected_other, delta=1e-9)
                    self.assertEqual(max(probabilities, key=probabilities.get), marked_state)

    def test_grover_builder_rejects_four_qubits(self) -> None:
        for marked_state in ("0000", "1111"):
            with self.subTest(marked_state=marked_state):
                with self.assertRaises(ValueError):
                    build_grover_sequence(4, marked_state)

    def test_grover_app_preset_marked_state_is_most_probable(self) -> None:
        preset_qubits, preset_gates = app_module._PRESETS["Grover (marked 111)"]

        probabilities = get_exact_probabilities(build_circuit(preset_qubits, preset_gates))

        self.assertEqual(max(probabilities, key=probabilities.get), "111")


if __name__ == "__main__":
    unittest.main()
