import matplotlib

matplotlib.use("Agg", force=True)

import json
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest.mock import patch  # noqa: E402

from streamlit.testing.v1 import AppTest  # noqa: E402

import quantum_engine  # noqa: E402
from app import _PRESETS  # noqa: E402
from quantum_engine import build_circuit, get_exact_probabilities  # noqa: E402

APP_PATH = next(
    path
    for path in (
        Path(__file__).resolve().with_name("app.py"),
        Path(__file__).resolve().parents[1] / "app.py",
    )
    if path.is_file()
)


class StreamlitAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = AppTest.from_file(str(APP_PATH)).run(timeout=60)
        self.assertEqual(len(self.app.exception), 0)

    def test_qft_preset_has_no_widget_default_warnings(self) -> None:
        self.app.selectbox(key="preset_name").select("QFT (3 qubits)").run(timeout=60)

        self.assertEqual(len(self.app.warning), 0)
        self.assertEqual(len(self.app.exception), 0)

    def test_maximum_displayed_outcomes_tracks_preset_and_qubit_count(self) -> None:
        self.app.selectbox(key="preset_name").select("QFT (3 qubits)").run(timeout=60)

        self.assertEqual(self.app.number_input(key="max_display_outcomes").value, 8)

        self.app.slider(key="qubit_count").set_value(4).run(timeout=60)

        self.assertEqual(self.app.number_input(key="max_display_outcomes").value, 16)
        self.assertEqual(len(self.app.exception), 0)

    def test_preset_loads_editably_and_clear_all_resets_editor(self) -> None:
        self.app.selectbox(key="preset_name").select("Bell State").run(timeout=30)

        self.assertEqual(self.app.slider(key="qubit_count").value, 2)
        self.assertEqual(self.app.number_input(key="gate_count").value, 2)
        self.assertEqual(self.app.selectbox(key="gate_1_type").value, "cx")
        self.assertEqual(self.app.selectbox(key="gate_1_operand_0").value, 0)
        self.assertEqual(self.app.selectbox(key="gate_1_operand_1").value, 1)

        self.app.selectbox(key="gate_0_type").select("x").run(timeout=30)
        self.assertEqual(self.app.selectbox(key="preset_name").value, "Custom circuit")
        self.assertEqual(self.app.selectbox(key="gate_0_type").value, "x")

        self.app.button(key="clear_all_gates_button").click().run(timeout=60)
        self.assertEqual(self.app.number_input(key="gate_count").value, 0)
        self.assertEqual(self.app.selectbox(key="preset_name").value, "Custom circuit")
        self.assertEqual(len(self.app.exception), 0)

    def test_custom_qft_custom_keeps_qft_gates_and_reports_current_messages(self) -> None:
        self.assertEqual(self.app.selectbox(key="preset_name").value, "Custom circuit")
        self.app.selectbox(key="gate_0_type").select("x").run(timeout=60)
        self.app.selectbox(key="gate_1_type").select("z").run(timeout=60)

        self.app.selectbox(key="preset_name").select("QFT (3 qubits)").run(timeout=60)
        self.assertIn(
            "Selecting a preset replaces your custom gates.",
            [element.value for element in self.app.info],
        )
        self.app.selectbox(key="preset_name").select("Custom circuit").run(timeout=60)

        expected_gates = [gate["gate"] for gate in _PRESETS["QFT (3 qubits)"][1]]
        final_gate_count = int(self.app.number_input(key="gate_count").value)
        final_gates = [
            self.app.selectbox(key=f"gate_{index}_type").value for index in range(final_gate_count)
        ]
        self.assertEqual(final_gates, expected_gates)
        self.assertNotEqual(final_gates, ["x", "z"])
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(len(self.app.warning), 0)
        self.assertEqual(
            [element.value for element in self.app.info],
            [
                "Build or load a circuit, then run it from the workspace tab.",
                "Run a circuit to inspect its ideal statevector.",
                "Enable “Use named parameter” on a rotation/phase gate to "
                "create a sweepable circuit parameter.",
            ],
        )
        self.assertEqual(
            [element.value for element in self.app.caption],
            [
                "Explore common circuits, inspect ideal quantum states, and compare "
                "ideal probabilities with sampled results.",
                "OpenQASM stores the current parameter value; JSON keeps parameter names.",
                "Gate order is preserved. Named rotation parameters can be swept "
                "in the Variational Lab.",
            ],
        )

    def test_identity_run_renders_chart_and_state_table(self) -> None:
        self.app.number_input(key="gate_count").set_value(0)
        self.app.button(key="run_simulation_button").click().run(timeout=60)

        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(len(self.app.error), 0)
        self.assertGreaterEqual(len(self.app.get("image")), 2)
        self.assertEqual(len(self.app.get("vega_lite_chart")), 2)
        self.assertEqual(len(self.app.get("dataframe")), 3)
        self.assertTrue(any("covering" in item.value for item in self.app.caption))
        chart_spec = json.loads(self.app.get("vega_lite_chart")[0].proto.spec)
        self.assertEqual(chart_spec["encoding"]["x"]["type"], "nominal")
        self.assertEqual(chart_spec["encoding"]["x"]["sort"], ["00", "01", "10", "11"])
        self.assertEqual(chart_spec["encoding"]["xOffset"]["field"], "Method")
        self.app.number_input(key="shots").set_value(2048).run(timeout=60)
        self.assertEqual(len(self.app.get("vega_lite_chart")), 0)

    def test_noise_toggle_runs_and_labels_comparison_as_noisy(self) -> None:
        self.app.selectbox(key="preset_name").select("Quantum Coin Toss").run(timeout=30)
        self.app.toggle(key="use_noise").set_value(True).run(timeout=30)
        self.app.button(key="run_simulation_button").click().run(timeout=60)

        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(len(self.app.error), 0)
        self.assertEqual(len(self.app.get("vega_lite_chart")), 2)
        self.assertTrue(self.app.session_state["simulation_result"]["noise_enabled"])
        self.assertIn(
            "Shots (noisy)",
            self.app.session_state["simulation_result"]["comparison_data"]["Method"].tolist(),
        )

    def test_large_noisy_circuit_skips_exact_noisy_distribution(self) -> None:
        self.app.slider(key="qubit_count").set_value(9).run(timeout=60)
        self.app.toggle(key="use_noise").set_value(True).run(timeout=60)
        self.app.button(key="run_simulation_button").click().run(timeout=60)

        result = self.app.session_state["simulation_result"]
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(len(self.app.error), 0)
        self.assertTrue(result["noise_enabled"])
        self.assertIsNone(result["noisy_probabilities"])
        self.assertNotIn("Noisy Exact", result["comparison_data"]["Method"].tolist())
        self.assertTrue(
            any(
                element.value == "Exact noisy probabilities are limited to 8 qubits. "
                "Showing ideal probabilities and sampled noisy shot frequencies only."
                for element in self.app.info
            )
        )

    def test_noisy_exact_limit_message_only_appears_above_eight_qubits_after_run(self) -> None:
        expected_message = (
            "Exact noisy probabilities are limited to 8 qubits. Showing ideal probabilities "
            "and sampled noisy shot frequencies only."
        )
        self.app.slider(key="qubit_count").set_value(10).run(timeout=60)
        self.app.toggle(key="use_noise").set_value(True).run(timeout=60)
        self.app.button(key="run_simulation_button").click().run(timeout=60)

        messages = [element.value for element in self.app.info]
        self.assertIn(expected_message, messages)
        self.assertTrue(all("running" not in message.lower() for message in messages))
        self.assertEqual(len(self.app.exception), 0)

        self.app.slider(key="qubit_count").set_value(8).run(timeout=60)
        self.app.button(key="run_simulation_button").click().run(timeout=60)

        messages = [element.value for element in self.app.info]
        self.assertFalse(
            any("Exact noisy probabilities are limited" in message for message in messages)
        )
        self.assertEqual(len(self.app.exception), 0)

    def test_all_presets_are_valid_circuits(self) -> None:
        expected_qubits = {
            "Bell State": 2,
            "GHZ State": 3,
            "Quantum Coin Toss": 1,
            "3-Qubit Uniform Superposition": 3,
            "QFT (3 qubits)": 3,
            "Grover (marked 111)": 3,
            "Deutsch–Jozsa (balanced)": 3,
            "Deutsch–Jozsa (constant)": 3,
            "Bernstein–Vazirani (secret 01)": 3,
        }
        for name, qubit_count in expected_qubits.items():
            with self.subTest(preset=name):
                preset = _PRESETS[name]
                self.assertIsNotNone(preset)
                preset_qubits, gates = preset
                self.assertEqual(preset_qubits, qubit_count)
                probabilities = get_exact_probabilities(build_circuit(preset_qubits, gates))
                self.assertAlmostEqual(sum(probabilities.values()), 1.0)

    def test_algorithm_presets_have_distinct_expected_measurement_outcomes(self) -> None:
        expected_outcomes = {
            "Deutsch–Jozsa (balanced)": "111",
            "Deutsch–Jozsa (constant)": "100",
            "Bernstein–Vazirani (secret 01)": "101",
        }
        for name, expected_outcome in expected_outcomes.items():
            with self.subTest(preset=name):
                qubit_count, gates = _PRESETS[name]
                probabilities = get_exact_probabilities(build_circuit(qubit_count, gates))
                dominant_outcome = max(probabilities, key=probabilities.get)
                self.assertEqual(dominant_outcome, expected_outcome)
                self.assertAlmostEqual(probabilities[expected_outcome], 1.0)

        self.assertNotEqual(
            _PRESETS["Deutsch–Jozsa (balanced)"][1],
            _PRESETS["Bernstein–Vazirani (secret 01)"][1],
        )

    def test_invalid_cnot_operands_are_shown_in_ui(self) -> None:
        self.app.selectbox(key="preset_name").select("Bell State").run(timeout=30)
        self.app.selectbox(key="gate_1_operand_1").set_value(0)
        self.app.button(key="run_simulation_button").click().run(timeout=60)

        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(len(self.app.warning), 1)
        self.assertIn("same qubit", self.app.warning[0].value)

    def test_qubit_count_shrink_keeps_available_gate_operands_distinct(self) -> None:
        self.app.slider(key="qubit_count").set_value(3).run(timeout=60)
        self.app.selectbox(key="gate_1_type").select("cx").run(timeout=60)
        self.app.selectbox(key="gate_1_operand_0").set_value(0).run(timeout=60)
        self.app.selectbox(key="gate_1_operand_1").set_value(2).run(timeout=60)

        self.app.slider(key="qubit_count").set_value(2).run(timeout=60)

        self.assertEqual(self.app.selectbox(key="gate_1_operand_0").value, 0)
        self.assertEqual(self.app.selectbox(key="gate_1_operand_1").value, 1)
        self.assertEqual(len(self.app.exception), 0)

    def test_cx_qubit_count_shrink_to_one_keeps_operands_valid(self) -> None:
        self.app.selectbox(key="preset_name").select("Bell State").run(timeout=60)
        self.app.slider(key="qubit_count").set_value(1).run(timeout=60)

        self.assertEqual(self.app.selectbox(key="gate_1_operand_0").value, 0)
        self.assertEqual(self.app.selectbox(key="gate_1_operand_1").value, 0)
        self.assertTrue(
            any(
                "gate no longer fits the current qubit count" in element.value
                for element in self.app.warning
            )
        )
        self.assertEqual(len(self.app.exception), 0)

    def test_qubit_count_shrink_notifies_when_gates_are_moved(self) -> None:
        self.app.slider(key="qubit_count").set_value(3).run(timeout=60)
        self.app.selectbox(key="gate_1_type").select("cx").run(timeout=60)
        self.app.selectbox(key="gate_1_operand_0").set_value(0).run(timeout=60)
        self.app.selectbox(key="gate_1_operand_1").set_value(2).run(timeout=60)

        self.app.slider(key="qubit_count").set_value(2).run(timeout=60)

        self.assertIn(
            "Gates were moved to fit the new qubit count.",
            [element.value for element in self.app.info],
        )
        self.assertEqual(len(self.app.exception), 0)

    def test_ccx_qubit_count_shrink_keeps_operands_valid(self) -> None:
        self.app.slider(key="qubit_count").set_value(3).run(timeout=60)
        self.app.selectbox(key="gate_0_type").select("ccx").run(timeout=60)
        for operand_index, qubit in enumerate((0, 1, 2)):
            self.app.selectbox(key=f"gate_0_operand_{operand_index}").set_value(qubit).run(
                timeout=60
            )

        self.app.slider(key="qubit_count").set_value(2).run(timeout=60)

        self.assertEqual(self.app.selectbox(key="gate_0_operand_0").value, 0)
        self.assertEqual(self.app.selectbox(key="gate_0_operand_1").value, 1)
        self.assertEqual(self.app.selectbox(key="gate_0_operand_2").value, 0)
        self.assertEqual(len(self.app.exception), 0)

    def test_unexpected_simulator_errors_are_friendly_in_the_ui(self) -> None:
        with patch.object(
            quantum_engine,
            "AerSimulator",
            side_effect=RuntimeError("test-only simulator failure"),
        ):
            self.app.button(key="run_simulation_button").click().run(timeout=60)

        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(len(self.app.error), 1)
        self.assertIn("could not be completed", self.app.error[0].value)

    def test_step_through_updates_intermediate_statevector(self) -> None:
        self.app.selectbox(key="preset_name").select("Bell State").run(timeout=60)
        self.app.button(key="run_simulation_button").click().run(timeout=60)
        self.app.slider(key="gate_step").set_value(1).run(timeout=60)

        prefix_state = self.app.session_state["simulation_result"]["prefix_steps"][1]["state_data"]
        probabilities = dict(
            zip(prefix_state["Bitstring"], prefix_state["Probability"], strict=False)
        )
        self.assertAlmostEqual(probabilities["00"], 0.5)
        self.assertAlmostEqual(probabilities["01"], 0.5)
        self.assertEqual(len(self.app.exception), 0)

    def test_variational_tab_renders_circuit_with_two_named_parameters(self) -> None:
        self.app.selectbox(key="gate_0_type").select("ry").run(timeout=60)
        self.app.checkbox(key="gate_0_parameterized").set_value(True).run(timeout=60)
        self.app.text_input(key="gate_0_parameter_name").set_value("a").run(timeout=60)
        self.app.selectbox(key="gate_1_type").select("rx").run(timeout=60)
        self.app.checkbox(key="gate_1_parameterized").set_value(True).run(timeout=60)
        self.app.text_input(key="gate_1_parameter_name").set_value("b").run(timeout=60)

        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(len(self.app.error), 0)
        self.assertIn("parameter_value_a", self.app.session_state)
        self.assertIn("parameter_value_b", self.app.session_state)
        self.assertGreaterEqual(len(self.app.get("vega_lite_chart")), 1)


if __name__ == "__main__":
    unittest.main()
