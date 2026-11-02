import matplotlib

matplotlib.use("Agg", force=True)

import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import app as app_module  # noqa: E402
from quantum_engine import (  # noqa: E402
    MAX_GATES,
    MAX_QUBITS,
    MAX_SHOTS,
    build_circuit,
    build_grover_sequence,
    build_noise_model,
    build_qft_sequence,
    calculate_shot_standard_errors,
    draw_circuit_mpl,
    get_exact_probabilities,
    get_noisy_probabilities,
    get_statevector_data,
    import_circuit_json,
    run_parameter_sweep,
    run_shot_experiment,
    validate_circuit_parameters,
)

try:
    import psutil
except ImportError:
    psutil = None


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
STRESS_ENABLED = os.environ.get("RUN_STRESS") == "1"
_GATES = ("h", "x", "rx", "ry", "rz", "cp", "cx", "cz", "swap", "ccx")
_PRESET_LABELS = tuple(name for name in app_module._PRESETS if name != "Custom circuit")


def _random_gate_sequence(qubit_count: int, gate_count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    gates: list[dict[str, Any]] = []
    for index in range(gate_count):
        gate = _GATES[index] if index < len(_GATES) else rng.choice(_GATES)
        arity = {"cp": 2, "cx": 2, "cz": 2, "swap": 2, "ccx": 3}.get(gate, 1)
        qubits = rng.sample(range(qubit_count), arity)
        spec: dict[str, Any] = {"gate": gate, "qubits": qubits}
        if gate in ("rx", "ry", "rz", "cp"):
            spec["angle"] = rng.uniform(-math.pi, math.pi)
        gates.append(spec)
    return gates


def _maximum_circuit(seed: int = 764_351, qubit_count: int = MAX_QUBITS):
    return build_circuit(qubit_count, _random_gate_sequence(qubit_count, MAX_GATES, seed))


class StressMetricsMixin:
    def setUp(self) -> None:
        self._stress_started = time.perf_counter()
        self._stress_process = psutil.Process() if psutil is not None else None
        self._stress_rss_before = (
            self._stress_process.memory_info().rss if self._stress_process is not None else None
        )

    def tearDown(self) -> None:
        elapsed = time.perf_counter() - self._stress_started
        if self._stress_process is None:
            rss_delta = "unavailable (psutil not installed)"
        else:
            delta = self._stress_process.memory_info().rss - self._stress_rss_before
            rss_delta = f"{delta / (1024**2):+.2f} MiB"
        print(
            f"STRESS_METRIC {self.id()} elapsed={elapsed:.3f}s rss_delta={rss_delta}",
            flush=True,
        )

    def assert_rss_growth_below(self, limit_bytes: int) -> None:
        if self._stress_process is None:
            return
        rss_after = self._stress_process.memory_info().rss
        growth = max(0, rss_after - self._stress_rss_before)
        self.assertLess(growth, limit_bytes)


@unittest.skipUnless(STRESS_ENABLED, "Set RUN_STRESS=1 to run stress tests.")
class EngineStressTests(StressMetricsMixin, unittest.TestCase):
    def test_maximum_circuit_statevector(self) -> None:
        circuit = _maximum_circuit()
        started = time.perf_counter()
        _, probabilities = get_statevector_data(circuit)
        elapsed = time.perf_counter() - started

        self.assertAlmostEqual(sum(probabilities.values()), 1.0, delta=1e-9)
        self.assertLess(elapsed, 10.0)

    def test_maximum_circuit_ideal_shots(self) -> None:
        circuit = _maximum_circuit()
        started = time.perf_counter()
        counts = run_shot_experiment(circuit, 100_000, seed_simulator=834_593)
        elapsed = time.perf_counter() - started

        self.assertEqual(sum(counts.values()), 100_000)
        self.assertLess(elapsed, 15.0)

    def test_maximum_circuit_noisy_shots(self) -> None:
        circuit = _maximum_circuit()
        noise_model = build_noise_model(0.2, 0.2, 0.2, 0.2, 0.2)
        started = time.perf_counter()
        counts = run_shot_experiment(
            circuit,
            100_000,
            seed_simulator=29_101,
            noise_model=noise_model,
        )
        elapsed = time.perf_counter() - started

        self.assertEqual(sum(counts.values()), 100_000)
        self.assertLess(elapsed, 30.0)
        self.assert_rss_growth_below(1_500 * 1024**2)

    def test_maximum_size_exact_noisy_probabilities_and_qubit_limit(self) -> None:
        circuit = build_circuit(8, _random_gate_sequence(8, MAX_GATES, 182_037))
        noise_model = build_noise_model(0.2, 0.2, 0.2, 0.2, 0.2)
        started = time.perf_counter()
        probabilities = get_noisy_probabilities(circuit, noise_model, 0.2)
        elapsed = time.perf_counter() - started

        self.assertAlmostEqual(sum(probabilities.values()), 1.0, delta=1e-9)
        self.assertLess(elapsed, 30.0)
        with self.assertRaisesRegex(ValueError, "limited to 8 qubits"):
            get_noisy_probabilities(
                build_circuit(9, []),
                noise_model,
                0.2,
            )

    def test_validation_boundaries(self) -> None:
        for qubit_count, gates in (
            (10, [{"gate": "h", "qubits": [0]}] * 51),
            (11, []),
            (0, []),
        ):
            for operation in (validate_circuit_parameters, build_circuit):
                with self.subTest(qubit_count=qubit_count, operation=operation.__name__):
                    with self.assertRaises(ValueError):
                        operation(qubit_count, gates)

        valid_circuit = build_circuit(10, _random_gate_sequence(10, 50, 18_937))
        with self.assertRaises(ValueError):
            run_shot_experiment(valid_circuit, 100_001)

        for probability in (0.2000001, math.nan, math.inf, True, "0.1"):
            with self.subTest(probability=probability):
                with self.assertRaises(ValueError):
                    build_noise_model(bit_flip_probability=probability)

        validate_circuit_parameters(10, _random_gate_sequence(10, 50, 18_937))
        self.assertEqual(len(build_circuit(10, _random_gate_sequence(10, 50, 18_937)).data), 50)
        self.assertEqual(
            sum(run_shot_experiment(valid_circuit, MAX_SHOTS, seed_simulator=4).values()),
            100_000,
        )
        self.assertIsNotNone(build_noise_model(bit_flip_probability=0.2))
        self.assertIsNone(build_noise_model(bit_flip_probability=0.0))

    def test_maximum_noise_model_build_time(self) -> None:
        started = time.perf_counter()
        noise_model = build_noise_model(0.2, 0.2, 0.2, 0.2, 0.2)

        self.assertIsNotNone(noise_model)
        self.assertLess(time.perf_counter() - started, 3.0)

    def test_random_circuits_match_ideal_shot_statistics(self) -> None:
        for circuit_index in range(25):
            rng = random.Random(490_001 + circuit_index)
            qubit_count = rng.randint(3, 5)
            gate_count = rng.randint(10, 30)
            circuit = build_circuit(
                qubit_count,
                _random_gate_sequence(qubit_count, gate_count, 730_000 + circuit_index),
            )
            exact = get_exact_probabilities(circuit)
            counts = run_shot_experiment(
                circuit,
                20_000,
                seed_simulator=910_000 + circuit_index,
            )
            observed_counts = {outcome: counts.get(outcome, 0) for outcome in exact}
            standard_errors = calculate_shot_standard_errors(observed_counts, 20_000)
            for outcome, probability in exact.items():
                with self.subTest(circuit=circuit_index, outcome=outcome):
                    count = counts.get(outcome, 0)
                    if probability < 1e-12:
                        self.assertEqual(count, 0)
                    frequency = count / 20_000
                    self.assertLessEqual(
                        abs(frequency - probability),
                        5 * standard_errors[outcome] + 0.002,
                    )

    def test_random_noisy_circuits_match_noisy_shot_samples(self) -> None:
        noise_model = build_noise_model(0.05, 0.05, 0.05, 0.05, 0.05)
        for circuit_index in range(3):
            circuit = build_circuit(
                4,
                _random_gate_sequence(4, 20, 521_000 + circuit_index),
            )
            exact_noisy = get_noisy_probabilities(circuit, noise_model, 0.05)
            counts = run_shot_experiment(
                circuit,
                100_000,
                seed_simulator=601_000 + circuit_index,
                noise_model=noise_model,
            )
            maximum_difference = max(
                abs(exact_noisy[outcome] - counts.get(outcome, 0) / 100_000)
                for outcome in exact_noisy
            )
            self.assertLess(maximum_difference, 0.01)

    def test_simulator_seed_determinism(self) -> None:
        circuit = build_circuit(1, [{"gate": "h", "qubits": [0]}])
        first = run_shot_experiment(circuit, 10_000, seed_simulator=104_729)
        repeated = run_shot_experiment(circuit, 10_000, seed_simulator=104_729)
        different = run_shot_experiment(circuit, 10_000, seed_simulator=104_723)

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, different)

    def test_validation_fuzz_rejects_only_with_value_error(self) -> None:
        rng = random.Random(47_299)
        malformed_specs: list[Any] = []
        gates = list(_GATES) + ["", None, 4, [], {}]
        qubit_values: list[Any] = [
            -1,
            0.5,
            True,
            None,
            "0",
            [],
            10**1000,
            0,
            1,
            2,
        ]
        for _ in range(500):
            gate = rng.choice(gates)
            spec: Any
            if rng.randrange(5) == 0:
                spec = rng.choice([None, True, 5, "h", [], [1, 2]])
            else:
                spec = {
                    "gate": gate,
                    "qubits": [rng.choice(qubit_values) for _ in range(rng.randrange(5))],
                }
                if rng.randrange(2):
                    spec["extra"] = rng.choice([None, "extra", {}, []])
                if gate in ("rx", "ry", "rz", "cp"):
                    spec["angle"] = rng.choice(
                        [math.nan, math.inf, True, "not a parameter!", 10**1000, 0.2]
                    )
                if rng.randrange(4) == 0:
                    spec.pop("qubits", None)
            malformed_specs.append(spec)

        for index, spec in enumerate(malformed_specs):
            for operation in (validate_circuit_parameters, build_circuit):
                try:
                    operation(5, [spec])
                except ValueError:
                    continue
                except Exception as error:
                    self.fail(
                        f"Fuzz case {index} raised {type(error).__name__}: {error}; spec={spec!r}"
                    )

        json_values: list[Any] = [
            None,
            True,
            0,
            "circuit",
            [],
            {},
            {"qubit_count": 1, "gates": []},
            {"qubit_count": 1, "gates": [{"gate": "h", "qubits": [0]}]},
        ]
        for _ in range(500):
            value = rng.choice(json_values)
            encoded = json.dumps(value)
            try:
                import_circuit_json(encoded)
            except ValueError:
                continue
            except Exception as error:
                self.fail(f"JSON fuzz raised {type(error).__name__}: {error}; value={value!r}")

        malformed_json = [
            "{",
            '{"qubit_count": 1, "gates": [}',
            '{"qubit_count": NaN, "gates": []}',
            "[" * 500 + "0" + "]" * 500,
        ]
        for encoded in malformed_json:
            try:
                import_circuit_json(encoded)
            except ValueError:
                continue
            except Exception as error:
                self.fail(f"Malformed JSON raised {type(error).__name__}: {error}")

    def test_sequence_builders_accept_or_raise_value_error(self) -> None:
        for qubit_count in range(1, 11):
            try:
                sequence = build_qft_sequence(qubit_count)
                validate_circuit_parameters(qubit_count, sequence)
            except ValueError:
                pass
            try:
                grover_sequence = build_grover_sequence(qubit_count, "1" * qubit_count)
                validate_circuit_parameters(qubit_count, grover_sequence)
            except ValueError:
                pass

    def test_ten_qubit_parameter_sweep(self) -> None:
        circuit = build_circuit(
            10,
            [
                {"gate": "ry", "qubits": [0], "angle": "alpha"},
                {"gate": "rx", "qubits": [9], "angle": "beta"},
                {"gate": "cx", "qubits": [0, 1]},
            ],
        )
        values = np.linspace(-math.pi, math.pi, 101).tolist()
        started = time.perf_counter()
        results = run_parameter_sweep(circuit, "alpha", values, {"beta": 0.25})

        self.assertEqual(len(results), 101)
        self.assertLess(time.perf_counter() - started, 30.0)

    def test_circuit_drawing_and_shot_repetition_memory(self) -> None:
        circuit = build_circuit(
            2,
            [{"gate": "h", "qubits": [0]}, {"gate": "cx", "qubits": [0, 1]}],
        )
        import matplotlib.pyplot as plt

        for _ in range(100):
            figure = draw_circuit_mpl(circuit)
            plt.close(figure)
        for index in range(50):
            run_shot_experiment(circuit, 100, seed_simulator=77_000 + index)
        self.assert_rss_growth_below(200 * 1024**2)


@unittest.skipUnless(STRESS_ENABLED, "Set RUN_STRESS=1 to run stress tests.")
class StreamlitStressTests(StressMetricsMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.app = AppTest.from_file(str(APP_PATH)).run(timeout=60)
        self.assertEqual(len(self.app.exception), 0)

    def _configure_mixed_noise_circuit(self, qubit_count: int) -> list[dict[str, Any]]:
        gate_cycle = ("h", "x", "ry", "rz", "cx", "cz", "cp", "swap", "ccx")
        gate_sequence: list[dict[str, Any]] = []
        self.app.slider(key="qubit_count").set_value(qubit_count)
        self.app.number_input(key="gate_count").set_value(50)
        self.app.number_input(key="shots").set_value(100_000)
        self.app.session_state["use_noise"] = True
        self.app.run(timeout=60)

        for index in range(50):
            gate = gate_cycle[index % len(gate_cycle)]
            self.app.session_state[f"gate_{index}_type"] = gate
        self.app.run(timeout=60)

        for index in range(50):
            gate = gate_cycle[index % len(gate_cycle)]
            first_qubit = index % qubit_count
            arity = {"cx": 2, "cz": 2, "cp": 2, "swap": 2, "ccx": 3}.get(gate, 1)
            operands = [((first_qubit + offset) % qubit_count) for offset in range(arity)]
            gate_spec: dict[str, Any] = {"gate": gate, "qubits": operands}
            if arity == 1:
                self.app.session_state[f"gate_{index}_qubit"] = operands[0]
            else:
                for operand_index, qubit in enumerate(operands):
                    self.app.session_state[f"gate_{index}_operand_{operand_index}"] = qubit
            if gate in ("ry", "rz", "cp"):
                angle = 0.25 + 0.01 * index
                self.app.session_state[f"gate_{index}_angle"] = angle
                gate_spec["angle"] = angle
            gate_sequence.append(gate_spec)

        for key in (
            "bit_flip_probability",
            "phase_flip_probability",
            "depolarizing_probability",
            "amplitude_damping_probability",
            "readout_error_probability",
        ):
            self.app.session_state[key] = 0.2
        self.app.run(timeout=60)
        self.assertEqual(len(self.app.exception), 0)
        return gate_sequence

    def test_random_action_fuzzer(self) -> None:
        for seed in range(5):
            app = AppTest.from_file(str(APP_PATH)).run(timeout=60)
            rng = random.Random(8_118_001 + seed)
            action_log: list[str] = []
            try:
                for step in range(150):
                    action = rng.randrange(10)
                    if action == 0:
                        value = rng.randint(1, MAX_QUBITS)
                        action_log.append(f"qubit_count={value}")
                        app.slider(key="qubit_count").set_value(value).run(timeout=60)
                    elif action == 1:
                        value = rng.randint(0, MAX_GATES)
                        action_log.append(f"gate_count={value}")
                        app.number_input(key="gate_count").set_value(value).run(timeout=60)
                    elif action == 2:
                        gate_count = int(app.session_state["gate_count"])
                        if gate_count == 0:
                            app.number_input(key="gate_count").set_value(1).run(timeout=60)
                            gate_count = 1
                        index = rng.randrange(gate_count)
                        gate_widget = app.selectbox(key=f"gate_{index}_type")
                        gate = rng.choice(gate_widget.options)
                        action_log.append(f"gate_{index}_type={gate}")
                        gate_widget.select(gate).run(timeout=60)
                    elif action == 3:
                        gate_count = int(app.session_state["gate_count"])
                        if gate_count == 0:
                            app.number_input(key="gate_count").set_value(1).run(timeout=60)
                            gate_count = 1
                        index = rng.randrange(gate_count)
                        gate = app.selectbox(key=f"gate_{index}_type").value
                        arity = {"cp": 2, "cx": 2, "cz": 2, "swap": 2, "ccx": 3}.get(gate, 1)
                        operand_index = rng.randrange(arity)
                        key = (
                            f"gate_{index}_qubit"
                            if arity == 1
                            else f"gate_{index}_operand_{operand_index}"
                        )
                        value = rng.randrange(int(app.session_state["qubit_count"]))
                        action_log.append(f"{key}={value}")
                        app.selectbox(key=key).set_value(value).run(timeout=60)
                    elif action == 4:
                        value = not app.toggle(key="use_noise").value
                        action_log.append(f"use_noise={value}")
                        app.toggle(key="use_noise").set_value(value).run(timeout=60)
                    elif action == 5:
                        preset = rng.choice(_PRESET_LABELS)
                        action_log.append(f"preset={preset}")
                        app.selectbox(key="preset_name").select(preset).run(timeout=60)
                    elif action == 6:
                        action_log.append("clear_all_gates")
                        app.button(key="clear_all_gates_button").click().run(timeout=60)
                    elif action == 7:
                        value = rng.randint(1, MAX_SHOTS)
                        action_log.append(f"shots={value}")
                        app.number_input(key="shots").set_value(value).run(timeout=60)
                    elif action == 8:
                        qubit_count = int(app.session_state["qubit_count"])
                        value = rng.randint(1, min(50, 2**qubit_count))
                        action_log.append(f"max_display_outcomes={value}")
                        app.number_input(key="max_display_outcomes").set_value(value).run(
                            timeout=60
                        )
                    else:
                        run_button = app.button(key="run_simulation_button")
                        if run_button.disabled:
                            action_log.append("run_simulation skipped (disabled)")
                        else:
                            action_log.append("run_simulation")
                            run_button.click().run(timeout=60)
                    self.assertEqual(
                        len(app.exception),
                        0,
                        f"seed={seed}, step={step}, action_log={action_log!r}",
                    )
            except Exception as error:
                self.fail(
                    f"Random-action fuzzer failed for seed {seed}: "
                    f"{type(error).__name__}: {error}\nFull action log: {action_log!r}"
                )

    def test_maximum_configuration_runs_through_ui(self) -> None:
        self._configure_mixed_noise_circuit(10)

        started = time.perf_counter()
        self.app.button(key="run_simulation_button").click().run(timeout=60)

        self.assertEqual(len(self.app.exception), 0)
        self.assertLess(time.perf_counter() - started, 60.0)
        result = self.app.session_state["simulation_result"]
        self.assertEqual(result["qubit_count"], 10)
        self.assertEqual(result["shots"], 100_000)
        self.assertEqual(len(result["gate_sequence"]), 50)
        gate_names = {gate["gate"] for gate in result["gate_sequence"]}
        self.assertTrue({"cx", "ccx", "swap", "cp"}.issubset(gate_names))
        self.assertIsNone(result["noisy_probabilities"])
        self.assertTrue(result["noise_enabled"])
        for key in (
            "bit_flip_probability",
            "phase_flip_probability",
            "depolarizing_probability",
            "amplitude_damping_probability",
            "readout_error_probability",
        ):
            self.assertEqual(result[key], 0.2)

    def test_eight_qubit_maximum_configuration_has_exact_noisy_probabilities(self) -> None:
        self._configure_mixed_noise_circuit(8)

        started = time.perf_counter()
        self.app.button(key="run_simulation_button").click().run(timeout=60)
        elapsed = time.perf_counter() - started

        self.assertEqual(len(self.app.exception), 0)
        self.assertLess(elapsed, 60.0)
        result = self.app.session_state["simulation_result"]
        self.assertEqual(result["qubit_count"], 8)
        self.assertEqual(result["shots"], 100_000)
        self.assertEqual(len(result["gate_sequence"]), 50)
        gate_names = {gate["gate"] for gate in result["gate_sequence"]}
        self.assertTrue({"cx", "ccx", "swap", "cp"}.issubset(gate_names))
        self.assertTrue(result["noise_enabled"])
        self.assertIsNotNone(result["noisy_probabilities"])
        self.assertAlmostEqual(sum(result["noisy_probabilities"].values()), 1.0, delta=1e-9)

    def test_ten_qubit_named_parameters_render_variational_tab(self) -> None:
        self.app.slider(key="qubit_count").set_value(10).run(timeout=60)
        self.app.selectbox(key="gate_0_type").select("ry").run(timeout=60)
        self.app.checkbox(key="gate_0_parameterized").set_value(True).run(timeout=60)
        self.app.text_input(key="gate_0_parameter_name").set_value("alpha").run(timeout=60)
        self.app.selectbox(key="gate_1_type").select("rx").run(timeout=60)
        self.app.checkbox(key="gate_1_parameterized").set_value(True).run(timeout=60)
        self.app.text_input(key="gate_1_parameter_name").set_value("beta").run(timeout=60)
        self.assertEqual(len(self.app.exception), 0)
        self.assertIn("parameter_value_alpha", self.app.session_state)
        self.assertIn("parameter_value_beta", self.app.session_state)
        self.assertGreaterEqual(len(self.app.get("vega_lite_chart")), 1)

    def test_gate_step_slider_visits_every_prefix(self) -> None:
        self.app.selectbox(key="preset_name").select("Bell State").run(timeout=60)
        self.app.button(key="run_simulation_button").click().run(timeout=60)
        step_count = len(self.app.session_state["simulation_result"]["prefix_steps"])

        for step in range(step_count):
            self.app.slider(key="gate_step").set_value(step).run(timeout=60)
            self.assertEqual(len(self.app.exception), 0, f"Failed at gate_step={step}")


if __name__ == "__main__":
    unittest.main()
