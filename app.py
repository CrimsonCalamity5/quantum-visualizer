"""Streamlit interface for the Quantum Circuit Visualizer."""

from __future__ import annotations

import faulthandler
import logging
import math
from io import BytesIO
from typing import Any

import altair as alt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quantum_engine import (
    MAX_GATES,
    MAX_NOISY_EXACT_QUBITS,
    MAX_QUBITS,
    MAX_SHOTS,
    bind_circuit_parameters,
    build_circuit,
    build_grover_sequence,
    build_noise_model,
    build_prefix_circuit,
    build_qft_sequence,
    calculate_shot_standard_errors,
    draw_circuit_mpl,
    draw_circuit_text,
    export_circuit_json,
    export_circuit_qasm,
    get_noisy_probabilities,
    get_qubit_state_metrics,
    get_statevector_data,
    import_circuit_json,
    run_parameter_sweep,
    run_shot_experiment,
    run_toy_vqe,
    validate_circuit_parameters,
)

logger = logging.getLogger(__name__)
_MAX_CIRCUIT_UPLOAD_BYTES = 1_000_000

_GATES = (
    "h",
    "x",
    "y",
    "z",
    "s",
    "t",
    "p",
    "rx",
    "ry",
    "rz",
    "cx",
    "cz",
    "cp",
    "swap",
    "ccx",
)
_ANGLE_GATES = frozenset({"p", "cp", "rx", "ry", "rz"})
_PRESETS: dict[str, tuple[int, list[dict[str, object]]] | None] = {
    "Custom circuit": None,
    "Bell State": (
        2,
        [{"gate": "h", "qubits": [0]}, {"gate": "cx", "qubits": [0, 1]}],
    ),
    "GHZ State": (
        3,
        [
            {"gate": "h", "qubits": [0]},
            {"gate": "cx", "qubits": [0, 1]},
            {"gate": "cx", "qubits": [1, 2]},
        ],
    ),
    "Quantum Coin Toss": (1, [{"gate": "h", "qubits": [0]}]),
    "3-Qubit Uniform Superposition": (
        3,
        [
            {"gate": "h", "qubits": [0]},
            {"gate": "h", "qubits": [1]},
            {"gate": "h", "qubits": [2]},
        ],
    ),
    "QFT (3 qubits)": (3, build_qft_sequence(3)),
    "Grover (marked 111)": (3, build_grover_sequence(3, "111")),
    "Deutsch–Jozsa (balanced)": (
        3,
        [
            {"gate": "x", "qubits": [2]},
            *[{"gate": "h", "qubits": [q]} for q in range(3)],
            {"gate": "cx", "qubits": [0, 2]},
            {"gate": "cx", "qubits": [1, 2]},
            {"gate": "h", "qubits": [0]},
            {"gate": "h", "qubits": [1]},
            {"gate": "h", "qubits": [2]},
        ],
    ),
    "Deutsch–Jozsa (constant)": (
        3,
        [
            {"gate": "x", "qubits": [2]},
            *[{"gate": "h", "qubits": [q]} for q in range(3)],
            {"gate": "h", "qubits": [0]},
            {"gate": "h", "qubits": [1]},
            {"gate": "h", "qubits": [2]},
        ],
    ),
    "Bernstein–Vazirani (secret 01)": (
        3,
        [
            {"gate": "x", "qubits": [2]},
            *[{"gate": "h", "qubits": [q]} for q in range(3)],
            {"gate": "cx", "qubits": [0, 2]},
            {"gate": "h", "qubits": [0]},
            {"gate": "h", "qubits": [1]},
            {"gate": "h", "qubits": [2]},
        ],
    ),
}


def _clear_gate_widget_state() -> None:
    for index in range(MAX_GATES):
        for field in (
            "type",
            "control",
            "target",
            "qubit",
            "angle",
            "parameterized",
            "parameter_name",
            "operand_0",
            "operand_1",
            "operand_2",
        ):
            st.session_state.pop(f"gate_{index}_{field}", None)


def _set_gate_sequence_in_state(qubit_count: int, gates: list[dict[str, object]]) -> None:
    _clear_gate_widget_state()
    st.session_state["qubit_count"] = qubit_count
    st.session_state["max_display_outcomes"] = min(50, 2**qubit_count)
    st.session_state["gate_count"] = len(gates)
    for index, gate_spec in enumerate(gates):
        gate = str(gate_spec["gate"])
        qubits = gate_spec["qubits"]
        st.session_state[f"gate_{index}_type"] = gate
        if gate in ("cx", "cz", "cp", "swap"):
            st.session_state[f"gate_{index}_operand_0"] = qubits[0]
            st.session_state[f"gate_{index}_operand_1"] = qubits[1]
        elif gate == "ccx":
            for operand_index, qubit in enumerate(qubits):
                st.session_state[f"gate_{index}_operand_{operand_index}"] = qubit
        else:
            st.session_state[f"gate_{index}_qubit"] = qubits[0]
        if gate in _ANGLE_GATES:
            angle = gate_spec["angle"]
            if isinstance(angle, str):
                st.session_state[f"gate_{index}_parameterized"] = True
                st.session_state[f"gate_{index}_parameter_name"] = angle
            else:
                st.session_state[f"gate_{index}_parameterized"] = False
                st.session_state[f"gate_{index}_angle"] = angle


def _load_selected_preset() -> None:
    st.session_state.pop("simulation_result", None)
    preset = _PRESETS[st.session_state["preset_name"]]
    if preset is None:
        st.session_state["_has_custom_gates"] = True
        return

    if st.session_state.get("_has_custom_gates", False):
        st.session_state["preset_notice"] = "Selecting a preset replaces your custom gates."
    qubit_count, gates = preset
    _set_gate_sequence_in_state(qubit_count, gates)
    st.session_state["_has_custom_gates"] = False


def _load_uploaded_circuit() -> None:
    uploaded_file = st.session_state.get("circuit_file")
    if uploaded_file is None:
        st.session_state["import_message"] = ("error", "Choose a JSON file first.")
        return
    try:
        raw_content = uploaded_file.getvalue()
        if len(raw_content) > _MAX_CIRCUIT_UPLOAD_BYTES:
            raise ValueError("Circuit JSON uploads must be 1 MB or smaller.")
        content = raw_content.decode("utf-8")
        qubit_count, gates = import_circuit_json(content)
    except (UnicodeDecodeError, ValueError) as error:
        st.session_state["import_message"] = (
            "error",
            f"Could not import circuit: {error}",
        )
        return
    _set_gate_sequence_in_state(qubit_count, gates)
    st.session_state["preset_name"] = "Custom circuit"
    st.session_state.pop("simulation_result", None)
    st.session_state["_has_custom_gates"] = True
    st.session_state["import_message"] = ("success", "Circuit imported and validated.")


def _mark_preset_custom() -> None:
    st.session_state.pop("simulation_result", None)
    st.session_state["_has_custom_gates"] = True
    if "preset_name" in st.session_state:
        st.session_state["preset_name"] = "Custom circuit"


def _invalidate_simulation() -> None:
    st.session_state.pop("simulation_result", None)


def _handle_qubit_count_change() -> None:
    _mark_preset_custom()
    qubit_count = int(st.session_state["qubit_count"])
    st.session_state["max_display_outcomes"] = min(50, 2**qubit_count)
    valid_qubits = range(qubit_count)
    moved_gates = False
    for index in range(MAX_GATES):
        gate = st.session_state.get(f"gate_{index}_type")
        arity = {"cx": 2, "cz": 2, "cp": 2, "swap": 2, "ccx": 3}.get(gate, 1)
        if arity == 1:
            key = f"gate_{index}_qubit"
            if key in st.session_state and st.session_state[key] not in valid_qubits:
                st.session_state[key] = 0
                moved_gates = True
            continue

        used_qubits: set[int] = set()
        for operand_index in range(arity):
            key = f"gate_{index}_operand_{operand_index}"
            if key not in st.session_state:
                continue
            qubit = st.session_state[key]
            if qubit not in valid_qubits or qubit in used_qubits:
                available_qubits = set(valid_qubits) - used_qubits
                if available_qubits:
                    qubit = min(available_qubits)
                    st.session_state[key] = qubit
                    moved_gates = True
                elif qubit not in valid_qubits:
                    qubit = 0
                    st.session_state[key] = qubit
                    moved_gates = True
            used_qubits.add(qubit)
    if moved_gates:
        st.session_state["qubit_count_notice"] = "Gates were moved to fit the new qubit count."


def _clear_all_gates() -> None:
    _clear_gate_widget_state()
    st.session_state["gate_count"] = 0
    st.session_state["preset_name"] = "Custom circuit"
    st.session_state["_has_custom_gates"] = False
    st.session_state.pop("simulation_result", None)


def _build_state_inspector_data(
    amplitudes: dict[str, complex], probabilities: dict[str, float]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bitstring, amplitude in amplitudes.items():
        real = float(amplitude.real)
        imaginary = float(amplitude.imag)
        if math.isclose(abs(amplitude), 0.0, abs_tol=1e-12):
            phase_radians = None
            phase_degrees = None
        else:
            phase_radians = math.atan2(imaginary, real)
            phase_degrees = math.degrees(phase_radians)
        rows.append(
            {
                "Bitstring": bitstring,
                "Real Component": real,
                "Imaginary Component": imaginary,
                "Probability": probabilities[bitstring],
                "Phase (Radians)": phase_radians,
                "Phase (Degrees)": phase_degrees,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "Bitstring",
            "Real Component",
            "Imaginary Component",
            "Probability",
            "Phase (Radians)",
            "Phase (Degrees)",
        ],
    )


@st.cache_data(show_spinner=False)
def _cached_circuit_png(qasm_key: str, _circuit: Any) -> bytes:
    """Cache circuit PNG bytes by its stable OpenQASM representation."""
    figure = draw_circuit_mpl(_circuit)
    import matplotlib.pyplot as plt

    buffer = BytesIO()
    try:
        figure.savefig(buffer, format="png", bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(figure)
        buffer.close()


def _show_circuit(circuit: Any) -> None:
    """Show a circuit diagram; adding ?plain=1 to the URL uses the text drawer."""
    if st.query_params.get("plain") == "1":
        st.code(draw_circuit_text(circuit), language=None)
    else:
        st.image(_cached_circuit_png(export_circuit_qasm(circuit), circuit))


@st.cache_data(show_spinner=False, max_entries=8)
def _cached_parameter_sweep(
    qubit_count: int,
    gate_sequence: tuple[tuple[str, tuple[int, ...], float | None, str | None], ...],
    swept_parameter_names: tuple[str, ...],
    sweep_range: tuple[float, float],
    sweep_points: int,
    fixed_parameter_values: tuple[tuple[str, float], ...],
) -> list[dict[str, object]]:
    """Cache exact sweep results using only stable, hashable circuit inputs."""
    if len(swept_parameter_names) != 1:
        raise ValueError("Exactly one sweep parameter must be selected.")
    engine_gate_sequence: list[dict[str, object]] = []
    for gate, operands, angle, parameter_name in gate_sequence:
        gate_spec: dict[str, object] = {"gate": gate, "qubits": list(operands)}
        if parameter_name is not None:
            gate_spec["angle"] = parameter_name
        elif angle is not None:
            gate_spec["angle"] = angle
        engine_gate_sequence.append(gate_spec)

    sweep_start, sweep_end = sweep_range
    values = [
        sweep_start + (sweep_end - sweep_start) * index / (sweep_points - 1)
        for index in range(sweep_points)
    ]
    return run_parameter_sweep(
        build_circuit(qubit_count, engine_gate_sequence),
        swept_parameter_names[0],
        values,
        fixed_values=dict(fixed_parameter_values),
    )


def _select_sweep_outcomes(
    sweep_results: list[dict[str, object]], qubit_count: int, display_limit: int
) -> tuple[list[str], str]:
    """Choose sweep outcomes by peak support and summarize hidden probability."""
    outcome_probabilities = [result["probabilities"] for result in sweep_results]
    peak_probabilities: dict[str, float] = {}
    for probabilities in outcome_probabilities:
        for outcome, probability in probabilities.items():
            peak_probabilities[outcome] = max(
                peak_probabilities.get(outcome, 0.0),
                float(probability),
            )

    ranked_outcomes = sorted(
        peak_probabilities,
        key=lambda outcome: (-peak_probabilities[outcome], outcome),
    )
    displayed_outcomes = [
        outcome for outcome in ranked_outcomes if peak_probabilities[outcome] >= 0.001
    ][:display_limit]
    if not displayed_outcomes and ranked_outcomes:
        displayed_outcomes = ranked_outcomes[:1]

    displayed_set = set(displayed_outcomes)
    hidden_probability = max(
        (
            sum(
                float(probability)
                for outcome, probability in probabilities.items()
                if outcome not in displayed_set
            )
            for probabilities in outcome_probabilities
        ),
        default=0.0,
    )
    caption = (
        f"Showing {len(displayed_outcomes)} of {2**qubit_count} outcomes "
        "(peak probability >= 0.1%). Hidden outcomes total at most "
        f"{hidden_probability:.2%} at any sweep point"
    )
    return displayed_outcomes, caption


def _build_comparison_data(
    exact_probabilities: dict[str, float],
    shot_counts: dict[str, int],
    shots: int,
    shot_label: str = "Shots",
    noisy_probabilities: dict[str, float] | None = None,
    shot_errors: dict[str, float] | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """Align exact probabilities and observed shot frequencies by bitstring."""
    outcomes = list(dict.fromkeys([*exact_probabilities, *shot_counts]))
    rows: list[dict[str, Any]] = []
    for outcome in outcomes:
        rows.append(
            {
                "Outcome": outcome,
                "Method": "Exact",
                "Probability": exact_probabilities.get(outcome, 0.0),
                "Error": 0.0,
            }
        )
        if noisy_probabilities is not None:
            rows.append(
                {
                    "Outcome": outcome,
                    "Method": "Noisy Exact",
                    "Probability": noisy_probabilities.get(outcome, 0.0),
                    "Error": 0.0,
                }
            )
        rows.append(
            {
                "Outcome": outcome,
                "Method": shot_label,
                "Probability": shot_counts.get(outcome, 0) / shots,
                "Error": (shot_errors or {}).get(outcome, 0.0),
            }
        )
    comparison_data = pd.DataFrame(rows, columns=["Outcome", "Method", "Probability", "Error"])
    comparison_data["Outcome"] = comparison_data["Outcome"].astype(str)
    return outcomes, comparison_data


def _comparison_legend_methods(comparison_data: pd.DataFrame) -> list[str]:
    method_order = ["Exact", "Noisy Exact", "Shots", "Shots (noisy)"]
    present_methods = set(comparison_data["Method"])
    return [method for method in method_order if method in present_methods]


def _rank_outcomes(
    exact_probabilities: dict[str, float],
    shot_counts: dict[str, int],
    shots: int,
    limit: int,
    noisy_probabilities: dict[str, float] | None = None,
) -> list[str]:
    """Rank the union by strongest ideal, noisy-exact, or observed support."""
    outcomes = set(exact_probabilities) | set(shot_counts)
    if noisy_probabilities is not None:
        outcomes |= set(noisy_probabilities)

    def score(outcome: str) -> tuple[float, str]:
        values = [
            exact_probabilities.get(outcome, 0.0),
            shot_counts.get(outcome, 0) / shots,
        ]
        if noisy_probabilities is not None:
            values.append(noisy_probabilities.get(outcome, 0.0))
        return -max(values), outcome

    return sorted(outcomes, key=score)[:limit]


def _filter_result_frames(
    state_data: pd.DataFrame,
    comparison_data: pd.DataFrame,
    outcomes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = set(outcomes)
    return (
        state_data.loc[state_data["Bitstring"].isin(selected)].reset_index(drop=True),
        comparison_data.loc[comparison_data["Outcome"].isin(selected)].reset_index(drop=True),
    )


def _render_gate_editor(qubit_count: int, gate_count: int) -> list[dict[str, object]]:
    """Render ordered gate controls and return engine-compatible specifications."""
    gate_sequence: list[dict[str, object]] = []
    qubit_indices = list(range(qubit_count))

    for index in range(gate_count):
        with st.expander(f"Gate {index + 1}", expanded=(index == 0)):
            gate = st.selectbox(
                "Gate type",
                options=_GATES,
                key=f"gate_{index}_type",
                on_change=_mark_preset_custom,
            )
            arity = {"cx": 2, "cz": 2, "cp": 2, "swap": 2, "ccx": 3}.get(gate, 1)
            if arity > 1:
                columns = st.columns(arity)
                operands = [
                    columns[operand_index].selectbox(
                        (
                            "Control qubit"
                            if gate in ("cx", "ccx", "cp") and operand_index == 0
                            else (
                                "Target qubit"
                                if gate in ("cx", "ccx", "cp") and operand_index == arity - 1
                                else f"Qubit {operand_index + 1}"
                            )
                        ),
                        options=qubit_indices,
                        key=f"gate_{index}_operand_{operand_index}",
                        on_change=_mark_preset_custom,
                    )
                    for operand_index in range(arity)
                ]
                gate_spec = {"gate": gate, "qubits": operands}
            else:
                qubit = st.selectbox(
                    "Qubit",
                    options=qubit_indices,
                    key=f"gate_{index}_qubit",
                    on_change=_mark_preset_custom,
                )
                gate_spec = {"gate": gate, "qubits": [qubit]}
            if gate in _ANGLE_GATES:
                parameterized = st.checkbox(
                    "Use named parameter",
                    key=f"gate_{index}_parameterized",
                    on_change=_mark_preset_custom,
                )
                if parameterized:
                    parameter_name_key = f"gate_{index}_parameter_name"
                    st.session_state.setdefault(parameter_name_key, f"theta_{index}")
                    gate_spec["angle"] = st.text_input(
                        "Parameter name",
                        key=parameter_name_key,
                        on_change=_mark_preset_custom,
                    )
                else:
                    angle_key = f"gate_{index}_angle"
                    st.session_state.setdefault(angle_key, 0.0)
                    gate_spec["angle"] = st.number_input(
                        "Angle (radians)",
                        step=0.1,
                        format="%.4f",
                        key=angle_key,
                        on_change=_mark_preset_custom,
                    )
            gate_sequence.append(gate_spec)

    return gate_sequence


def _render_probability_chart(outcomes: list[str], comparison_data: pd.DataFrame) -> None:
    legend_methods = _comparison_legend_methods(comparison_data)
    method_colors = {
        "Exact": "#8ecae6",
        "Noisy Exact": "#219ebc",
        "Shots": "#ffb703",
        "Shots (noisy)": "#fb8500",
    }
    chart = (
        alt.Chart(comparison_data)
        .mark_bar()
        .encode(
            x=alt.X(
                "Outcome:N",
                sort=alt.Sort(outcomes),
                axis=alt.Axis(labelExpr="datum.value"),
                title="Outcome (q[n-1] ... q1 q0)",
            ),
            xOffset=alt.XOffset("Method:N"),
            y=alt.Y(
                "Probability:Q",
                scale=alt.Scale(zero=True),
                title="Probability",
            ),
            color=alt.Color(
                "Method:N",
                title="Method",
                scale=alt.Scale(
                    domain=legend_methods,
                    range=[method_colors[method] for method in legend_methods],
                ),
            ),
            tooltip=[
                alt.Tooltip("Outcome:N", title="Outcome"),
                alt.Tooltip("Method:N", title="Method"),
                alt.Tooltip("Probability:Q", format=".4f"),
            ],
        )
        .properties(height=420)
    )
    shot_rows = comparison_data.loc[
        comparison_data["Method"].str.startswith("Shots") & (comparison_data["Error"] > 0)
    ]
    if not shot_rows.empty:
        chart = chart + (
            alt.Chart(shot_rows)
            .mark_errorbar(ticks=True)
            .encode(
                x=alt.X("Outcome:N", sort=alt.Sort(outcomes)),
                xOffset=alt.XOffset("Method:N"),
                y=alt.Y("Probability:Q", scale=alt.Scale(zero=True)),
                yError=alt.YError("Error:Q"),
                color=alt.Color("Method:N"),
            )
        )
    st.altair_chart(chart, width="stretch")


def _render_state_inspector(state_data: pd.DataFrame) -> None:
    numeric_columns = [
        "Real Component",
        "Imaginary Component",
        "Probability",
        "Phase (Radians)",
        "Phase (Degrees)",
    ]
    st.dataframe(
        state_data.style.format(
            {column: "{:.4f}" for column in numeric_columns},
            na_rep="—",
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "This is the ideal pre-noise pure state. Phase is undefined for "
        "zero-amplitude basis states and depends on the circuit's global phase."
    )


def _render_bloch_sphere(metrics: list[dict[str, float | int]]) -> None:
    latitude = [math.pi * index / 20 for index in range(21)]
    longitude = [2 * math.pi * index / 40 for index in range(41)]
    sphere_x = [[math.sin(phi) * math.cos(theta) for theta in longitude] for phi in latitude]
    sphere_y = [[math.sin(phi) * math.sin(theta) for theta in longitude] for phi in latitude]
    sphere_z = [[math.cos(phi) for _ in longitude] for phi in latitude]
    figure = go.Figure()
    figure.add_trace(
        go.Surface(
            x=sphere_x,
            y=sphere_y,
            z=sphere_z,
            opacity=0.15,
            showscale=False,
            colorscale=[[0, "#90caf9"], [1, "#90caf9"]],
            hoverinfo="skip",
        )
    )
    colors = ["#d62828", "#2a9d8f", "#6a4c93", "#f4a261", "#264653"]
    for metric in metrics:
        qubit = int(metric["qubit"])
        x, y, z = (float(metric[key]) for key in ("x", "y", "z"))
        figure.add_trace(
            go.Scatter3d(
                x=[0, x],
                y=[0, y],
                z=[0, z],
                mode="lines+markers",
                name=f"q{qubit}",
                line={"width": 6, "color": colors[qubit % len(colors)]},
                marker={"size": [2, 5]},
                hovertemplate=(
                    f"q{qubit}<br>Bloch vector=({x:.3f}, {y:.3f}, {z:.3f})<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        height=440,
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        scene={
            "xaxis": {"range": [-1, 1], "title": "X"},
            "yaxis": {"range": [-1, 1], "title": "Y"},
            "zaxis": {"range": [-1, 1], "title": "Z"},
            "aspectmode": "cube",
        },
        legend={"orientation": "h"},
    )
    st.plotly_chart(figure, width="stretch")


def _render_phase_chart(state_data: pd.DataFrame) -> None:
    valid_phase = state_data.dropna(subset=["Phase (Degrees)"])
    if valid_phase.empty:
        st.info("No nonzero amplitudes have a defined phase to display.")
        return
    chart = (
        alt.Chart(valid_phase)
        .mark_circle(size=100)
        .encode(
            x=alt.X("Real Component:Q", title="Real amplitude"),
            y=alt.Y("Imaginary Component:Q", title="Imaginary amplitude"),
            color=alt.Color(
                "Phase (Degrees):Q",
                scale=alt.Scale(scheme="rainbow", domain=[-180, 180]),
                title="Phase (degrees)",
            ),
            tooltip=[
                alt.Tooltip("Bitstring:N"),
                alt.Tooltip("Real Component:Q", format=".4f"),
                alt.Tooltip("Imaginary Component:Q", format=".4f"),
                alt.Tooltip("Phase (Degrees):Q", format=".2f"),
            ],
        )
        .properties(height=340)
    )
    st.altair_chart(chart, width="stretch")


def _calculate_selected_parameter_values(circuit: Any) -> dict[str, float]:
    values = {}
    for parameter in circuit.parameters:
        base_key = f"parameter_value_{parameter.name}"
        slider_key = f"{base_key}_slider"
        input_key = f"{base_key}_input"

        # Initialize default value in session state if it doesn't exist
        if base_key not in st.session_state:
            st.session_state[base_key] = 0.0

        # Ensure individual widget states stay aligned with the core value
        if slider_key not in st.session_state:
            st.session_state[slider_key] = st.session_state[base_key]
        if input_key not in st.session_state:
            st.session_state[input_key] = st.session_state[base_key]

        # Syncing functions that run the instant you move or type a value
        def sync_from_slider(bk=base_key, sk=slider_key, ik=input_key):
            st.session_state[bk] = st.session_state[sk]
            st.session_state[ik] = st.session_state[sk]
            _invalidate_simulation()

        def sync_from_input(bk=base_key, sk=slider_key, ik=input_key):
            st.session_state[bk] = st.session_state[ik]
            st.session_state[sk] = st.session_state[ik]
            _invalidate_simulation()

        # Render side-by-side using columns (3:1 size ratio looks cleanest)
        col1, col2 = st.columns([3, 1])

        with col1:
            st.slider(
                label=f"{parameter.name} (radians)",
                min_value=-6.28,
                max_value=6.28,
                step=0.01,
                format="%.2f",
                key=slider_key,
                on_change=sync_from_slider,
            )

        with col2:
            st.number_input(
                label="Type value",
                min_value=-6.28,
                max_value=6.28,
                step=0.01,
                format="%.2f",
                label_visibility="collapsed",
                key=input_key,
                on_change=sync_from_input,
            )

        values[parameter.name] = st.session_state[base_key]

    return values


def main() -> None:
    faulthandler.enable()
    st.set_page_config(
        page_title="Quantum Circuit Visualizer",
        page_icon="⚛️",
        layout="wide",
    )
    st.session_state.setdefault("qubit_count", min(2, MAX_QUBITS))
    st.session_state.setdefault("gate_count", 2)
    st.title("Quantum Circuit Visualizer")
    st.caption(
        "Explore common circuits, inspect ideal quantum states, and compare "
        "ideal probabilities with sampled results."
    )
    with st.expander("👋 New to Quantum? Read this 60-Second Beginner's Guide!", expanded=True):
        st.markdown("### ⚛️ What is this app?")
        st.write(
            "In theory, quantum computers use physics to solve impossible problems. "
            "In reality, nature is messy. "
            "Everyday heat and interference create **Noise**, which causes mistakes. "
            "This app lets you design quantum programs and see exactly how "
            "real-world errors corrupt perfect math."
        )

        # Single tab group for the beginner's guide
        tab_guide_ws, tab_guide_comp, tab_guide_inspect, tab_guide_lab = st.tabs(
            [
                "🎨 The Workspace & Gates",
                "📊 Simulation & Noise",
                "🔍 State Inspector",
                "🧪 Variational Lab",
            ]
        )

        with tab_guide_ws:
            st.markdown("#### **The Workspace: Writing your Quantum Recipe**")
            st.write(
                "This is your drafting canvas. You give step-by-step instructions "
                "(**Gates**) to your information units (**Qubits**)."
            )

            st.markdown("##### **⚡ Core Concepts**")
            st.markdown("""
            * **Presets Menu:** Auto-loads famous programs like the **Bell State** 
            or the **Quantum Fourier Transform (QFT)**.
            * **Superposition:** Puts a qubit into a fluid state where it acts as both 0 
            and 1 at the same time (like a spinning coin before it lands).
            * **Entanglement:** Digitally links qubits together so that whatever happens 
            to one instantly changes the other.
            """)

            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.markdown("""
                * **`h` (Hadamard):** Creates **Superposition** (spins your coin).
                * **`x` (NOT / Bit-Flip):** Flips a qubit completely (0 becomes 1, 1 becomes 0).
                * **`y` & `z`:** Advanced flips that alter the mathematical wave direction 
                (**Phase**).
                * **`s` & `t`:** Small, fixed phase nudges (quarter and eighth turns).
                * **`rx`, `ry`, `rz`, `p`:** Custom rotation angles. Use the slider or type 
                numbers to smoothly twist a qubit in any direction.
                """)

            with col_g2:
                st.markdown("""
                * **`cx` (CNOT):** Flips a target qubit *only if* a control qubit is 1. 
                This is your primary **Entanglement** tool!
                * **`cz` & `cp`:** Controlled phase gates that alter wave directions only 
                when connected qubits are active.
                * **`swap`:** Physically trades the exact states of two qubits.
                * **`ccx` (Toffoli):** Flips a target qubit *only if two* control qubits 
                are active (works like a quantum 'AND' gate).
                """)

        with tab_guide_comp:
            st.markdown("#### **Simulation Comparison: Testing Three Worlds at Once**")
            st.write(
                "When you run your program, the app tracks three different mathematical "
                "models simultaneously to isolate errors:"
            )
            st.markdown("""
            * **Exact View:** Pure probability math under perfect, zero-error laboratory conditions.
            * **Noisy Exact:** Simulates actual hardware faults like **Bit/Phase-Flips** 
              (accidental data flipping), **Depolarizing Noise** (raw data scrambling), 
              **Amplitude Damping** (energy escaping), and **Readout Errors** (faulty tools).
            * **Finite Shots:** Simulates practical hardware execution by running your code 
              repeatedly (**Shots**) to sample statistical roughness over fixed runs 
              (like 1,024 runs).
            """)

        with tab_guide_inspect:
            st.markdown("#### **State Inspector: Looking Under the Hood**")
            st.write(
                "Quantum states are invisible. This tab pulls back the curtain to run "
                "deep diagnostic medical checks on your code's health:"
            )
            st.markdown("""
            * **Statevector Matrix:** Shows the raw mathematical coordinates and arrows 
              directing your active data.
            * **Per-Qubit Reduced States:** Isolates individual qubits from the group to check 
              if they are behaving correctly on their own lines.
            * **State Purity:** A health score from 0.0 to 1.0. A perfect ideal state scores 
              a `1.0`. As environmental noise infects your system, this score drops.
            * **von Neumann Entropy:** Measures the structural chaos of your quantum 
              information. Perfect states have `0` entropy. High entropy means your 
              useful program has decayed into useless heat.
            """)

        with tab_guide_lab:
            st.markdown("#### **Variational Lab: Advanced Hybrid Engineering**")
            st.write(
                "Modern quantum computing uses a regular 'classical' computer to tightly "
                "monitor, correct, and tune a noisy quantum chip:"
            )
            st.markdown(r"""
            * **Parameter Sweep:** Instead of testing just one angle, this feature automatically 
              sweeps a gate's rotation across a full landscape ($0$ to $2\pi$). It stitches 
              the results into a continuous wave line graph.
            * **Toy VQE Lab (Variational Quantum Eigensolver):** Runs a smart
              machine-learning-style 
              optimization loop. It demonstrates how software actively tunes gate paths to 
            find 
              the lowest possible energy state of a molecule.
            """)

    with st.sidebar:
        st.header("Circuit and simulation")
        st.selectbox(
            "Circuit preset",
            options=list(_PRESETS),
            key="preset_name",
            on_change=_load_selected_preset,
            help=(
                "Presets load editable gate sequences. The three-qubit H-gate "
                "preset prepares a uniform superposition, not a QFT. Algorithm "
                "presets state their oracle/secret in the name."
            ),
        )
        preset_notice = st.session_state.pop("preset_notice", None)
        if preset_notice is not None:
            st.info(preset_notice)
        uploaded_file = st.file_uploader(
            "Import circuit JSON",
            type=["json"],
            key="circuit_file",
        )
        st.button(
            "Load JSON circuit",
            key="load_json_button",
            on_click=_load_uploaded_circuit,
            disabled=uploaded_file is None,
        )
        import_message = st.session_state.pop("import_message", None)
        if import_message is not None:
            if import_message[0] == "error":
                st.error(import_message[1])
            else:
                st.success(import_message[1])

        qubit_count = st.slider(
            "Number of qubits",
            min_value=1,
            max_value=MAX_QUBITS,
            key="qubit_count",
            on_change=_handle_qubit_count_change,
            help=f"Exact statevector simulation is limited to {MAX_QUBITS} qubits.",
        )
        qubit_count_notice = st.session_state.pop("qubit_count_notice", None)
        if qubit_count_notice is not None:
            st.info(qubit_count_notice)
        shots = st.number_input(
            "Shot count",
            min_value=1,
            max_value=MAX_SHOTS,
            value=1024,
            step=1,
            key="shots",
            on_change=_invalidate_simulation,
        )
        use_seed = st.checkbox(
            "Set simulator seed",
            value=False,
            key="use_seed",
            on_change=_invalidate_simulation,
        )
        seed_simulator: int | None = None
        if use_seed:
            seed_simulator = st.number_input(
                "Simulator seed",
                min_value=0,
                value=0,
                step=1,
                key="seed_simulator",
                on_change=_invalidate_simulation,
            )

        state_count = 2**qubit_count
        st.session_state["max_display_outcomes"] = min(
            st.session_state.get("max_display_outcomes", min(50, state_count)),
            state_count,
        )
        max_display_outcomes = st.number_input(
            "Maximum displayed outcomes",
            min_value=1,
            max_value=state_count,
            step=1,
            key="max_display_outcomes",
            help="The same top outcomes are shown in the chart and state table.",
        )

        st.divider()
        use_noise = st.toggle(
            "Simulate gate/readout noise",
            value=False,
            key="use_noise",
            on_change=_invalidate_simulation,
        )
        bit_flip_probability = 0.0
        phase_flip_probability = 0.0
        depolarizing_probability = 0.0
        amplitude_damping_probability = 0.0
        readout_error_probability = 0.0
        if use_noise:
            st.caption(
                "Pauli/gate channels act after supported gates; idle-time noise "
                f"is not modeled. Exact noisy probabilities are limited to "
                f"{MAX_NOISY_EXACT_QUBITS} qubits; noisy shots still run above "
                "that limit."
            )
            bit_flip_probability = st.slider(
                "Bit-Flip Error Probability",
                min_value=0.0,
                max_value=0.2,
                value=0.01,
                step=0.01,
                key="bit_flip_probability",
                on_change=_invalidate_simulation,
            )
            phase_flip_probability = st.slider(
                "Phase-Flip Error Probability",
                min_value=0.0,
                max_value=0.2,
                value=0.01,
                step=0.01,
                key="phase_flip_probability",
                on_change=_invalidate_simulation,
            )
            depolarizing_probability = st.slider(
                "Depolarizing Error Probability",
                min_value=0.0,
                max_value=0.2,
                value=0.0,
                step=0.01,
                key="depolarizing_probability",
                on_change=_invalidate_simulation,
            )
            amplitude_damping_probability = st.slider(
                "Amplitude Damping Probability",
                min_value=0.0,
                max_value=0.2,
                value=0.0,
                step=0.01,
                key="amplitude_damping_probability",
                on_change=_invalidate_simulation,
            )
            readout_error_probability = st.slider(
                "Readout Error Probability",
                min_value=0.0,
                max_value=0.2,
                value=0.0,
                step=0.01,
                key="readout_error_probability",
                on_change=_invalidate_simulation,
            )

    workspace_tab, comparison_tab, inspector_tab, variational_tab = st.tabs(
        [
            "Workspace & Circuit",
            "Simulation Comparison",
            "Quantum State Inspector",
            "Variational Lab",
        ]
    )

    with workspace_tab:
        st.subheader("Ordered gate sequence")
        st.button(
            "Clear All Gates",
            key="clear_all_gates_button",
            on_click=_clear_all_gates,
            help="Remove all gates and switch to a custom empty circuit.",
        )
        gate_count_state = int(st.session_state["gate_count"])
        st.session_state["gate_step"] = min(
            st.session_state.get("gate_step", gate_count_state), gate_count_state
        )
        gate_count = st.number_input(
            "Number of gates",
            min_value=0,
            max_value=MAX_GATES,
            step=1,
            key="gate_count",
            on_change=_mark_preset_custom,
            help="Set to 0 to simulate the identity circuit.",
        )
        gate_sequence = _render_gate_editor(qubit_count, int(gate_count))
        try:
            validate_circuit_parameters(qubit_count, gate_sequence)

            parameterized_circuit = build_circuit(qubit_count, gate_sequence)
            parameter_values = _calculate_selected_parameter_values(parameterized_circuit)
            preview_circuit = bind_circuit_parameters(parameterized_circuit, parameter_values)
            st.subheader("Live circuit preview")
            _show_circuit(preview_circuit)
            col_json, col_qasm = st.columns(2)
            col_json.download_button(
                "Download circuit JSON",
                data=export_circuit_json(qubit_count, gate_sequence),
                file_name="quantum_circuit.json",
                mime="application/json",
                key="download_json_button",
            )
            col_qasm.download_button(
                "Download OpenQASM 3",
                data=export_circuit_qasm(preview_circuit),
                file_name="quantum_circuit.qasm",
                mime="text/plain",
                key="download_qasm_button",
            )
            col_qasm.caption(
                "OpenQASM stores the current parameter value; JSON keeps parameter names."
            )
        except ValueError as error:
            parameterized_circuit = None
            parameter_values = {}
            preview_circuit = None
            unfit_gate = next(
                (
                    str(gate_spec["gate"])
                    for gate_spec in gate_sequence
                    if {"cx": 2, "cz": 2, "cp": 2, "swap": 2, "ccx": 3}.get(gate_spec["gate"], 1)
                    > qubit_count
                ),
                None,
            )
            if unfit_gate is not None:
                st.warning(
                    f"Preview unavailable: The {unfit_gate} gate no longer fits "
                    "the current qubit count."
                )
            else:
                st.warning(f"Preview unavailable: {error}")

        st.caption(
            "Gate order is preserved. Named rotation parameters can be swept "
            "in the Variational Lab."
        )

        run_clicked = st.button(
            "Run simulation",
            type="primary",
            key="run_simulation_button",
            disabled=preview_circuit is None,
        )
        if run_clicked:
            st.session_state.pop("simulation_result", None)
            with st.spinner("Simulating..."):
                try:
                    circuit = preview_circuit
                    if circuit is None:
                        raise ValueError("The circuit preview is not valid.")
                    amplitudes, exact_probabilities = get_statevector_data(circuit)
                    noise_model = (
                        build_noise_model(
                            bit_flip_probability,
                            phase_flip_probability,
                            depolarizing_probability,
                            amplitude_damping_probability,
                            readout_error_probability,
                        )
                        if use_noise
                        else None
                    )
                    if noise_model is not None and (qubit_count > MAX_NOISY_EXACT_QUBITS):
                        st.info(
                            "Exact noisy probabilities are limited to "
                            f"{MAX_NOISY_EXACT_QUBITS} qubits. Showing ideal probabilities "
                            "and sampled noisy shot frequencies only."
                        )
                    noisy_probabilities = (
                        get_noisy_probabilities(
                            circuit,
                            noise_model,
                            readout_error_probability,
                        )
                        if noise_model is not None and qubit_count <= MAX_NOISY_EXACT_QUBITS
                        else None
                    )
                    shot_counts = run_shot_experiment(
                        circuit,
                        int(shots),
                        seed_simulator=(
                            int(seed_simulator) if seed_simulator is not None else None
                        ),
                        noise_model=noise_model,
                    )
                    shot_label = "Shots (noisy)" if noise_model is not None else "Shots"
                    shot_errors = calculate_shot_standard_errors(shot_counts, int(shots))
                    outcomes, comparison_data = _build_comparison_data(
                        exact_probabilities,
                        shot_counts,
                        int(shots),
                        shot_label=shot_label,
                        noisy_probabilities=noisy_probabilities,
                        shot_errors=shot_errors,
                    )
                    state_data = _build_state_inspector_data(
                        amplitudes,
                        exact_probabilities,
                    )
                    metrics = get_qubit_state_metrics(circuit)
                    prefix_steps: list[dict[str, Any]] = []
                    for prefix_length in range(len(gate_sequence) + 1):
                        prefix = build_prefix_circuit(qubit_count, gate_sequence, prefix_length)
                        prefix_values = {
                            parameter.name: parameter_values[parameter.name]
                            for parameter in prefix.parameters
                        }
                        prefix = bind_circuit_parameters(prefix, prefix_values)
                        prefix_amplitudes, prefix_probabilities = get_statevector_data(prefix)
                        prefix_steps.append(
                            {
                                "circuit": prefix,
                                "state_data": _build_state_inspector_data(
                                    prefix_amplitudes, prefix_probabilities
                                ),
                            }
                        )
                except ValueError as error:
                    st.error(f"Invalid circuit or simulation settings: {error}")
                except Exception:
                    logger.exception("Quantum circuit simulation failed unexpectedly.")
                    st.error(
                        "The simulation could not be completed. Check the circuit "
                        "and settings, then try again."
                    )
                else:
                    st.session_state["simulation_result"] = {
                        "outcomes": outcomes,
                        "comparison_data": comparison_data,
                        "state_data": state_data,
                        "shot_counts": shot_counts,
                        "exact_probabilities": exact_probabilities,
                        "noisy_probabilities": noisy_probabilities,
                        "shot_errors": shot_errors,
                        "metrics": metrics,
                        "prefix_steps": prefix_steps,
                        "gate_sequence": gate_sequence,
                        "qubit_count": qubit_count,
                        "parameterized_circuit": parameterized_circuit,
                        "parameter_values": parameter_values,
                        "shots": int(shots),
                        "noise_enabled": noise_model is not None,
                        "bit_flip_probability": bit_flip_probability,
                        "phase_flip_probability": phase_flip_probability,
                        "depolarizing_probability": depolarizing_probability,
                        "amplitude_damping_probability": amplitude_damping_probability,
                        "readout_error_probability": readout_error_probability,
                    }

        simulation_result = st.session_state.get("simulation_result")
        if simulation_result is not None:
            st.subheader("Step through gate prefixes")
            if len(simulation_result["prefix_steps"]) > 1:
                step = st.slider(
                    "Show state after gate",
                    min_value=0,
                    max_value=len(simulation_result["prefix_steps"]) - 1,
                    key="gate_step",
                )
            else:
                step = 0
            prefix = simulation_result["prefix_steps"][step]
            _show_circuit(prefix["circuit"])
            st.caption(
                f"Showing the ideal state after {step} of "
                f"{len(simulation_result['gate_sequence'])} gates."
            )
            st.subheader("State after selected gate")
            prefix_state_data = prefix["state_data"].nlargest(
                int(max_display_outcomes), "Probability"
            )
            _render_state_inspector(prefix_state_data.reset_index(drop=True))

    simulation_result = st.session_state.get("simulation_result")
    with comparison_tab:
        st.subheader("Ideal, noisy, and sampled outcome probabilities")
        if simulation_result is None:
            st.info("Build or load a circuit, then run it from the workspace tab.")
        else:
            visible_outcomes = _rank_outcomes(
                simulation_result["exact_probabilities"],
                simulation_result["shot_counts"],
                simulation_result["shots"],
                int(max_display_outcomes),
                simulation_result["noisy_probabilities"],
            )
            displayed_ideal_probability = sum(
                simulation_result["exact_probabilities"].get(outcome, 0.0)
                for outcome in visible_outcomes
            )
            displayed_shot_probability = (
                sum(
                    simulation_result["shot_counts"].get(outcome, 0) for outcome in visible_outcomes
                )
                / simulation_result["shots"]
            )
            st.caption(
                f"Showing top {len(visible_outcomes)} of "
                f"{len(simulation_result['outcomes'])} outcomes (covering "
                f"{displayed_ideal_probability:.1%} of ideal probability, "
                f"{displayed_shot_probability:.1%} of shots)."
            )
            _, comparison_data = _filter_result_frames(
                simulation_result["state_data"],
                simulation_result["comparison_data"],
                visible_outcomes,
            )
            _render_probability_chart(
                visible_outcomes,
                comparison_data,
            )
            if simulation_result["noise_enabled"]:
                if simulation_result["noisy_probabilities"] is None:
                    st.info(
                        "Exact noisy probabilities are limited to "
                        f"{MAX_NOISY_EXACT_QUBITS} qubits. Showing ideal probabilities "
                        "and sampled noisy shot frequencies only."
                    )
                    st.caption(
                        "Noisy shot bars sample the configured model. Idle-time "
                        "noise is not modeled."
                    )
                else:
                    st.caption(
                        "Exact bars show the ideal state. Noisy Exact uses density "
                        "matrix evolution; shot bars sample the same model. Pauli "
                        "bit-flip "
                        f"{simulation_result['bit_flip_probability']:.2%}, phase-flip "
                        f"{simulation_result['phase_flip_probability']:.2%}, "
                        "depolarizing "
                        f"{simulation_result['depolarizing_probability']:.2%}, "
                        "amplitude damping "
                        f"{simulation_result['amplitude_damping_probability']:.2%}, "
                        "readout "
                        f"{simulation_result['readout_error_probability']:.2%}. "
                        "Idle-time noise is not modeled."
                    )
            else:
                st.caption(
                    "Error bars show one binomial standard error for finite-shot "
                    "sampling; they do not represent device uncertainty."
                )
            st.caption(
                "Outcome labels use Qiskit's little-endian convention: q0 is the rightmost bit."
            )

    with inspector_tab:
        st.subheader("Ideal quantum state")
        if simulation_result is None:
            st.info("Run a circuit to inspect its ideal statevector.")
        else:
            visible_outcomes = _rank_outcomes(
                simulation_result["exact_probabilities"],
                simulation_result["shot_counts"],
                simulation_result["shots"],
                int(max_display_outcomes),
                simulation_result["noisy_probabilities"],
            )
            state_data, _ = _filter_result_frames(
                simulation_result["state_data"],
                simulation_result["comparison_data"],
                visible_outcomes,
            )
            _render_state_inspector(state_data)
            st.subheader("Per-qubit reduced states")
            _render_bloch_sphere(simulation_result["metrics"])
            metric_frame = pd.DataFrame(simulation_result["metrics"]).rename(
                columns={
                    "qubit": "Qubit",
                    "x": "Bloch X",
                    "y": "Bloch Y",
                    "z": "Bloch Z",
                    "purity": "Purity",
                    "entropy": "Entropy (bits)",
                }
            )
            st.dataframe(
                metric_frame.style.format(
                    {column: "{:.4f}" for column in metric_frame.columns if column != "Qubit"}
                ),
                width="stretch",
                hide_index=True,
            )
            st.subheader("Complex amplitude phase")
            _render_phase_chart(state_data)

    with variational_tab:
        st.subheader("Parameterized circuit sweep")
        if parameterized_circuit is None or not parameterized_circuit.parameters:
            st.info(
                "Enable “Use named parameter” on a rotation/phase gate to "
                "create a sweepable circuit parameter."
            )
        else:
            parameter_names: list[str] = sorted(
                parameter.name for parameter in parameterized_circuit.parameters
            )
            selected_parameter = st.selectbox(
                "Sweep parameter",
                options=parameter_names,
                index=0,
                key="sweep_parameter_name",
            )
            sweep_start, sweep_end = st.slider(
                "Sweep range (radians)",
                min_value=-2.0 * math.pi,
                max_value=2.0 * math.pi,
                value=(0.0, 2.0 * math.pi),
                step=0.01,
                format="%.2f",
                key="sweep_range",
            )
            sweep_points = st.slider(
                "Sweep points",
                min_value=5,
                max_value=101,
                value=41,
                step=1,
                key="sweep_points",
            )
            try:
                fixed_values = {
                    name: value
                    for name, value in parameter_values.items()
                    if name != selected_parameter
                }
                sweep_gate_sequence = tuple(
                    (
                        str(gate_spec["gate"]),
                        tuple(int(qubit) for qubit in gate_spec["qubits"]),
                        (
                            None
                            if isinstance(gate_spec.get("angle"), str)
                            else (float(gate_spec["angle"]) if "angle" in gate_spec else None)
                        ),
                        (
                            str(gate_spec["angle"])
                            if isinstance(gate_spec.get("angle"), str)
                            else None
                        ),
                    )
                    for gate_spec in gate_sequence
                )
                sweep_results = _cached_parameter_sweep(
                    int(qubit_count),
                    sweep_gate_sequence,
                    (selected_parameter,),
                    (float(sweep_start), float(sweep_end)),
                    int(sweep_points),
                    tuple(sorted((name, float(value)) for name, value in fixed_values.items())),
                )
                displayed_sweep_outcomes, sweep_caption = _select_sweep_outcomes(
                    sweep_results,
                    int(qubit_count),
                    int(max_display_outcomes),
                )
                sweep_frame = pd.DataFrame(
                    [
                        {
                            "Angle": result["value"],
                            "Outcome": outcome,
                            "Probability": probability,
                        }
                        for result in sweep_results
                        for outcome, probability in result["probabilities"].items()
                        if outcome in displayed_sweep_outcomes
                    ]
                )
                sweep_chart = (
                    alt.Chart(sweep_frame)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("Angle:Q", title=f"{selected_parameter} (radians)"),
                        y=alt.Y("Probability:Q", scale=alt.Scale(zero=True)),
                        color=alt.Color("Outcome:N"),
                    )
                    .properties(height=340)
                )
                st.altair_chart(sweep_chart, width="stretch")
                st.caption(sweep_caption)
            except ValueError as error:
                st.error(f"Could not run parameter sweep: {error}")
            except Exception:
                logger.exception("Quantum circuit parameter sweep failed unexpectedly.")
                st.error(
                    "The parameter sweep could not be completed. Check the "
                    "circuit and settings, then try again."
                )

        st.divider()
        st.subheader("Toy VQE: minimize ⟨Z⟩")
        st.write(
            "A one-qubit Ry ansatz minimizes the Pauli-Z Hamiltonian. This small "
            "exact-statevector optimizer is a teaching demonstration."
        )
        if st.button("Run toy VQE", key="run_vqe_button"):
            result = run_toy_vqe()
            st.session_state["vqe_result"] = result
        vqe_result = st.session_state.get("vqe_result")
        if vqe_result is not None:
            energy_frame = pd.DataFrame(vqe_result["history"])
            energy_chart = (
                alt.Chart(energy_frame)
                .mark_line()
                .encode(
                    x=alt.X("theta:Q", title="θ (radians)"),
                    y=alt.Y("energy:Q", title="⟨Z⟩", scale=alt.Scale(zero=False)),
                )
                .properties(height=320)
            )
            st.altair_chart(energy_chart, width="stretch")
            st.metric("Estimated ground energy", f"{vqe_result['energy']:.6f}")
            st.caption(
                f"Exact ground energy: {vqe_result['ground_energy']:.1f}; "
                f"optimized θ: {vqe_result['theta']:.4f} radians."
            )


if __name__ == "__main__":
    main()
