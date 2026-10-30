import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

gate_voltage_files = {
    0.3: "VGS_0.3.txt",
    0.6: "VGS_0.6.txt",
    0.9: "VGS_0.9.txt",
    1.5: "VGS_1.5.txt",
}

device_width_um = 1.0
evaluation_voltages = [1.1, 1.5]

output_curves = {}
for gate_voltage, file_name in gate_voltage_files.items():
    table = pd.read_csv(file_name, sep="\t")
    table.columns = ["drain_voltage", "drain_current"]
    drain_voltage = table["drain_voltage"].to_numpy(dtype=float)
    drain_current = np.abs(table["drain_current"].to_numpy(dtype=float))
    order = np.argsort(drain_voltage)
    drain_voltage = drain_voltage[order]
    drain_current = drain_current[order]
    output_curves[gate_voltage] = {
        "drain_voltage": drain_voltage,
        "drain_current": drain_current,
        "output_conductance": np.gradient(drain_current, drain_voltage),
    }

figure_both, (axis_current, axis_conductance) = plt.subplots(1, 2, figsize=(13, 5))

for gate_voltage, curve in output_curves.items():
    curve_label = f"$V_{{GS}}$ = {gate_voltage} V"
    axis_current.plot(
        curve["drain_voltage"],
        curve["drain_current"] * 1e6 / device_width_um,
        linewidth=1.8,
        label=curve_label,
    )
    axis_conductance.plot(
        curve["drain_voltage"],
        curve["output_conductance"] * 1e6 / device_width_um,
        linewidth=1.8,
        linestyle="--",
        label=curve_label,
    )

axis_current.set_xlabel("Drain-to-source voltage $V_{DS}$ (V)")
axis_current.set_ylabel("Drain current $I_D$ (µA/µm)")
axis_current.set_title("(a) Output characteristics ($I_D$ vs $V_{DS}$)")
axis_current.grid(True, alpha=0.3)
axis_current.legend()

axis_conductance.set_xlabel("Drain-to-source voltage $V_{DS}$ (V)")
axis_conductance.set_ylabel("Output conductance $g_{ds}$ (µS/µm)")
axis_conductance.set_title("(b) Output conductance ($g_{ds}$ = dI$_D$/dV$_{DS}$)")
axis_conductance.grid(True, alpha=0.3)
axis_conductance.legend()

figure_both.tight_layout()
figure_both.savefig("S3_output_characteristics_and_gds.png", dpi=300)

summary_rows = []
for gate_voltage, curve in output_curves.items():
    for evaluation_voltage in evaluation_voltages:
        if evaluation_voltage > curve["drain_voltage"].max():
            continue
        current_here = np.interp(
            evaluation_voltage, curve["drain_voltage"], curve["drain_current"]
        )
        conductance_here = np.interp(
            evaluation_voltage, curve["drain_voltage"], curve["output_conductance"]
        )
        if conductance_here > 0:
            output_resistance_kohm_um = (1.0 / conductance_here) * device_width_um / 1e3
        else:
            output_resistance_kohm_um = np.inf
        summary_rows.append(
            {
                "VGS (V)": gate_voltage,
                "VDS (V)": evaluation_voltage,
                "ID (µA/µm)": current_here * 1e6 / device_width_um,
                "gds (µS/µm)": conductance_here * 1e6 / device_width_um,
                "ro (kΩ·µm)": output_resistance_kohm_um,
            }
        )

summary_table = pd.DataFrame(summary_rows)
summary_table.to_csv("S3_summary.csv", index=False)
print(summary_table.round(3).to_string(index=False))

plt.show()