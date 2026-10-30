import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

device_width_um = 1.0
criterion_current_a = 100e-9 * device_width_um / 0.065

curve_files = {
    "Nominal (nmos0)": {0.05: "VDS_50m.txt", 1.1: "VDS_1.1.txt"},
    "UA = UB = 0 (nmos1)": {0.05: "uaub0_VDS_50m.txt", 1.1: "uaub0_VDS_1.1.txt"},
    "VSAT x 100 (nmos2)": {0.05: "vsat100_VDS_50m.txt", 1.1: "vsat100_VDS_1.1.txt"},
}


def load_transfer_curve(file_name):
    table = pd.read_csv(file_name, sep="\t")
    gate_voltage = table.iloc[:, 0].to_numpy(dtype=float)
    drain_current = np.abs(table.iloc[:, 1].to_numpy(dtype=float))
    order = np.argsort(gate_voltage)
    return gate_voltage[order], drain_current[order]


curves = {
    model_label: {
        drain_voltage: load_transfer_curve(file_name)
        for drain_voltage, file_name in files.items()
    }
    for model_label, files in curve_files.items()
}

figure, axes = plt.subplots(2, 2, figsize=(12, 9))
drain_voltages = [0.05, 1.1]
drain_voltage_titles = {0.05: "$V_{DS}$ = 50 mV", 1.1: "$V_{DS}$ = 1.1 V"}

for row, drain_voltage in enumerate(drain_voltages):
    linear_axis, log_axis = axes[row]
    for model_label, model_curves in curves.items():
        gate_voltage, drain_current = model_curves[drain_voltage]
        current_density = drain_current / device_width_um
        linear_axis.plot(gate_voltage, current_density * 1e3, linewidth=1.8, label=model_label)
        log_axis.semilogy(gate_voltage, current_density, linewidth=1.8, label=model_label)

    linear_axis.set_title(f"Linear scale, {drain_voltage_titles[drain_voltage]}")
    linear_axis.set_xlabel("Gate-to-source voltage $V_{GS}$ (V)")
    linear_axis.set_ylabel("Drain current $I_D$ (mA/µm)")
    log_axis.set_title(f"Logarithmic scale, {drain_voltage_titles[drain_voltage]}")
    log_axis.set_xlabel("Gate-to-source voltage $V_{GS}$ (V)")
    log_axis.set_ylabel("Drain current $I_D$ (A/µm)")
    for axis in (linear_axis, log_axis):
        axis.grid(True, alpha=0.3)
        axis.legend()

figure.tight_layout()
figure.savefig("S4_physics_check.png", dpi=300)

summary_rows = []
for model_label, model_curves in curves.items():
    for drain_voltage in drain_voltages:
        gate_voltage, drain_current = model_curves[drain_voltage]
        threshold_voltage = np.interp(criterion_current_a, drain_current, gate_voltage)
        summary_rows.append(
            {
                "Model": model_label,
                "VDS (V)": drain_voltage,
                "VTH,cc (V)": threshold_voltage,
                "ION at VGS = 1.1 V (mA/µm)": np.interp(1.1, gate_voltage, drain_current) * 1e3 / device_width_um,
                "IOFF at VGS = 0 (nA/µm)": np.interp(0.0, gate_voltage, drain_current) * 1e9 / device_width_um,
            }
        )

summary_table = pd.DataFrame(summary_rows)
summary_table.to_csv("S4_summary.csv", index=False)
print(summary_table.round(3).to_string(index=False))

plt.show()