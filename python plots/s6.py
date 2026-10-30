import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

spice_export_file = "s6_fit.txt"
tcad_gate_sweep_file = "0.4volt.csv"
tcad_drain_sweep_file = "0.9.csv"
device_width_um = 1.0
drain_sweep_limit = 1.1


def read_spice_export(file_name):
    curves = {}
    current_mode = None
    sweep_values, current_values = [], []

    def store_curve():
        if current_mode is not None and sweep_values:
            curves[current_mode] = (np.array(sweep_values), np.abs(np.array(current_values)))

    with open(file_name, encoding="latin-1") as export_file:
        for line in export_file:
            line = line.strip()
            if line.startswith("Step Information"):
                store_curve()
                current_mode = int(line.split()[2].split("=")[1])
                sweep_values, current_values = [], []
            elif current_mode is not None and line:
                sweep_value, current_value = line.split()[:2]
                sweep_values.append(float(sweep_value))
                current_values.append(float(current_value))
    store_curve()
    return curves


def read_tcad_curve(file_name):
    table = pd.read_csv(file_name)
    return table.iloc[:, 0].to_numpy(dtype=float), np.abs(table.iloc[:, 1].to_numpy(dtype=float))


def threshold_max_gm(gate_voltage, drain_current):
    gm = np.gradient(drain_current, gate_voltage)
    peak = np.argmax(gm)
    return gate_voltage[peak] - drain_current[peak] / gm[peak]


def percent_errors(tcad_voltage, tcad_current, spice_voltage, spice_current):
    spice_at_tcad = np.interp(tcad_voltage, spice_voltage, spice_current)
    significant = tcad_current > 0.1 * tcad_current.max()
    errors = 100 * (spice_at_tcad[significant] - tcad_current[significant]) / tcad_current[significant]
    return np.mean(np.abs(errors)), np.max(np.abs(errors))


spice_curves = read_spice_export(spice_export_file)
spice_gate_voltage, spice_gate_current = spice_curves[0]
spice_drain_voltage, spice_drain_current = spice_curves[1]
keep = spice_drain_voltage <= drain_sweep_limit + 1e-9
spice_drain_voltage, spice_drain_current = spice_drain_voltage[keep], spice_drain_current[keep]

tcad_gate_voltage, tcad_gate_current = read_tcad_curve(tcad_gate_sweep_file)
tcad_drain_voltage, tcad_drain_current = read_tcad_curve(tcad_drain_sweep_file)

figure, (axis_transfer, axis_output) = plt.subplots(1, 2, figsize=(12, 4.8))

axis_transfer.plot(tcad_gate_voltage, tcad_gate_current / device_width_um * 1e6, "o",
                   markersize=5, label="TCAD")
axis_transfer.plot(spice_gate_voltage, spice_gate_current / device_width_um * 1e6,
                   linewidth=1.8, label="SPICE (nfit)")
axis_transfer.set_title("ID–VG at VDS = 0.4 V")
axis_transfer.set_xlabel("Gate-to-source voltage $V_{GS}$ (V)")
axis_transfer.set_ylabel("Drain current $I_D$ (µA/µm)")

axis_output.plot(tcad_drain_voltage, tcad_drain_current / device_width_um * 1e6, "o",
                 markersize=5, label="TCAD")
axis_output.plot(spice_drain_voltage, spice_drain_current / device_width_um * 1e6,
                 linewidth=1.8, label="SPICE (nfit)")
axis_output.set_title("ID–VD at VGS = 0.9 V")
axis_output.set_xlabel("Drain-to-source voltage $V_{DS}$ (V)")
axis_output.set_ylabel("Drain current $I_D$ (µA/µm)")

for axis in (axis_transfer, axis_output):
    axis.grid(True, alpha=0.3)
    axis.legend()

figure.tight_layout()
figure.savefig("S6_overlay.png", dpi=300)

transfer_mean, transfer_max = percent_errors(
    tcad_gate_voltage, tcad_gate_current, spice_gate_voltage, spice_gate_current)
output_mean, output_max = percent_errors(
    tcad_drain_voltage, tcad_drain_current, spice_drain_voltage, spice_drain_current)

summary = pd.DataFrame(
    {
        "Curve": ["ID–VG at VDS = 0.4 V", "ID–VD at VGS = 0.9 V"],
        "Mean |error| (%)": [transfer_mean, output_mean],
        "Max |error| (%)": [transfer_max, output_max],
    }
)
print(summary.round(1).to_string(index=False))
print(f"VTH,lin from max gm: TCAD = {threshold_max_gm(tcad_gate_voltage, tcad_gate_current):.3f} V, "
      f"SPICE = {threshold_max_gm(spice_gate_voltage, spice_gate_current):.3f} V")

plt.show()
