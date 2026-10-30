import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

export_file_name = "Final_Plots.txt"

experiments = [
    {"parameter": "VTH0", "labels": ["0.30 V", "0.423 V (nominal)", "0.55 V"]},
    {"parameter": "TOXE", "labels": ["1.5 nm", "1.85 nm (nominal)", "2.5 nm"]},
    {"parameter": "L (drawn)", "labels": ["55 nm", "65 nm (nominal)", "100 nm"],
     "length_um": [0.055, 0.065, 0.100]},
    {"parameter": "XL", "labels": ["-35 nm", "-30 nm (nominal)", "-15 nm"]},
    {"parameter": "LINT", "labels": ["2 nm", "5.25 nm (nominal)", "8 nm"]},
    {"parameter": "W (drawn)", "labels": ["0.5 µm", "1 µm (nominal)", "2 µm"],
     "width_um": [0.5, 1.0, 2.0]},
    {"parameter": "WINT", "labels": ["0 nm", "5 nm (nominal)", "40 nm"]},
    {"parameter": "NDEP", "labels": ["1.5e18 cm⁻³", "2.54e18 cm⁻³ (nominal)", "4e18 cm⁻³"]},
    {"parameter": "NSD", "labels": ["5e19 cm⁻³", "2e20 cm⁻³ (nominal)", "5e20 cm⁻³"]},
    {"parameter": "XJ", "labels": ["10 nm", "19.6 nm (nominal)", "30 nm"]},
    {"parameter": "NGATE", "labels": ["1e19 cm⁻³", "2e20 cm⁻³ (nominal)", "1e21 cm⁻³"]},
    {"parameter": "U0", "labels": ["300 cm²/V·s", "491 cm²/V·s (nominal)", "700 cm²/V·s"]},
    {"parameter": "VSAT", "labels": ["0.8e7 cm/s", "1.24e7 cm/s (nominal)", "2e7 cm/s"]},
    {"parameter": "RDSW", "labels": ["50 Ω·µm", "165 Ω·µm (nominal)", "400 Ω·µm"]},
]


def read_stepped_export(file_name):
    curves = {}
    current_key = None
    sweep_values, current_values = [], []

    def store_curve():
        if current_key is not None and sweep_values:
            curves[current_key] = (np.array(sweep_values), np.abs(np.array(current_values)))

    with open(file_name, encoding="latin-1") as export_file:
        for line in export_file:
            line = line.strip()
            if line.startswith("Step Information"):
                store_curve()
                words = line.split()
                mode_number = int(words[2].split("=")[1])
                case_number = int(words[3].split("=")[1])
                current_key = (mode_number, case_number)
                sweep_values, current_values = [], []
            elif current_key is not None and line:
                sweep_value, current_value = line.split()[:2]
                sweep_values.append(float(sweep_value))
                current_values.append(float(current_value))
    store_curve()
    return curves


def threshold_constant_current(gate_voltage, drain_current, width_um, length_um):
    criterion_current = 100e-9 * width_um / length_um
    return np.interp(criterion_current, drain_current, gate_voltage)


def subthreshold_swing(gate_voltage, drain_current, threshold_voltage):
    mask = (gate_voltage < threshold_voltage) & (drain_current > 0)
    swing = np.gradient(gate_voltage[mask], np.log10(drain_current[mask])) * 1000
    return swing.min()


curves = read_stepped_export(export_file_name)
summary_rows = []

for experiment_number, experiment in enumerate(experiments, start=1):
    widths_um = experiment.get("width_um", [1.0, 1.0, 1.0])
    lengths_um = experiment.get("length_um", [0.065, 0.065, 0.065])

    figure, (axis_linear, axis_log, axis_output) = plt.subplots(1, 3, figsize=(17, 4.8))
    line_styles = ["-", "--", ":"]
    for level in range(3):
        case_number = 3 * (experiment_number - 1) + level + 1
        gate_voltage, gate_sweep_current = curves[(0, case_number)]
        drain_voltage, drain_sweep_current = curves[(1, case_number)]
        label = f"{experiment['parameter']} = {experiment['labels'][level]}"

        axis_linear.plot(gate_voltage, gate_sweep_current * 1e6, linewidth=1.8, label=label,linestyle=line_styles[level])
        axis_log.semilogy(gate_voltage, gate_sweep_current, linewidth=1.8, label=label,linestyle=line_styles[level])
        axis_output.plot(drain_voltage, drain_sweep_current * 1e6, linewidth=1.8, label=label,linestyle=line_styles[level])

        threshold_voltage = threshold_constant_current(
            gate_voltage, gate_sweep_current, widths_um[level], lengths_um[level]
        )
        summary_rows.append(
            {
                "Exp": experiment_number,
                "Parameter": experiment["parameter"],
                "Value": experiment["labels"][level],
                "VTH,cc at VDS = 0.4 V (V)": threshold_voltage,
                "SS at VDS = 0.4 V (mV/dec)": subthreshold_swing(
                    gate_voltage, gate_sweep_current, threshold_voltage
                ),
                "IOFF at VGS = 0, VDS = 0.4 V (nA/µm)": gate_sweep_current[0] * 1e9 / widths_um[level],
                "ION at VGS = VDS = 1.1 V (µA/µm)": np.interp(1.1, drain_voltage, drain_sweep_current)
                * 1e6 / widths_um[level],
            }
        )

    axis_linear.set_title("ID–VG at VDS = 0.4 V (linear)")
    axis_linear.set_xlabel("Gate-to-source voltage $V_{GS}$ (V)")
    axis_linear.set_ylabel("Drain current $I_D$ (µA)")
    axis_log.set_title("ID–VG at VDS = 0.4 V (log)")
    axis_log.set_xlabel("Gate-to-source voltage $V_{GS}$ (V)")
    axis_log.set_ylabel("Drain current $I_D$ (A)")
    axis_output.set_title("ID–VD at VGS = 1.1 V")
    axis_output.set_xlabel("Drain-to-source voltage $V_{DS}$ (V)")
    axis_output.set_ylabel("Drain current $I_D$ (µA)")
    for axis in (axis_linear, axis_log, axis_output):
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)

    figure.suptitle(f"Experiment {experiment_number}: {experiment['parameter']}")
    figure.tight_layout()
    figure.savefig(f"S5_exp{experiment_number:02d}.png", dpi=300)
    plt.close(figure)

summary_table = pd.DataFrame(summary_rows)
print(summary_table.round(3).to_string(index=False))