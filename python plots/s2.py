import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

WIDTH_UM = 1.0
LENGTH_UM = 0.065
CRITERION_CURRENT_A = 100e-9 * WIDTH_UM / LENGTH_UM


def load_transfer_curve(file_name):
    data = pd.read_csv(file_name, sep="\t")
    gate_voltage = data.iloc[:, 0].to_numpy(dtype=float)
    drain_current = np.abs(data.iloc[:, 1].to_numpy(dtype=float))
    order = np.argsort(gate_voltage)
    return gate_voltage[order], drain_current[order]


def threshold_max_gm(gate_voltage, drain_current):
    gm = np.gradient(drain_current, gate_voltage)
    peak = np.argmax(gm)
    return gate_voltage[peak] - drain_current[peak] / gm[peak]


def threshold_constant_current(gate_voltage, drain_current):
    return np.interp(CRITERION_CURRENT_A, drain_current, gate_voltage)


def subthreshold_swing(gate_voltage, drain_current, threshold):
    mask = (gate_voltage < threshold) & (drain_current > 0)
    log_current = np.log10(drain_current[mask])
    swing = np.gradient(gate_voltage[mask], log_current) * 1000
    return np.min(swing)


def current_at(gate_voltage, drain_current, target_voltage):
    return np.interp(target_voltage, gate_voltage, drain_current)


vg_low, id_low = load_transfer_curve("VDS_50m.txt")
vg_high, id_high = load_transfer_curve("VDS_1.1.txt")

vth_linear = threshold_max_gm(vg_low, id_low)
vth_cc_low = threshold_constant_current(vg_low, id_low)
vth_cc_high = threshold_constant_current(vg_high, id_high)
dibl = (vth_cc_low - vth_cc_high) / (1.1 - 0.05) * 1000
ss_low = subthreshold_swing(vg_low, id_low, vth_cc_low)
ss_high = subthreshold_swing(vg_high, id_high, vth_cc_high)
ion = current_at(vg_high, id_high, 1.1) / WIDTH_UM
ioff = current_at(vg_high, id_high, 0.0) / WIDTH_UM

results = pd.DataFrame(
    {
        "Quantity": [
            "VTH,lin (V)", "VTH,cc at 50 mV (V)", "VTH,cc at 1.1 V (V)",
            "DIBL (mV/V)", "SS at 50 mV (mV/dec)", "SS at 1.1 V (mV/dec)",
            "ION (mA/um)", "IOFF (nA/um)",
        ],
        "SPICE": [
            vth_linear, vth_cc_low, vth_cc_high, dibl, ss_low, ss_high,
            ion * 1e3, ioff * 1e9,
        ],
        "Table 5.1": [0.38, 0.27, 0.13, 130, 90, 96, 1.15, 67],
    }
)
print(results.round(3).to_string(index=False))

fig, (linear_axis, log_axis) = plt.subplots(1, 2, figsize=(11, 4.5))
for voltage, current, label in [
    (vg_low, id_low, "VDS = 50 mV"),
    (vg_high, id_high, "VDS = 1.1 V"),
]:
    linear_axis.plot(voltage, current / WIDTH_UM * 1e3, label=label)
    log_axis.semilogy(voltage, current / WIDTH_UM, label=label)

linear_axis.set_xlabel("VGS (V)")
linear_axis.set_ylabel("ID (mA/um)")
linear_axis.set_title("Linear scale")
log_axis.set_xlabel("VGS (V)")
log_axis.set_ylabel("ID (A/um)")
log_axis.set_title("Logarithmic scale")
for axis in (linear_axis, log_axis):
    axis.grid(True, alpha=0.3)
    axis.legend()

plt.tight_layout()
plt.savefig("s2_transfer_characteristic.png", dpi=200)
plt.show()