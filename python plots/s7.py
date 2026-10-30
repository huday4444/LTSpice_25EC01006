import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

export_file = "Vout.txt"
supply_voltage = 1.1

data = pd.read_csv(export_file, sep="\t")
input_voltage = data.iloc[:, 0].to_numpy(dtype=float)
output_voltage = data.iloc[:, 1].to_numpy(dtype=float)

difference = output_voltage - input_voltage
crossing_index = np.where(np.diff(np.sign(difference)) != 0)[0][0]
fraction = difference[crossing_index] / (difference[crossing_index] - difference[crossing_index + 1])
switching_threshold = input_voltage[crossing_index] + fraction * (
    input_voltage[crossing_index + 1] - input_voltage[crossing_index]
)

print(f"Vout at Vin = 0 V:     {output_voltage[0]:.4f} V")
print(f"Vout at Vin = {supply_voltage} V:   {output_voltage[-1]:.4f} V")
print(f"Switching threshold:   {switching_threshold:.4f} V")
print(f"VDD / 2:               {supply_voltage / 2:.4f} V")

figure, axis = plt.subplots(figsize=(6, 5))

axis.plot(input_voltage, output_voltage, linewidth=2, label="Vout")
axis.plot(input_voltage, input_voltage, "--", color="gray", label="Vout = Vin")
axis.plot(switching_threshold, switching_threshold, "o", color="red",
          label=f"Switching threshold = {switching_threshold:.3f} V")

axis.set_xlabel("Input voltage Vin (V)")
axis.set_ylabel("Output voltage Vout (V)")
axis.set_title("CMOS inverter voltage transfer characteristic")
axis.set_xlim(0, supply_voltage)
axis.set_ylim(0, supply_voltage)
axis.grid(True, alpha=0.3)
axis.legend()

figure.tight_layout()
figure.savefig("s7_inverter_vtc.png", dpi=300)
plt.show()
