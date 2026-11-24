# /// script
# dependencies = ["numpy", "scikit-learn", "matplotlib"]
# ///
"""Train a small MLP that estimates angle of attack from simulated sensors only,
then compare it with the INS (Kalman) AoA and the ground truth.

    cargo run -p fsim-sim --release --example aoa_dataset -- aoa.csv 60
    uv run scripts/aoa_nn.py aoa.csv
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Sensor + INS + control inputs only. Never wind, turb, or alpha_true.
FEATURES = ["ax", "ay", "az", "gx", "gy", "gz", "baro", "ins_ub", "ins_vb", "ins_wb",
            "ins_roll", "ins_pitch", "elevator", "aileron", "throttle", "alpha_ins"]

path = sys.argv[1] if len(sys.argv) > 1 else "aoa.csv"
d = np.genfromtxt(path, delimiter=",", names=True)
d = d[d["t"] > 10.0]  # let the INS converge first
X = np.column_stack([d[f] for f in FEATURES])
y = d["alpha_true"]
test = d["run"] % 4 == 0  # whole flights held out, never seen in training

model = make_pipeline(
    StandardScaler(),
    MLPRegressor(hidden_layer_sizes=(32, 32), early_stopping=True, max_iter=300, random_state=0),
)
model.fit(X[~test], y[~test])
nn = model.predict(X)

rmse = lambda e: np.degrees(np.sqrt(np.mean(e**2)))
groups = {
    "all test": test,
    "calm (no wind)": test & (d["wind"] == 0),
    "turb < 1.5 m/s": test & (d["wind"] > 0) & (d["turb"] < 1.5),
    "turb >= 1.5 m/s": test & (d["turb"] >= 1.5),
}
print(f"{'test set':<18}{'rows':>8}{'Kalman RMSE':>14}{'NN RMSE':>10}   (deg)")
for name, m in groups.items():
    if m.any():
        print(f"{name:<18}{m.sum():>8}{rmse(d['alpha_ins'][m] - y[m]):>14.2f}{rmse(nn[m] - y[m]):>10.2f}")

# One held-out windy flight over time.
run = max(set(d["run"][test & (d["wind"] > 0)]), key=lambda r: d["turb"][d["run"] == r][0])
m = d["run"] == run
plt.figure(figsize=(11, 4))
plt.plot(d["t"][m], np.degrees(y[m]), "k", lw=1.5, label="truth (air-relative)")
plt.plot(d["t"][m], np.degrees(d["alpha_ins"][m]), "tab:orange", lw=1, label="Kalman (INS ground velocity)")
plt.plot(d["t"][m], np.degrees(nn[m]), "tab:blue", lw=1, label="neural net")
plt.xlabel("t [s]"), plt.ylabel("angle of attack [deg]")
plt.title(f"held-out flight {int(run)}: wind {d['wind'][m][0]:.1f} m/s, turbulence {d['turb'][m][0]:.1f} m/s")
plt.legend(), plt.grid(alpha=0.3), plt.tight_layout()
out = path.rsplit(".", 1)[0] + ".png"
plt.savefig(out, dpi=120)
print("plot:", out)
