# /// script
# dependencies = ["numpy", "scikit-learn", "matplotlib"]
# ///
"""Train a small MLP that estimates angle of attack from simulated sensors only,
then compare it with the INS (Kalman) AoA and the ground truth. Writes
<csv>.png and <csv>.weights.txt (loaded by `aoa_nn loop` to fly with it).

    cargo run -p fsim-sim --release --example aoa_nn -- dataset fighter aoa.csv 60
    uv run scripts/aoa_nn.py aoa.csv
    cargo run -p fsim-sim --release --example aoa_nn -- loop aoa.weights.txt
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

path = sys.argv[1] if len(sys.argv) > 1 else "aoa.csv"
d = np.genfromtxt(path, delimiter=",", names=True)
# Every column the Rust side wrote as an input; never wind, turb, or alpha_true.
FEATURES = [f for f in d.dtype.names if f not in ("run", "t", "wind", "turb", "alpha_true")]
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
stem = path.rsplit(".", 1)[0]
plt.savefig(stem + ".png", dpi=120)
print("plot:", stem + ".png")

# Plain-text weights for the Rust forward pass (see Mlp::load in aoa_nn.rs).
scaler, mlp = model[0], model[-1]
sizes = [len(FEATURES)] + [w.shape[1] for w in mlp.coefs_]
nums = [scaler.mean_, scaler.scale_]
for w, b in zip(mlp.coefs_, mlp.intercepts_):
    nums += [w.ravel(), b]
with open(stem + ".weights.txt", "w") as f:
    f.write(" ".join(map(str, sizes)) + "\n")
    f.write("\n".join(repr(float(v)) for v in np.concatenate(nums)) + "\n")
print("weights:", stem + ".weights.txt")
