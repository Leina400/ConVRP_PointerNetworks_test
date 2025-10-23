import os
import glob
import re
import matplotlib.pyplot as plt

#  Lire tous les fichiers de log gamma
log_files = sorted(glob.glob("training_log_gamma_*.txt"))

results = {}

for filename in log_files:
    gamma_match = re.search(r'gamma_(\d+\.?\d*)', filename)
    if not gamma_match:
        continue
    gamma = float(gamma_match.group(1))
    with open(filename, "r") as f:
        lines = f.readlines()

    consistencies = []
    tour_lengths = []

    for line in lines:
        if "Moyenne arcs communs" in line:
            value = float(re.findall(r": (\d+\.\d+)", line)[0])
            consistencies.append(value)
        elif "Moyenne des tournées test" in line:
            value = float(re.findall(r": (\d+\.\d+)", line)[0])
            tour_lengths.append(value)

    if consistencies and tour_lengths:
        results[gamma] = {
            "mean_consistency": sum(consistencies) / len(consistencies),
            "mean_test_length": sum(tour_lengths) / len(tour_lengths)
        }

# 2. Construire les points
points = [(1 - r["mean_consistency"], r["mean_test_length"], gamma) for gamma, r in results.items()]

# 3. Déterminer le front de Pareto
def is_dominated(p, others):
    return any(
        (q[0] <= p[0] and q[1] <= p[1]) and (q[0] < p[0] or q[1] < p[1])
        for q in others
    )

pareto_points = [p for p in points if not is_dominated(p, points)]
pareto_points.sort()  # trier par incohérence croissante

# 4. Tracer
plt.figure(figsize=(7, 5))

# Tracer tous les points (en fond)
x_all = [p[0] for p in points]
y_all = [p[1] for p in points]
plt.scatter(x_all, y_all, color='lightgray', label='Solutions explorées')

# Tracer le front de Pareto
x_pareto = [p[0] for p in pareto_points]
y_pareto = [p[1] for p in pareto_points]
plt.plot(x_pareto, y_pareto, 'ro-', label='Front de Pareto')

# Annoter les gamma
for x, y, gamma in pareto_points:
    plt.text(x + 0.0003, y + 0.03, f"γ={gamma}", fontsize=8)

plt.xlabel("Incohérence (1 - consistance moyenne)")
plt.ylabel("Longueur moyenne des tournées test")
plt.title("Front de Pareto : compromis consistance / efficacité")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("pareto_front.png", dpi=150)
plt.show()
