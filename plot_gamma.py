import os
import glob
import re
import matplotlib.pyplot as plt

# 1. Lire tous les fichiers de log gamma
log_files = sorted(glob.glob("training_log_gamma_*.txt"))
results = {}

# plot les gammas mais necessite des les changer a chaque fois manuellement -> a optimiser si temps (np.logspace, linspace...)
target_gammas = {0, 1, 1.5, 2, 2.5, 3, 4, 4.5, 5, 6, 7, 7.5, 8, 8.5, 9, 10}

for filename in log_files:
    gamma_match = re.search(r'gamma_(\d+\.?\d*)', filename)
    if not gamma_match:
        continue
    gamma = float(gamma_match.group(1))
    if gamma not in target_gammas:
        continue
    
    with open(filename, "r") as f:
        lines = f.readlines()

    # Extraire les 10 dernières valeurs de consistance
    consistency_lines = [line for line in lines if "Moyenne arcs communs" in line]
    last_consistency_values = consistency_lines[-10:]

    consistencies = []
    for line in last_consistency_values:
        value = float(re.findall(r": (\d+\.\d+)", line)[0])
        consistencies.append(value)

    # Extraire les longueurs de tournées test
    tour_lengths = []
    for line in lines:
        if "Moyenne des tournées test" in line:
            value = float(re.findall(r": (\d+\.\d+)", line)[0])
            tour_lengths.append(value)

    if consistencies and tour_lengths:
        results[gamma] = {
            "mean_consistency": sum(consistencies) / len(consistencies),  # entre 0 et 1
            "mean_test_length": sum(tour_lengths) / len(tour_lengths)
        }

# 2. Trier les résultats par gamma
gammas = sorted(results.keys())
consistencies = [results[g]["mean_consistency"] for g in gammas]
consistencies_pct = [100 * c for c in consistencies]  # pourcentage
tour_lengths = [results[g]["mean_test_length"] for g in gammas]

# 3. Figure 1 : Gamma → Consistance (%)
plt.figure(figsize=(7, 4))
plt.plot(gammas, consistencies_pct, 'o-', label="Consistance moyenne (%)", color='blue')
plt.xlabel("Gamma (poids de la consistance)")
plt.ylabel("Consistance moyenne (%)")
plt.title("Consistance en fonction de Gamma")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("gamma_vs_consistency.png", dpi=150)

# 4. Figure 2 : Gamma → Longueur des tournées
plt.figure(figsize=(7, 4))
plt.plot(gammas, tour_lengths, 's-', label="Longueur moyenne des tournées test", color='orange')
plt.xlabel("Gamma (poids de la consistance)")
plt.ylabel("Longueur moyenne des tournées test")
plt.title("Longueur des tournées en fonction de Gamma")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("gamma_vs_tour_length.png", dpi=150)

# 5. Figure 3 : Front de Pareto
# Crée les points (incohérence, longueur, gamma)
points = [(1 - r["mean_consistency"], r["mean_test_length"], gamma) for gamma, r in results.items()]

def is_dominated(p, others):
    return any(
        (q[0] <= p[0] and q[1] <= p[1]) and (q[0] < p[0] or q[1] < p[1])
        for q in others
    )

pareto_points = [p for p in points if not is_dominated(p, points)]
pareto_points.sort()

# Conversion en pourcentage pour l’axe des incohérences
x_all = [100 * p[0] for p in points]
y_all = [p[1] for p in points]
x_pareto = [100 * p[0] for p in pareto_points]
y_pareto = [p[1] for p in pareto_points]

plt.figure(figsize=(7, 5))
plt.scatter(x_all, y_all, color='lightgray', label='Solutions explorées')
plt.plot(x_pareto, y_pareto, 'ro-', label='Front de Pareto')

for x, y, gamma in zip(x_pareto, y_pareto, [p[2] for p in pareto_points]):
    plt.text(x + 0.3, y + 0.03, f"γ={gamma}", fontsize=8)

plt.xlabel("Incohérence (1 - consistance) [%]")
plt.ylabel("Longueur moyenne des tournées test")
plt.title("Front de Pareto : compromis consistance / efficacité")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("pareto_front.png", dpi=150)

plt.show()
