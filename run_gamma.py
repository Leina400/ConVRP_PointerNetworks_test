import torch
import argparse
from trainer import train_convrp
from tasks.consistent_vrp import ConsistentVRPDataset

# permettre de tester sur de sgammas differents

torch.manual_seed(30)  # pour reproductibilité

num_fixed = 7
num_nodes = 10
train_size = 256
valid_size = 64
batch_size = 12   # tester ensuite avec 12
seed = 30
gammas = [0.5, 1, 1.5, 2, 2.5, 3, 4, 4.5, 5, 6, 7, 7.5, 8, 8.5, 9, 10, 20]
max_demand = 9
max_load = 20  # dépend de num_nodes
num_days = 5

#génération unique des clients fixes
fixed_positions = torch.rand(1, 2, num_fixed)
fixed_demands = torch.randint(1, max_demand + 1, (1, 1, num_fixed)) / float(max_load)
fixed_clients_data = (fixed_positions, fixed_demands)


# Affichage des positions

print("Positions des clients fixes (x, y) :")
for i in range(num_fixed):
    x = fixed_positions[0, 0, i].item()
    y = fixed_positions[0, 1, i].item()
    print(f"  Client {i+1} : x = {x:.3f}, y = {y:.3f}")


#Prépare les arguments fixes
class Args:
    def __init__(self):
        self.seed = seed
        self.task = "convrp"
        self.num_nodes = num_nodes
        self.train_size = train_size
        self.valid_size = valid_size
        self.batch_size = batch_size
        self.hidden_size = 128
        self.num_layers = 1
        self.dropout = 0.1
        self.actor_lr = 5e-4
        self.critic_lr = 5e-4
        self.max_grad_norm = 2.
        self.num_fixed = num_fixed
        self.test = False
        self.checkpoint = None
        self.save_dir = None  # ou "results_gamma"

args = Args()

# Boucle sur les gamma
for gamma in gammas:
    print(f"\n====== Entraînement avec gamma = {gamma} ======\n")
    dataset = ConsistentVRPDataset(
        num_days=num_days,
        num_samples_per_day=train_size,
        input_size=num_nodes,
        max_load=max_load,
        max_demand=max_demand,
        seed=seed,
        num_fixed=num_fixed,
        fixed_clients_data=fixed_clients_data
    )

    train_convrp(args, gamma, fixed_clients_data=fixed_clients_data)


