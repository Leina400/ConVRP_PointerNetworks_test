# fichier prepare_fixed_clients.py
import torch

# Configuration
torch.manual_seed(42)  # pour reproductibilité
num_fixed = 7
max_demand = 9

# Clients fixes NON NORMALISÉS (ne pas diviser par max_load ici)
fixed_clients = {
    'positions': torch.rand(1, 2, num_fixed),
    'demands': torch.randint(1, max_demand + 1, (1, 1, num_fixed))
}

# Sauvegarde
torch.save(fixed_clients, 'fixed_clients_data.pt')
print("Fichier fixed_clients_data.pt généré avec succès")


# Affichage des positions
positions = fixed_clients['positions']
print("Positions des clients fixes (x, y) :")
for i in range(num_fixed):
    x = positions[0, 0, i].item()
    y = positions[0, 1, i].item()
    print(f"  Client {i+1} : x = {x:.3f}, y = {y:.3f}")

# Affichage des demandes
demands = fixed_clients['demands']
print("\nDemandes des clients fixes :")
for i in range(num_fixed):
    d = demands[0, 0, i].item()
    print(f"  Client {i+1} : demande = {d}")
