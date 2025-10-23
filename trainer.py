# fichier trainer.py
"""Defines the main trainer model for combinatorial problems
conda activate nazari-vrp

Each task must define the following functions:
* mask_fn: can be None
* update_fn: can be None
* reward_fn: specifies the quality of found solutions
* render_fn: Specifies how to plot found solutions. Can be None

--train-size=1024
--valid-size=128
--batch_size=64
--nodes=10
--hidden_size=128
--num_layers=1
--dropout=0.1
--actor_lr=5e-4
--critic_lr=5e-4
"""

import os
import time
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from model import DRL4TSP, Encoder
from tasks.vrp import render
#from task.consistent_vrp import ConsistentVRPDataset

# python trainer.py --task=convrp --nodes=10 --train-size=512 --valid-size=64 --batch_size=64

# on travaille sur GPU si dispo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu') 
#GAMMA = 10

#GAMMAS = [0, 0.5, 1, 2, 4, 8, 10, 15, 20, 30, 50]



class StateCritic(nn.Module):
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """
    # on definiti la classe StateCritic -> objet qui contient les couches, l'encodeur, decodeur
    # il s'agit d'un reseau convolutif 
    def __init__(self, static_size, dynamic_size, hidden_size):
        super(StateCritic, self).__init__()

        # encode les info statiques : demande (x,y) localisation et dynamique : charge restante
        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)

        # Define the encoder & decoder models
        # defintion des couches 
        self.fc1 = nn.Conv1d(hidden_size * 2, 20, kernel_size=1)
        # transfo non linéaire
        self.fc2 = nn.Conv1d(20, 20, kernel_size=1)
        # production d'un score par noeud
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)

        # transofme (x,y) -> representation vectorielle avec poids et biais 

        # initalisation des poids avec la methoed xavier
        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, static, dynamic):
    # décrit comment l’entrée est transformée pour produire une estimation du coût d’un état.
    # prend en entrée les info statiques et dynamiques des clients sur le noeud courant

        # Use the probabilities of visiting each
        # passage dans un petit CNN
        static_hidden = self.static_encoder(static)
        dynamic_hidden = self.dynamic_encoder(dynamic)
        # Chaque client est représenté par un vecteur hidden_size de features.

        # concatenation des edux encodages (dynamiques + statiques)
        hidden = torch.cat((static_hidden, dynamic_hidden), 1)

        #  transforme les 256 features en 20 features
        output = F.relu(self.fc1(hidden))
        output = F.relu(self.fc2(output))
        # output.shape = [batch_size, 1, n_nodes]
        # on somme les score sur tous les noeuds :
        output = self.fc3(output).sum(dim=2)
        return output
    # C’est le coût estimé pour chaque instance de batch
       

# herite de nn.module et est un petit reseau convolutif
class Critic(nn.Module):
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, hidden_size):
        super(Critic, self).__init__()

        # Define the encoder & decoder models
        self.fc1 = nn.Conv1d(1, hidden_size, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden_size, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)
    # prend en entrée un vecteur [B,N]
    def forward(self, input):

        output = F.relu(self.fc1(input.unsqueeze(1)))
        output = F.relu(self.fc2(output)).squeeze(2)
        output = self.fc3(output).sum(dim=2)
        return output

'''
data_loader : lot de problèmes VRP ou TSP à résoudre (validation set),
actor : le modèle à tester (policy network),
reward_fn : fonction qui mesure la qualité des tournées générées,
render_fn (optionnel) : fonction pour tracer les solutions,
save_dir : dossier où sauver les visualisations,
num_plot : nombre de solutions à visualiser (max).
'''

def validate(data_loader, actor, reward_fn, render_fn=None, save_dir='.',
             num_plot=5, gamma = None):
    """Used to monitor progress on a validation set & optionally plot solution."""

    # met le modele en evaluation
    actor.eval()

    # creer le dossier ou on va visualiser
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # init les recompenses
    rewards = []
    # pour chaque batch du jeu de validation
    for batch_idx, batch in enumerate(data_loader):
        # on recupere les données (position, clien, demandes, pos init)
        static, dynamic, x0 = batch

        # on envoir sur gpu ou cpu
        static = static.to(device)
        dynamic = dynamic.to(device)
        x0 = x0.to(device) if len(x0) > 0 else None

        # execution de la politique sans gradient
        # on evalue juste la politique on veut pas l'ameliorer on est en valdiation
        with torch.no_grad():
            tour_indices, _ = actor.forward(static, dynamic, x0)

        # calcul de la recompense
        reward_output = reward_fn(static, tour_indices)
        reward = reward_output[0] if isinstance(reward_output, tuple) else reward_output
        reward = reward.mean().item()

        # on stocke la moyenne des recompense dans la liste
        rewards.append(reward)

        if render_fn is not None and batch_idx < num_plot:
            name = f'batch{batch_idx}_len{reward:.4f}_gamma{gamma}.png' if gamma is not None else f'batch{batch_idx}_len{reward:.4f}.png'
            path = os.path.join(save_dir, name)
            render_fn(static, tour_indices, path, gamma=gamma)


    # remise en mode entrainement et on renvoie la moyenne des recompenses
    actor.train()
    return np.mean(rewards)


def train(actor, critic, task, num_nodes, train_data, valid_data, reward_fn,
          render_fn, batch_size, actor_lr, critic_lr, max_grad_norm,gamma, **kwargs):
    import datetime, os, time, numpy as np
    from torch.utils.data import DataLoader
    import torch.nn.functional as F

   # save_dir = os.path.join(task, '%d' % num_nodes, str(datetime.datetime.now().time()).replace(':', '_'))

    if 'save_dir' in kwargs and kwargs['save_dir'] is not None:
        save_dir = kwargs['save_dir']
    else:
        now = str(datetime.datetime.now().time()).replace(':', '_')
        save_dir = os.path.join(task, '%d' % num_nodes, now)



    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)

    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size, shuffle=False)

    best_reward = np.inf
   # gamma = kwargs.get("gamma", 100.0)

    # 4 epochs
    num_epoch = 5
    for epoch in range(num_epoch):
        actor.train()
        critic.train()
        times, losses, rewards, critic_rewards = [], [], [], []

        epoch_start = time.time()
        start = epoch_start
        final_conistency_score = []

        for batch_idx, batch in enumerate(train_loader):
            static, dynamic, x0 = batch
            static = static.to(device)
            dynamic = dynamic.to(device)
            x0 = x0.to(device) if x0 is not None else None

 

            tour_indices, tour_logp, entropy = actor(static, dynamic, x0, return_entropy=True)

            reward_output = reward_fn(static, tour_indices)
            '''
            if isinstance(reward_output, tuple):
                reward, consistency_scores = reward_output
            else:
                reward = reward_output
                consistency_scores = torch.zeros_like(reward)
            
            if isinstance(reward_output, tuple):
                reward, consistency_scores = reward_output
                # === Buffer des 10 dernières consistance ===
                last_consistencies.append(consistency_scores.detach().cpu())
                if len(last_consistencies) > 10:
                    last_consistencies.pop(0)
            else:
                reward = reward_output
                consistency_scores = torch.zeros_like(reward)
                '''
            

            if isinstance(reward_output, tuple):
                reward, consistency_scores = reward_output
                if epoch == 4:  # ou epoch == num_epochs - 1 si tu préfères rendre ça général
                    final_conistency_score.append(consistency_scores.detach().cpu())
                    

            critic_est = critic(static, dynamic).view(-1)
            advantage = reward - critic_est

        
            '''
            print(f"[Batch {batch_idx}] Mean reward: {reward.mean().item():.4f} | Mean advantage: {advantage.mean().item():.4f}")
            print(f"[Batch {batch_idx}] Mean consistency: {consistency_scores.mean().item():.4f}")
            '''



            #Normalisation du bonus 
            bonus = consistency_scores.detach()
            bonus = (bonus - bonus.mean()) / (bonus.std() + 1e-5)

            #Bi-objectif : PG + consistance 
            actor_loss_pg = torch.mean(advantage.detach() * tour_logp.sum(dim=1))
            actor_loss_consistency = -gamma * bonus.mean()
            actor_loss = actor_loss_pg + actor_loss_consistency - 0.2 * entropy.mean()

            critic_loss = torch.mean(advantage ** 2)

            # Backpropagation 
            actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            actor_optim.step()

            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_optim.step()

         
            critic_rewards.append(critic_est.mean().item())
            rewards.append(reward.mean().item())
            losses.append(actor_loss.item())

            if (batch_idx + 1) % 100 == 0:
                end = time.time()
                print(f'  Batch {batch_idx+1}/{len(train_loader)}, reward: {np.mean(rewards[-100:]):.3f}, loss: {np.mean(losses[-100:]):.4f}, time: {end - start:.2f}s')
                start = end

        # on prend en compte la consistance sur le dernier entrainement du jour 
        #if last_consistencies:
         #   mean_final_consistency = torch.cat(last_consistencies).mean().item()
          #  print(f"[Résumé Final] Epoch {epoch}, Consistance Moyenne (10 derniers batches) : {mean_final_consistency:.4f}")

        if final_conistency_score:
            mean_final_consistency = torch.cat(final_conistency_score).mean().item()
            print(f"[Résumé Final] Epoch {epoch}, Consistance Moyenne (TOUS les batches) : {mean_final_consistency:.4f}")


        #Validation
        mean_loss = np.mean(losses)
        mean_reward = np.mean(rewards)
        epoch_dir = os.path.join(checkpoint_dir, str(epoch))
        os.makedirs(epoch_dir, exist_ok=True)
        torch.save(actor.state_dict(), os.path.join(epoch_dir, 'actor.pt'))
        torch.save(critic.state_dict(), os.path.join(epoch_dir, 'critic.pt'))

        valid_dir = os.path.join(save_dir, str(epoch))
        mean_valid = validate(valid_loader, actor, reward_fn, render_fn, valid_dir, num_plot=5, gamma = gamma)

        if mean_valid < best_reward:
            best_reward = mean_valid
            torch.save(actor.state_dict(), os.path.join(save_dir, 'actor.pt'))
            torch.save(critic.state_dict(), os.path.join(save_dir, 'critic.pt'))

       
        print(f"Reward: {reward.mean():.4f}, Advantage: {advantage.mean():.4f}, Logp: {tour_logp.sum(dim=1).mean():.4f}, Actor loss: {actor_loss:.4f}")
        print(f'Mean epoch loss/reward: {mean_loss:.4f}, {mean_reward:.4f}, val: {mean_valid:.4f}, time: {time.time() - epoch_start:.2f}s')


# (*)
''''''
def train_vrp(args, fixed_clients_data=None):


    # Goals from paper:
    # VRP10, Capacity 20:  4.84  (Greedy)
    # VRP20, Capacity 30:  6.59  (Greedy)
    # VRP50, Capacity 40:  11.39 (Greedy)
    # VRP100, Capacity 50: 17.23  (Greedy)

    from tasks import vrp
    from tasks.vrp import VehicleRoutingDataset

    # Determines the maximum amount of load for a vehicle based on num nodes
    # capacité max du véhicule, selon le nombre de clients. (VRP 10 -> capacité 20 on essaye 30 max)
    LOAD_DICT = {10: 20, 20: 30, 50: 40, 100: 50}
    # demande max d'un client
    MAX_DEMAND = 9
    STATIC_SIZE = 3 # (x, y) position + flag (ponctuel ou fixe)
    DYNAMIC_SIZE = 2 # (load, demand) reste et demande
    NUM_FIXED = 7

    # Donne la capacité du véhicule pour ce problème.
    max_load = LOAD_DICT[args.num_nodes]

    # on genere les clients fixes une seule fois 
    if fixed_clients_data is None:
        fixed_clients_data = (
            torch.rand(1, 2, NUM_FIXED),
            torch.randint(1, MAX_DEMAND + 1, (1, 1, NUM_FIXED)) / float(max_load)
        )

    # (*) generation des données 
    # Génère train_size et valid_size problèmes VRP avec num_nodes clients
    train_data = VehicleRoutingDataset(args.train_size,
                                       args.num_nodes,
                                       max_load,
                                       MAX_DEMAND,
                                       args.seed,
                                       NUM_FIXED,
                                       fixed_clients_data=fixed_clients_data)
                                

    valid_data = VehicleRoutingDataset(args.valid_size,
                                       args.num_nodes,
                                       max_load,
                                       MAX_DEMAND,
                                       args.seed + 1,
                                       NUM_FIXED,
                                       fixed_clients_data=fixed_clients_data)

    # modele qui propose des tournées
    actor = DRL4TSP(STATIC_SIZE,
                    DYNAMIC_SIZE,
                    args.hidden_size,
                    train_data.update_dynamic,
                    train_data.update_mask,
                    args.num_layers,
                    args.dropout).to(device)

    # evalueateur 
    critic = StateCritic(STATIC_SIZE, DYNAMIC_SIZE, args.hidden_size).to(device)

    # recuperation des arguments
    # kwargs devient un dictionnaire regroupant tout ce qu’il faut passer à train(...).
    kwargs = vars(args)
    kwargs['train_data'] = train_data
    kwargs['valid_data'] = valid_data
    kwargs['reward_fn'] = vrp.reward
    kwargs['render_fn'] = vrp.render

    if args.checkpoint:
        path = os.path.join(args.checkpoint, 'actor.pt')
        actor.load_state_dict(torch.load(path, device))

        path = os.path.join(args.checkpoint, 'critic.pt')
        critic.load_state_dict(torch.load(path, device))

    # entrainement du modèle si test pas activé
    if not args.test:
        train(actor, critic, **kwargs)

    # evaluation sur un test set
    test_data = VehicleRoutingDataset(args.valid_size,
                                      args.num_nodes,
                                      max_load,
                                      MAX_DEMAND,
                                      args.seed + 2)

    test_dir = 'test'
    # genere de nouveaux exemples 
    test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)
    out = validate(test_loader, actor, vrp.reward, vrp.render, test_dir, num_plot=5)
    # moyenne des tournées sur test set
    print('Average tour length: ', out)


def train_convrp(args, gamma, fixed_clients_data=None):
    from tasks.vrp import VehicleRoutingDataset, consistent_reward, reward
    from tasks.consistent_vrp import ConsistentVRPDataset
    from tasks import vrp
    from model import DRL4TSP
    from torch.utils.data import DataLoader
    import torch.nn.functional as F
    import os

    import torch
    import numpy as np
    import random


    def set_global_seeds(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    set_global_seeds(args.seed)

    start_global = time.time()
    final_mean_cons = None

    #log_file = open("training_log_convrp.txt", "w")
    log_file = open(f"training_log_gamma_{gamma}.txt", "w")




    LOAD_DICT = {10: 20, 20: 30, 50: 40, 100: 50}
    MAX_DEMAND = 9
    STATIC_SIZE = 3
    DYNAMIC_SIZE = 2
    NUM_DAYS = 5
    NUM_FIXED = args.num_fixed

  
    if fixed_clients_data is None:
        fixed_clients_data = (
            torch.rand(1, 2, NUM_FIXED),
            torch.randint(1, MAX_DEMAND + 1, (1, 1, NUM_FIXED)) / float(LOAD_DICT[args.num_nodes])
        )

    print(f"[γ={gamma}] Positions clients fixes :")
    print(fixed_clients_data[0])
    print(f"[γ={gamma}] Demandes clients fixes :")
    print(fixed_clients_data[1])





    full_dataset = ConsistentVRPDataset(num_days=NUM_DAYS, num_samples_per_day=args.train_size,
                                        input_size=args.num_nodes,
                                        max_load=LOAD_DICT[args.num_nodes],
                                        max_demand=MAX_DEMAND,
                                        seed=args.seed,
                                        num_fixed=NUM_FIXED,
                                        fixed_clients_data = fixed_clients_data)

    previous_tours = None

    print(f"[γ={gamma}] 2Positions clients fixes :")
    print(fixed_clients_data[0])
    print(f"[γ={gamma}] 2Demandes clients fixes :")
    print(fixed_clients_data[1])



    actor = DRL4TSP(STATIC_SIZE, DYNAMIC_SIZE, args.hidden_size,
                    full_dataset[0].update_dynamic, full_dataset[0].update_mask,
                    args.num_layers, args.dropout).to(device)

    critic = StateCritic(STATIC_SIZE, DYNAMIC_SIZE, args.hidden_size).to(device)

    for day in range(NUM_DAYS):
        
        print(f"=== JOUR {day}  et CAPACITE MAX ===")
        train_data = full_dataset[day]
        valid_data = VehicleRoutingDataset(
            args.valid_size,
            args.num_nodes,
            max_load=LOAD_DICT[args.num_nodes],
            max_demand=MAX_DEMAND,
            seed=args.seed + 100 + day,
            num_fixed=NUM_FIXED,
            fixed_clients_data=full_dataset.datasets[day].fixed_clients_data
        )

        # initialises un nouvel actor et critic à chaque jour



        #  On collectera les tournées du jour pour la consistance
        collected_tours = []
        '''
        def reward_with_consistency(static, tour_indices):
            total_reward, consistency_scores = consistent_reward(static, tour_indices, previous_tours, gamma=GAMMA, return_consistency=True)

            if (day > 0):
                mean_consistency = consistency_scores.mean().item()
                tour_lengths = reward(static, tour_indices)
                avg_tour_length = tour_lengths.mean().item()
                print(f"\tMoyenne arcs communs (jour {day}): {mean_consistency:.4f} et gamma = {GAMMA} ")
                log_file.write(f"[JOUR {day}] | gamma= {GAMMA} | Moyenne arcs communs: {mean_consistency:.4f}\n")
            return total_reward
            
    '''
        def reward_with_consistency(static, tour_indices):
            nonlocal final_mean_cons
            total_reward, consistency_scores = consistent_reward(
                static,
                tour_indices,
                previous_tours,
                gamma=gamma,
                return_consistency=True
            )


            # penalite non linéaire
            penalite = gamma*(1-consistency_scores)**2
            reward_penalite = total_reward + penalite

            
            mean_consistency = consistency_scores.mean().item()
            final_mean_cons = mean_consistency
            avg_tour_length = reward(static, tour_indices).mean().item()
            if day > 0:
                print(f"\tMoyenne arcs communs (jour {day}): {mean_consistency:.4f} et gamma = {gamma}")
            log_file.write(f"[JOUR {day}] | gamma= {gamma} | Moyenne arcs communs: {mean_consistency:.4f}\n")


            return reward_penalite, consistency_scores



        train(actor, critic,
            task='convrp',
            num_nodes=args.num_nodes,
            train_data=train_data,
            valid_data=valid_data,
            reward_fn=reward_with_consistency,
            render_fn=lambda static, tours, path, **kwargs: render(static, tours, path, num_fixed=NUM_FIXED, **kwargs),
            batch_size=args.batch_size,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            max_grad_norm=args.max_grad_norm,
            gamma = gamma)

        #  Évaluation sur le set d'entraînement 
        train_loader = DataLoader(train_data, args.batch_size, False, num_workers=0)
        train_score = validate(train_loader, actor, reward, render_fn=vrp.render, save_dir=f'day_{day}_train', num_plot=5, gamma=gamma)
        print(f"[JOUR {day}] Moyenne des tournées entraînement : {train_score:.4f}")
        #log_file.write(f"[JOUR {day}] Moyenne des tournées entraînement : {train_score:.4f}\n")

        import torch.nn.functional as F

        all_tours = []
        max_len = 0
        test_loader = DataLoader(valid_data, args.batch_size, False, num_workers=0)

        # 1. Collecte des tournées et calcul de la longueur max
        with torch.no_grad():
            for batch in test_loader:
                static, dynamic, x0 = batch
                tour_indices, _ = actor(static.to(device), dynamic.to(device), x0.to(device))
                all_tours.append(tour_indices.cpu())
                max_len = max(max_len, tour_indices.size(1))

        # 2. Padding des tournées avec -1
        padded_tours = []
        for t in all_tours:
            if t.size(1) < max_len:
                t = F.pad(t, (0, max_len - t.size(1)), value=-1)
            padded_tours.append(t)

        # 3. Concaténation des tournées
        previous_tours = torch.cat(padded_tours, dim=0)  # (B, max_len)

        '''
        # === Générer les tournées de test ===
        test_loader = DataLoader(valid_data, args.batch_size, False, num_workers=0)
        all_tours = []
        with torch.no_grad():
            for batch in test_loader:
                static, dynamic, x0 = batch
                tour_indices, _ = actor(static.to(device), dynamic.to(device), x0.to(device))
                all_tours.append(tour_indices.cpu())

        #On garde en mémoire les tournées du jour d pour les utiliser comme référence au jour d+1
        previous_tours = torch.cat(all_tours, dim=0)  # pour jour suivant
        '''

        out = validate(test_loader, actor, vrp.reward, vrp.render, f'day_{day}_test', num_plot=5, gamma=gamma)
        print(f'[JOUR {day}] Moyenne des tournées test :', out)
        print("GAMMA = ", gamma)
        log_file.write(f"[JOUR {day}] Moyenne des tournées test : {out:.4f}\n") # Taille validation: {args.valid_size}\n")

    log_file.close()

    # Résumé texte sauvegardé dans dossier
    summary_dir = "summaries"
    os.makedirs(summary_dir, exist_ok=True)
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_path = os.path.join(summary_dir, f"summary_gamma_{gamma}_{timestamp}.txt")

    with open(summary_path, "w") as f:
        f.write("Résumé entraînement conVRP\n")
        f.write(f"Gamma utilisé         : {gamma}\n")
        f.write(f"Durée totale (s)      : {time.time() - start_global:.2f}\n")
        f.write(f"Jour final            : {NUM_DAYS - 1}\n")
        f.write(f"Consistance finale    : {final_mean_cons:.4f}\n")
        f.write(f"Longueur moyenne test : {out:.4f}\n")

    print(f"\nRésumé sauvegardé dans {summary_path}")



# exemple : python trainer.py --task=vrp --nodes=10 --train-size=512 --valid-size=64 --batch_size=64
#  Selon --task=tsp ou --task=vrp, appelle train_tsp(args) ou train_vrp(args)
# recuperer les args en ligne de commande lors de l'execution
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Combinatorial Optimization')
    parser.add_argument('--seed', default=12345, type=int)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--task', default='tsp')
    parser.add_argument('--nodes', dest='num_nodes', default=20, type=int)
    parser.add_argument('--actor_lr', default=5e-4, type=float)
    parser.add_argument('--critic_lr', default=5e-4, type=float)
    parser.add_argument('--max_grad_norm', default=2., type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--layers', dest='num_layers', default=1, type=int)
    parser.add_argument('--train-size',default=1000000, type=int)
    parser.add_argument('--valid-size', default=1000, type=int)


    parser.add_argument('--save-dir', default=None, type=str, help='Custom save directory')


    # ===
    parser.add_argument('--num-fixed', default=7, type=int)
    parser.add_argument('--gamma', default=0, type=float)


    args = parser.parse_args()

   # GAMMA = args.gamma


    #print('NOTE: SETTTING CHECKPOINT: ')
    #args.checkpoint = os.path.join('vrp', '10', '12_59_47.350165' + os.path.sep)
    #print(args.checkpoint)

    if args.task == 'tsp':
        train_tsp(args)
    elif args.task == 'vrp':
        train_vrp(args)
    elif args.task == 'convrp':
        train_convrp(args, args.gamma)

    else:
        raise ValueError('Task <%s> not understood'%args.task)
    

