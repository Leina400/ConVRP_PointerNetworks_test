# fichier model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

# herite de Module
class Encoder(nn.Module):
    """Encodes the static & dynamic states using 1d Convolution."""

    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        # Applique une convolution 1D avec un noyau de taille 1
        # Convertit chaque entrée (de dimension input_size) en un vecteur de taille hidden_size. (projection)
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=1)

    def forward(self, input):
        output = self.conv(input)
        return output  # (batch, hidden_size, seq_len)


class Attention(nn.Module):
    """Calculates attention over the input nodes given the current state."""

    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        
        # vecteur v pour projeter le score d'attention final
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                          device=device, requires_grad=True))

        # W processes features from static decoder elements (matrice des poids)
        self.W = nn.Parameter(torch.zeros((1, hidden_size, 3 * hidden_size),
                                          device=device, requires_grad=True))

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden):
        # decodeur_hidden -> résumé de l’état actuel
        batch_size, hidden_size, _ = static_hidden.size()

        # préparer le vecteur de contexte du décodeur pour qu’il soit compatible avec les encodages 
        hidden = decoder_hidden.unsqueeze(2).expand_as(static_hidden)
        # concatenationn element statiiques et dynamique
        hidden = torch.cat((static_hidden, dynamic_hidden, hidden), 1)
        # Chaque nœud est maintenant représenté par un vecteur contenant sa position encodée, demande, contexte du decodeur

        # Broadcast some dimensions so we can do batch-matrix-multiply
        v = self.v.expand(batch_size, 1, hidden_size)
        W = self.W.expand(batch_size, hidden_size, -1)

        # Elle sert à calculer un score pour chaque nœud
        # transformation linéaire -> dans l'espace d'attention, quels infos ressortent le + ?
        attns = torch.bmm(v, torch.tanh(torch.bmm(W, hidden)))
        # softmax pour avoir un vecteur de proba des infos les + importantes
        attns = F.softmax(attns, dim=2)  # (batch, seq_len)

        # retourn un vecteur de poids d’attention
        return attns
    


class Pointer(nn.Module):
    """Calculates the next state given the previous state and input embeddings."""
    # utilise une RNN (ici, une GRU) et de l’attention pour pointer vers un prochain nœud.

    def __init__(self, hidden_size, num_layers=1, dropout=0.2):
        super(Pointer, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # W et v servent à calculer l'énergie des paires (nœud, contexte) pour choisir où aller.
        # Used to calculate probability of selecting next state
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                          device=device, requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 2 * hidden_size),
                                          device=device, requires_grad=True))

        # Used to compute a representation of the current decoder output
        # La GRU prend une entrée à chaque étape de la tournée (decoder_hidden) et 
        # génère un nouvel état rnn_out + un état caché last_hh
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        
        # pondérer l’attention sur tous les nœuds en fonction de rnn_out.
        self.encoder_attn = Attention(hidden_size)

        # On applique le dropout à la sortie de la GRU, et à last_hh si num_layers == 1.
        self.drop_rnn = nn.Dropout(p=dropout)
        self.drop_hh = nn.Dropout(p=dropout)

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden, last_hh):
        # sert à déterminer quelle ville visiter ensuite

        rnn_out, last_hh = self.gru(decoder_hidden.transpose(2, 1), last_hh)
        rnn_out = rnn_out.squeeze(1)
        # rnn_out est la sortie à l'étape courante


        # Always apply dropout on the RNN output
        rnn_out = self.drop_rnn(rnn_out)
        if self.num_layers == 1:
            # If > 1 layer dropout is already applied
            last_hh = self.drop_hh(last_hh) 

        # Given a summary of the output, find an  input context
        # On calcule les scores d’attention 
        enc_attn = self.encoder_attn(static_hidden, dynamic_hidden, rnn_out)
        # On applique l’attention sur les encodages static_hidden, pour obtenir un contexte vectoriel 
        context = enc_attn.bmm(static_hidden.permute(0, 2, 1))  # (B, 1, num_feats)

        # Calculate the next output using Batch-matrix-multiply ops
        context = context.transpose(1, 2).expand_as(static_hidden)
        energy = torch.cat((static_hidden, context), dim=1)  # (B, num_feats, seq_len)

        v = self.v.expand(static_hidden.size(0), -1, -1)
        W = self.W.expand(static_hidden.size(0), -1, -1)

        # On applique une transformation linéaire suivie de tanh, puis on calcule un produit scalaire avec v pour chaque noeud.
        probs = torch.bmm(v, torch.tanh(torch.bmm(W, energy))).squeeze(1)

        return probs, last_hh


class DRL4TSP(nn.Module):
    """Defines the main Encoder, Decoder, and Pointer combinatorial models.

    Parameters
    ----------
    static_size: int
        Defines how many features are in the static elements of the model
        (e.g. 2 for (x, y) coordinates)
    dynamic_size: int > 1
        Defines how many features are in the dynamic elements of the model
        (e.g. 2 for the VRP which has (load, demand) attributes. The TSP doesn't
        have dynamic elements, but to ensure compatility with other optimization
        problems, assume we just pass in a vector of zeros.
    hidden_size: int
        Defines the number of units in the hidden layer for all static, dynamic,
        and decoder output units.
    update_fn: function or None
        If provided, this method is used to calculate how the input dynamic
        elements are updated, and is called after each 'point' to the input element.
    mask_fn: function or None
        Allows us to specify which elements of the input sequence are allowed to
        be selected. This is useful for speeding up training of the networks,
        by providing a sort of 'rules' guidlines to the algorithm. If no mask
        is provided, we terminate the search after a fixed number of iterations
        to avoid tours that stretch forever ( eviter d'explorer des noeuds deja vu ou valides)
    num_layers: int
        Specifies the number of hidden layers to use in the decoder RNN
    dropout: float
        Defines the dropout rate for the decoder
    """

    def __init__(self, static_size, dynamic_size, hidden_size,
                 update_fn=None, mask_fn=None, num_layers=1, dropout=0.):
        super(DRL4TSP, self).__init__()

        if dynamic_size < 1:
            raise ValueError(':param dynamic_size: must be > 0, even if the '
                             'problem has no dynamic elements')

        self.update_fn = update_fn
        self.mask_fn = mask_fn

        # intialisation
        # Define the encoder & decoder models
        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)
        self.decoder = Encoder(static_size, hidden_size)
        #  décide à chaque étape où aller
        self.pointer = Pointer(hidden_size, num_layers, dropout)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

        # Used as a proxy initial state in the decoder when not specified
        self.x0 = torch.zeros((1, static_size, 1), requires_grad=True, device=device)

    def forward(self, static, dynamic, decoder_input=None, last_hh=None, return_entropy=False):
        """
        Parameters
        ----------
        
        static: Array of size (batch_size, feats, num_cities)
            Defines the elements to consider as static. For the TSP, this could be
            things like the (x, y) coordinates, which won't change
        dynamic: Array of size (batch_size, feats, num_cities)
            Defines the elements to consider as static. For the VRP, this can be
            things like the (load, demand) of each city. If there are no dynamic
            elements, this can be set to None
        decoder_input: Array of size (batch_size, num_feats)
            Defines the outputs for the decoder. Currently, we just use the
            static elements (e.g. (x, y) coordinates), but this can technically
            be other things as well
        last_hh: Array of size (batch_size, num_hidden)
            Defines the last hidden state for the RNN
        """

        batch_size, input_size, sequence_size = static.size()

        # Si pas d’entrée pour le décodeur, on commence au dépôt.
        if decoder_input is None:
            decoder_input = self.x0.expand(batch_size, -1, -1)

        # Always use a mask - if no function is provided, we don't update it
        mask = torch.ones(batch_size, sequence_size, device=device)

        # Structures for holding the output sequences
        tour_idx, tour_logp = [], []
        #   liste des clients visités et log proba de chaque décision
        max_steps = sequence_size if self.mask_fn is None else 1000

        # Static elements only need to be processed once, and can be used across
        # all 'pointing' iterations. When / if the dynamic elements change,
        # their representations will need to get calculated again.
        #  Encodage des entrées
        static_hidden = self.static_encoder(static)
        dynamic_hidden = self.dynamic_encoder(dynamic)
        all_entropy = []

        for _ in range(max_steps):
            # Si aucune ville visitable 
            if not mask.byte().any():
                break

            # ... but compute a hidden rep for each element added to sequence
            # Encodage du décodeur
            decoder_hidden = self.decoder(decoder_input)

            # Produit une distribution probs de taille (B, N) : proba de visiter chaque ville.
            probs, last_hh = self.pointer(static_hidden,
                                          dynamic_hidden,
                                          decoder_hidden, last_hh)
            temperature = 2.0  # ou essaie 2.5
            probs = F.softmax((probs + mask.log()) / temperature, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)  # shape: [B]
            all_entropy.append(entropy)




            max_probs, _ = probs.max(dim=1)
          #  print("Max prob par sample:", max_probs[:10])
            # Affiche les 10 premiers


            # When training, sample the next step according to its probability.
            # During testing, we can take the greedy approach and choose highest
            if self.training:
                m = torch.distributions.Categorical(probs)
                # À chaque étape, le modèle a calculé une distribution probs sur les villes restantes. 

                # Sometimes an issue with Categorical & sampling on GPU; See:
                # https://github.com/pemami4911/neural-combinatorial-rl-pytorch/issues/5
                # Tire un indice (clients) selon la distribution probs.
                ptr = m.sample()
                # si probs = [0.1, 0.3, 0.6], le tirage peut tomber plus souvent sur l’indice 2.


                while not torch.gather(mask, 1, ptr.data.unsqueeze(1)).byte().all():
                    ptr = m.sample()
                # Calcul du log-probabilité
                logp = m.log_prob(ptr)
            else:
                prob, ptr = torch.max(probs, 1)  # Greedy -> on choisit l’action avec la meilleure
                #  prob -> les plus hautes probabilités) pour chaque exemple du batch.
                #  ptr -> les indices correspondants 
                logp = prob.log()
                # logp est utilisé dans le calcul de la loss (policy gradient) 

            # After visiting a node update the dynamic representation
            if self.update_fn is not None:
                # mettre à jour les états dynamiques (load, demand) après avoir visité pointé par ptr
                dynamic = self.update_fn(dynamic, ptr.data)
                dynamic_hidden = self.dynamic_encoder(dynamic)

                # Since we compute the VRP in minibatches, some tours may have
                # number of stops. We force the vehicles to remain at the depot 
                # in these cases, and logp := 0

                # On vérifie si toutes les demandes sont satisfaites (i.e., tournées terminées).
                is_done = dynamic[:, 1].sum(1).eq(0).float()
                logp = logp * (1. - is_done)
                # pour les exemples où la tournée est terminée (is_done = 1), on met leur logp à 0.

            # And update the mask so we don't re-visit if we don't need to
            if self.mask_fn is not None:
                mask = self.mask_fn(mask, dynamic, ptr.data).detach()

            tour_logp.append(logp.unsqueeze(1))
            tour_idx.append(ptr.data.unsqueeze(1))

            decoder_input = torch.gather(static, 2,
                                         ptr.view(-1, 1, 1)
                                         .expand(-1, input_size, 1)).detach()

        tour_idx = torch.cat(tour_idx, dim=1)  # (batch_size, seq_len)
        tour_logp = torch.cat(tour_logp, dim=1)  # (batch_size, seq_len)

        if return_entropy:
            entropy_total = torch.stack(all_entropy, dim=1).sum(dim=1)  # [B]
            return tour_idx, tour_logp, entropy_total
        else:
            return tour_idx, tour_logp


if __name__ == '__main__':
    raise Exception('Cannot be called from main')
