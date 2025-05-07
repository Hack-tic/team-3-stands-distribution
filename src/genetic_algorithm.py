import random
import time
import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict
from src.models import Entreprise, Salle

class OrganisateurStandsGenetique:
    def __init__(self, entreprises: List[Entreprise], salles: List[Salle], emplacements_fixes: Dict[str, str] = None):
        self.entreprises = entreprises
        self.salles = salles
        self.emplacements_fixes = emplacements_fixes or {}
        
        # Précalculer les concurrents pour optimisation
        self.concurrents_indices = self._precalculer_concurrents()
        
        # Convertir les noms de salles fixes en indices
        self.indices_fixes = {}
        if self.emplacements_fixes:
            for nom_entreprise, nom_salle in self.emplacements_fixes.items():
                # Trouver l'indice de l'entreprise
                idx_entreprise = next((i for i, e in enumerate(self.entreprises) 
                                  if e.nom == nom_entreprise), None)
                
                # Trouver l'indice de la salle
                idx_salle = next((i for i, s in enumerate(self.salles) 
                             if s.nom == nom_salle), None)
                
                if idx_entreprise is not None and idx_salle is not None:
                    self.indices_fixes[idx_entreprise] = idx_salle
                else:
                    if idx_entreprise is None:
                        print(f"Attention: Entreprise fixe '{nom_entreprise}' non trouvée dans la liste des entreprises")
                    if idx_salle is None:
                        print(f"Attention: Salle fixe '{nom_salle}' non trouvée dans la liste des salles")
        
        # Paramètres optimisés pour des performances ultimes
        self.taille_population = 300       # Augmenté pour plus de diversité génétique
        self.nb_generations = 500          # Augmenté pour une meilleure convergence
        self.taux_mutation_initial = 0.5   # Mutation initiale plus élevée pour l'exploration
        self.taux_mutation_final = 0.05    # Mutation finale plus faible pour le raffinement
        self.taille_tournoi = 10           # Tournoi plus compétitif
        self.proportion_elite = 0.2        # Augmentation de la proportion d'élite pour conserver les meilleures solutions
        self.historique_scores = []        # Pour tracer l'évolution
        
        # Précalculer les poids de taille pour optimisation
        self.poids_taille = {'grande': 1.8, 'moyenne': 1.2, 'petite': 0.7}
        
    def _precalculer_concurrents(self) -> List[Set[int]]:
        """Précalcule les indices des entreprises concurrentes pour chaque entreprise"""
        concurrents = []
        for i, e1 in enumerate(self.entreprises):
            # Pour chaque entreprise, stocker les indices de ses concurrents
            concurrent_indices = set()
            for j, e2 in enumerate(self.entreprises):
                if e2.nom in e1.concurrent_de:
                    concurrent_indices.add(j)
            concurrents.append(concurrent_indices)
        return concurrents
        
    def creer_individu(self) -> List[int]:
        """Crée un placement aléatoire (chromosome) en tenant compte des capacités et des emplacements fixes"""
        placements = []
        occupation = [0] * len(self.salles)
        
        # Placer d'abord les entreprises fixes et mettre à jour l'occupation
        for idx_entreprise, idx_salle in self.indices_fixes.items():
            occupation[idx_salle] += 1
        
        # Trier les entreprises par taille et popularité (combinées)
        # Prioriser les grandes entreprises populaires
        indices_non_fixes = [i for i in range(len(self.entreprises)) if i not in self.indices_fixes]
        indices_tries = sorted(indices_non_fixes, 
                              key=lambda i: (-self.poids_taille[self.entreprises[i].taille], 
                                            -self.entreprises[i].popularite_info))
        
        for idx in indices_tries:
            # Préférer les salles avec la meilleure position_qualite qui ont encore de la place
            salles_disponibles = [(i, self.salles[i].position_qualite) 
                                 for i in range(len(self.salles)) 
                                 if occupation[i] < self.salles[i].capacite]
            
            if not salles_disponibles:  # Si toutes pleines, prendre toutes les salles
                salles_disponibles = [(i, self.salles[i].position_qualite) for i in range(len(self.salles))]
            
            # Vérifier les concurrents déjà placés
            poids_concurrents = np.ones(len(self.salles))
            for concurrent_idx in self.concurrents_indices[idx]:
                if concurrent_idx in self.indices_fixes:
                    # Pénaliser fortement la salle où se trouve le concurrent fixe
                    salle_concurrent = self.indices_fixes[concurrent_idx]
                    poids_concurrents[salle_concurrent] = 0.001
                    
            # Pondérer le choix en fonction de plusieurs facteurs:
            # 1. Qualité de la position (position_qualite)
            # 2. Taille de l'entreprise (grandes entreprises privilégiées pour bonnes positions)
            # 3. Popularité de l'entreprise
            # 4. Capacité restante des salles
            # 5. Éviter de placer des concurrents ensemble
            
            poids = []
            for s, qualite in salles_disponibles:
                # Plus la capacité restante est grande, mieux c'est
                capacite_restante = max(1, self.salles[s].capacite - occupation[s])
                
                # Score de combinaison taille/popularité
                score_entreprise = self.poids_taille[self.entreprises[idx].taille] * self.entreprises[idx].popularite_info
                
                # Le poids final combine tous les facteurs
                poids_final = (qualite * score_entreprise * capacite_restante * poids_concurrents[s])
                poids.append(poids_final)
            
            # Normaliser pour éviter une prédominance excessive d'un facteur
            if sum(poids) > 0:
                poids = np.array(poids) / sum(poids)
                indices_salles = [s[0] for s in salles_disponibles]
                salle_choisie = np.random.choice(indices_salles, p=poids)
            else:
                # Fallback: choisir aléatoirement
                salle_choisie = random.choice([s[0] for s in salles_disponibles])
            
            placements.append(salle_choisie)
            occupation[salle_choisie] += 1
            
        # Créer le chromosome complet
        resultat = [0] * len(self.entreprises)
        
        # Placer les entreprises fixes à leur place désignée
        for idx_entreprise, idx_salle in self.indices_fixes.items():
            resultat[idx_entreprise] = idx_salle
        
        # Placer les autres entreprises selon l'ordre de placement
        idx_placement = 0
        for i in range(len(self.entreprises)):
            if i not in self.indices_fixes:
                resultat[i] = placements[idx_placement]
                idx_placement += 1
                
        return resultat

    def calculer_fitness(self, chromosome: List[int]) -> float:
        """Fonction d'évaluation optimisée avec des pondérations améliorées et des calculs plus efficaces"""
        score = 0.0
        
        # Vérifier les emplacements fixes (contrainte absolue)
        for idx_entreprise, idx_salle in self.indices_fixes.items():
            if chromosome[idx_entreprise] != idx_salle:
                # Pénalité massive qui rend la solution non viable
                return -100000
        
        # Structures de données pour l'analyse
        occupation_salles = defaultdict(int)
        types_par_salle = {i: {'grande': 0, 'moyenne': 0, 'petite': 0} for i in range(len(self.salles))}
        secteurs_par_salle = defaultdict(lambda: defaultdict(int))
        entreprises_par_salle = defaultdict(list)
        
        # Première passe: collecter les données
        for i, salle_index in enumerate(chromosome):
            entreprise = self.entreprises[i]
            occupation_salles[salle_index] += 1
            types_par_salle[salle_index][entreprise.taille] += 1
            secteurs_par_salle[salle_index][entreprise.secteur] += 1
            entreprises_par_salle[salle_index].append(i)
        
        # 1. Calcul des pénalités de capacité (critère primordial)
        penalite_capacite = 0
        for salle_id, occupation in occupation_salles.items():
            salle = self.salles[salle_id]
            if occupation > salle.capacite:
                surcharge = occupation - salle.capacite
                # Pénalité exponentielle en fonction de la surcharge
                penalite_capacite += 300 * surcharge**2.5
        
        score -= penalite_capacite
        
        # 2. Pénalité pour les concurrents dans la même salle
        penalite_concurrents = 0
        total_concurrents = 0
        for salle_id, entreprises_indices in entreprises_par_salle.items():
            for idx, i in enumerate(entreprises_indices):
                for j in entreprises_indices[idx+1:]:
                    if j in self.concurrents_indices[i]:
                        # Pénalité plus forte pour les concurrents
                        penalite_concurrents += 800
                        total_concurrents += 1
        
        score -= penalite_concurrents
        
        # 3. Bonus pour le placement optimal selon la taille et la popularité
        bonus_placement = 0
        for i, salle_id in enumerate(chromosome):
            entreprise = self.entreprises[i]
            salle = self.salles[salle_id]
            
            # Plus la salle est bien placée et l'entreprise populaire/grande, plus le bonus est élevé
            importance = self.poids_taille[entreprise.taille] * (1 + entreprise.popularite_info)
            bonus_placement += salle.position_qualite * importance * 3
        
        score += bonus_placement
        
        # 4. Bonus pour répartition équilibrée des types d'entreprises par salle
        bonus_equilibre = 0
        for salle_id, types in types_par_salle.items():
            # Plus l'équilibre est bon (variance faible), plus le bonus est important
            counts = [types['grande'], types['moyenne'], types['petite']]
            if sum(counts) > 0:  # Éviter division par zéro
                # Écart-type normalisé = écart-type / moyenne (coefficient de variation)
                mean = sum(counts) / 3
                if mean > 0:
                    stdev = np.std(counts)
                    cv = stdev / mean  # Coefficient de variation
                    # Moins le coefficient de variation est élevé, plus le bonus est important
                    bonus_equilibre += 50 / (1 + cv**2)
        
        score += bonus_equilibre
        
        # 5. Bonus pour diversité des secteurs par salle (évite la concentration de secteurs)
        bonus_diversite = 0
        for salle_id, secteurs in secteurs_par_salle.items():
            # Bonus proportionnel au nombre de secteurs différents
            diversite = len(secteurs)
            bonus_diversite += 25 * diversite
            
            # Pénalité pour concentration excessive d'un même secteur
            max_secteur_count = max(secteurs.values()) if secteurs else 0
            total_entreprises_salle = sum(secteurs.values())
            if total_entreprises_salle > 0:
                # Si plus d'un tiers des entreprises sont du même secteur
                ratio_concentration = max_secteur_count / total_entreprises_salle
                if ratio_concentration > 0.33:
                    excedent = ratio_concentration - 0.33
                    bonus_diversite -= 150 * excedent * total_entreprises_salle
        
        score += bonus_diversite
        
        # 6. Bonus global pour certaines conditions spéciales
        # Si aucun concurrent dans la même salle
        if total_concurrents == 0:
            score += 2000
        
        # Bonus pour utilisation optimale des salles (éviter salles vides ou surpeuplées)
        ecart_type_occupation = np.std([occupation_salles[i] for i in range(len(self.salles))])
        score -= 50 * ecart_type_occupation  # Pénaliser forte dispersion d'occupation
        
        return score
        
    def selection(self, population: List[List[int]], scores: List[float]) -> List[List[int]]:
        """Sélection par tournoi améliorée avec élitisme et pression sélective ajustée"""
        # Conservation des meilleurs individus (élitisme)
        nb_elites = int(self.proportion_elite * len(population))
        indices_tries = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        elites = [population[i] for i in indices_tries[:nb_elites]]
        
        # Sélection par tournoi avec probabilité proportionnelle au rang
        nouvelle_population = elites.copy()
        
        # Version améliorée du tournoi pour mieux préserver la diversité
        while len(nouvelle_population) < len(population):
            # Utiliser deux tournois et choisir au hasard entre les gagnants
            # pour maintenir la diversité génétique
            tournoi1 = random.sample(range(len(population)), self.taille_tournoi)
            tournoi2 = random.sample(range(len(population)), self.taille_tournoi)
            
            gagnant1 = max(tournoi1, key=lambda i: scores[i])
            gagnant2 = max(tournoi2, key=lambda i: scores[i])
            
            # Si les gagnants sont identiques, prendre le meilleur
            if gagnant1 == gagnant2:
                gagnant = gagnant1
            else:
                # Sinon, choisir avec une probabilité biaisée vers le meilleur
                prob_gagnant1 = scores[gagnant1] / (scores[gagnant1] + scores[gagnant2])
                gagnant = gagnant1 if random.random() < prob_gagnant1 else gagnant2
            
            nouvelle_population.append(population[gagnant])
        
        return nouvelle_population

    def croisement(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Croisement amélioré avec stratégies multiples"""
        taille = len(parent1)
        if taille <= 2:
            return parent1[:], parent2[:]
            
        # Stratégie dynamique de croisement basée sur la diversité génétique
        # Choisir aléatoirement entre croisement à un point, deux points, et uniforme
        strategie = random.choice(['un_point', 'deux_points', 'uniforme'])
        
        # Version améliorée qui respecte les emplacements fixes
        enfant1 = parent1.copy()
        enfant2 = parent2.copy()
        
        # Obtenir les indices modifiables (non fixes)
        indices_modifiables = [i for i in range(taille) if i not in self.indices_fixes]
        
        if not indices_modifiables:
            # Si tous les emplacements sont fixes, aucun croisement n'est possible
            return parent1[:], parent2[:]
        
        if strategie == 'un_point':
            # Choix du point de coupure (uniquement parmi les indices modifiables)
            if len(indices_modifiables) >= 2:
                point_idx = random.randrange(len(indices_modifiables) - 1) + 1
                point = indices_modifiables[point_idx]
                
                # Échanger les segments après le point
                for i in indices_modifiables:
                    if i >= point:
                        enfant1[i], enfant2[i] = enfant2[i], enfant1[i]
                        
        elif strategie == 'deux_points':
            # Choix des points de coupure
            if len(indices_modifiables) >= 3:
                points_idx = sorted(random.sample(range(1, len(indices_modifiables)), 2))
                points = [indices_modifiables[idx] for idx in points_idx]
                
                # Échanger le segment entre les deux points
                for i in indices_modifiables:
                    if points[0] <= i < points[1]:
                        enfant1[i], enfant2[i] = enfant2[i], enfant1[i]
                        
        elif strategie == 'uniforme':
            # Croisement uniforme avec probabilité
            for i in indices_modifiables:
                if random.random() < 0.5:
                    enfant1[i], enfant2[i] = enfant2[i], enfant1[i]
        
        return enfant1, enfant2
    
    def mutation(self, chromosome: List[int], taux_mutation: float):
        """Mutation adaptative améliorée avec plusieurs stratégies et intensité variable"""
        # Mutation adaptative qui respecte les emplacements fixes
        if random.random() < taux_mutation:
            # Obtenir les indices des entreprises modifiables (non fixes)
            indices_modifiables = [i for i in range(len(chromosome)) if i not in self.indices_fixes]
            
            # Si aucun indice n'est modifiable, pas de mutation possible
            if not indices_modifiables:
                return
            
            # Sélectionner une stratégie de mutation avec pondération
            strategies = {
                'permutation': 0.5,    # Permuter deux entreprises
                'déplacement': 0.3,    # Déplacer une entreprise à une nouvelle salle
                'inversion': 0.1,      # Inverser un segment
                'scramble': 0.1        # Mélanger un segment
            }
            
            # Choisir stratégie selon probabilités
            type_mutation = random.choices(
                list(strategies.keys()), 
                weights=list(strategies.values()), 
                k=1
            )[0]
            
            if type_mutation == 'permutation' and len(indices_modifiables) >= 2:
                # Permutation de deux positions
                i, j = random.sample(indices_modifiables, 2)
                chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
                
            elif type_mutation == 'déplacement' and indices_modifiables:
                # Déplacer 1-3 entreprises vers de nouvelles salles
                nb_entreprises = min(len(indices_modifiables), random.randint(1, 3))
                for _ in range(nb_entreprises):
                    i = random.choice(indices_modifiables)
                    # Éviter de déplacer vers la même salle
                    nouvelle_salle = random.randint(0, len(self.salles) - 1)
                    while nouvelle_salle == chromosome[i] and len(self.salles) > 1:
                        nouvelle_salle = random.randint(0, len(self.salles) - 1)
                    chromosome[i] = nouvelle_salle
                    indices_modifiables.remove(i)  # Ne pas sélectionner deux fois
                
            elif type_mutation == 'inversion' and len(indices_modifiables) >= 2:
                # Créer des segments inversibles continus (sans emplacements fixes)
                segments = []
                debut = None
                for i in range(len(chromosome)):
                    if i in indices_modifiables:
                        if debut is None:
                            debut = i
                    elif debut is not None:
                        if i - debut > 1:  # Segment d'au moins 2 éléments
                            segments.append((debut, i - 1))
                        debut = None
                
                # Ajouter le dernier segment si nécessaire
                if debut is not None and len(chromosome) - debut > 1:
                    segments.append((debut, len(chromosome) - 1))
                
                if segments:
                    debut, fin = random.choice(segments)
                    chromosome[debut:fin+1] = reversed(chromosome[debut:fin+1])
                    
            elif type_mutation == 'scramble' and len(indices_modifiables) >= 3:
                # Mélanger un segment aléatoire
                # Sélectionner une section contiguë d'indices modifiables
                indices_modifiables.sort()
                sequences = []
                seq_start = 0
                
                for i in range(1, len(indices_modifiables)):
                    if indices_modifiables[i] != indices_modifiables[i-1] + 1:
                        if i - seq_start >= 3:  # Au moins 3 positions contiguës
                            sequences.append((seq_start, i-1))
                        seq_start = i
                        
                # Vérifier la dernière séquence
                if len(indices_modifiables) - seq_start >= 3:
                    sequences.append((seq_start, len(indices_modifiables)-1))
                    
                if sequences:
                    start_idx, end_idx = random.choice(sequences)
                    segment = indices_modifiables[start_idx:end_idx+1]
                    
                    # Sauvegarder les valeurs actuelles
                    valeurs = [chromosome[idx] for idx in segment]
                    random.shuffle(valeurs)
                    
                    # Réassigner les valeurs mélangées
                    for i, idx in enumerate(segment):
                        chromosome[idx] = valeurs[i]

    def local_refinement(self, solution: List[int], iterations: int = 100) -> List[int]:
        """Performs a lightweight hill-climbing improvement on the given solution."""
        best_solution = solution.copy()
        best_score = self.calculer_fitness(best_solution)
        for _ in range(iterations):
            candidate = best_solution.copy()
            indices_modifiables = [i for i in range(len(candidate)) if i not in self.indices_fixes]
            if indices_modifiables:
                idx = random.choice(indices_modifiables)
                old_val = candidate[idx]
                new_val = random.randint(0, len(self.salles)-1)
                while new_val == old_val and len(self.salles) > 1:
                    new_val = random.randint(0, len(self.salles)-1)
                candidate[idx] = new_val
                candidate_score = self.calculer_fitness(candidate)
                if candidate_score > best_score:
                    best_solution = candidate
                    best_score = candidate_score
        return best_solution

    def optimiser(self, verbose: bool = True) -> Dict[str, str]:
        """Algorithme génétique optimisé avec paramètres adaptatifs et stratégies avancées"""
        debut = time.time()
        
        # Création de la population initiale avec diversité contrôlée
        population = []
        for _ in range(self.taille_population):
            population.append(self.creer_individu())
        
        # Introduire de la diversité en modifiant légèrement chaque individu
        for i in range(1, len(population)):
            for _ in range(random.randint(1, 5)):  # Appliquer 1-5 mutations
                self.mutation(population[i], 0.8)  # Taux élevé pour diversifier
        
        meilleur_score = float('-inf')
        meilleure_solution = None
        sans_amelioration = 0  # Compteur de générations sans amélioration
        stagnation_prolongee = 0  # Compteur de périodes de stagnation
        
        scores_moyens = []
        scores_max = []
        
        # Régime dynamique: alternance entre exploration et exploitation
        mode_exploration = True
        
        for generation in range(self.nb_generations):
            # Calcul du taux de mutation adaptatif
            progress = generation / self.nb_generations
            
            # Adaptation dynamique du taux de mutation selon le mode et la progression
            if mode_exploration:
                taux_mutation = self.taux_mutation_initial - progress * 0.5 * (self.taux_mutation_initial - self.taux_mutation_final)
            else:
                taux_mutation = self.taux_mutation_final + progress * 0.3 * (self.taux_mutation_initial - self.taux_mutation_final)
            
            # Évaluation de la population 
            scores = [self.calculer_fitness(individu) for individu in population]
            
            # Statistiques
            score_moyen = sum(scores) / len(scores)
            score_max = max(scores)
            scores_moyens.append(score_moyen)
            scores_max.append(score_max)
            
            # Mise à jour de la meilleure solution
            max_score_idx = scores.index(score_max)
            if score_max > meilleur_score:
                amelioration = score_max - meilleur_score
                meilleur_score = score_max
                meilleure_solution = population[max_score_idx].copy()
                sans_amelioration = 0
                if verbose and (amelioration > 0.1 or generation % 10 == 0):
                    print(f"Génération {generation}: Nouveau meilleur score = {meilleur_score:.2f}")
            else:
                sans_amelioration += 1
            
            # Gestion de stagnation - changer de mode
            if sans_amelioration > 30:
                stagnation_prolongee += 1
                sans_amelioration = 0
                
                # Alternance entre exploration et exploitation
                mode_exploration = not mode_exploration
                
                if verbose:
                    mode = "exploration" if mode_exploration else "exploitation"
                    print(f"Stagnation détectée! Passage en mode {mode}")
                
                if mode_exploration:
                    # Introduire de la diversité (immigration)
                    nb_nouveaux = int(0.2 * self.taille_population)
                    for i in range(nb_nouveaux):
                        idx = random.randrange(len(population))
                        population[idx] = self.creer_individu()
            
            # Arrêt anticipé en cas de stagnation prolongée
            if stagnation_prolongee >= 5 and generation > 150:
                if verbose:
                    print(f"Arrêt anticipé à la génération {generation} - Convergence détectée")
                break
                
            # Sélection
            population = self.selection(population, scores)
            
            # Croisement avec stratégie adaptative
            nouvelle_population = []
            
            # Conserver l'élite intacte
            nb_elites = int(self.proportion_elite * len(population))
            indices_tries = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            elite = [population[i] for i in indices_tries[:nb_elites]]
            nouvelle_population.extend(elite)
            
            # Pool pour le croisement
            pool = [population[i] for i in indices_tries[nb_elites:]]
            random.shuffle(pool)
            
            # Croisement pour le reste
            while len(nouvelle_population) < self.taille_population:
                if len(pool) >= 2:
                    parent1, parent2 = pool.pop(), pool.pop()
                    enfant1, enfant2 = self.croisement(parent1, parent2)
                    
                    # N'ajouter que jusqu'à la taille de population cible
                    nouvelle_population.append(enfant1)
                    if len(nouvelle_population) < self.taille_population:
                        nouvelle_population.append(enfant2)
                else:
                    # S'il reste un membre isolé du pool
                    if pool:
                        nouvelle_population.append(pool.pop())
                    else:
                        break
            
            # S'assurer d'avoir exactement la taille de population voulue
            if len(nouvelle_population) < self.taille_population:
                # Compléter avec de nouveaux individus
                while len(nouvelle_population) < self.taille_population:
                    nouvelle_population.append(self.creer_individu())
            
            # Mutation adaptative - plus forte en mode exploration
            for i in range(nb_elites, len(nouvelle_population)):
                self.mutation(nouvelle_population[i], taux_mutation)
            
            population = nouvelle_population
        
        fin = time.time()
        duree = fin - debut
        
        if verbose:
            print(f"\nOptimisation terminée en {duree:.2f} secondes")
            print(f"Meilleur score: {meilleur_score:.2f}")
            self.historique_scores = {'moyens': scores_moyens, 'max': scores_max}
            
        # Étape de raffinement ultime : utiliser le hill-climbing sur le meilleur chromosome
        meilleure_solution = self.local_refinement(meilleure_solution, iterations=200)
        
        # Conversion du meilleur chromosome en placement
        placement = {}
        for i, salle_index in enumerate(meilleure_solution):
            placement[self.entreprises[i].nom] = self.salles[salle_index].nom
            
        return placement

