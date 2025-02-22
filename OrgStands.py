import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
import random
import pandas as pd
import os

# Définir le chemin du dossier
dossier = os.getcwd()

# Construire les chemins complets des fichiers
chemin_entreprises = os.path.join(dossier, "Data/entreprises.csv")
chemin_salles = os.path.join(dossier, "Data/salles.csv")


@dataclass
class Entreprise:
    nom: str
    taille: str  
    popularite_info: float
    secteur: str
    concurrent_de: List[str]

@dataclass
class Salle:
    nom: str
    capacite: int
    position_qualite: float

def charger_entreprises(chemin_entreprises: str) -> List[Entreprise]:
    """Charge les données des entreprises depuis un fichier CSV"""
    df = pd.read_csv(chemin_entreprises)
    entreprises = []
    
    for _, row in df.iterrows():
        # Convertir la chaîne de concurrents en liste
        concurrents = row['concurrent_de'].split(';') if isinstance(row['concurrent_de'], str) else []
        
        entreprise = Entreprise(
            nom=row['nom'],
            taille=row['taille'],
            popularite_info=float(row['popularite_info']),
            secteur=row['secteur'],
            concurrent_de=concurrents
        )
        entreprises.append(entreprise)
    
    return entreprises

def charger_salles(chemin_salles: str) -> List[Salle]:
    """Charge les données des salles depuis un fichier CSV"""
    df = pd.read_csv(chemin_salles)
    salles = []
    
    for _, row in df.iterrows():
        salle = Salle(
            nom=row['nom'],
            capacite=int(row['capacite']),
            position_qualite=float(row['position_qualite'])
        )
        salles.append(salle)
    
    return salles

class OrganisateurStandsGenetique:
    def __init__(self, entreprises: List[Entreprise], salles: List[Salle]):
        self.entreprises = entreprises
        self.salles = salles
        self.taille_population = 50
        self.nb_generations = 100
        self.taux_mutation = 0.1
        
    def creer_individu(self) -> List[int]:
        """Crée un placement aléatoire (chromosome)"""
        return [random.randint(0, len(self.salles)-1) for _ in range(len(self.entreprises))]
         #fonction calcule un score pour chaque solution en considérant
    def calculer_fitness(self, chromosome: List[int]) -> float:
        score = 0.0
        placement = {}  # Entreprise -> Salle
        occupation_salles = {i: 0 for i in range(len(self.salles))}
        
        # Vérifier les contraintes et calculer le score
        for i, salle_index in enumerate(chromosome):
            entreprise = self.entreprises[i]
            salle = self.salles[salle_index]
            occupation_salles[salle_index] += 1
            
            # Pénalité pour dépassement de capacité
            if occupation_salles[salle_index] > salle.capacite:
                score -= 100
            
            # Bonus pour bonne position selon la taille/popularité
            poids_taille = {'grande': 1.0, 'moyenne': 0.7, 'petite': 0.4}
            score += salle.position_qualite * poids_taille[entreprise.taille] * entreprise.popularite_info
            
            # Pénalité pour concurrents dans la même salle
            for j, autre_salle_index in enumerate(chromosome):
                if i != j and autre_salle_index == salle_index:
                    autre_entreprise = self.entreprises[j]
                    if autre_entreprise.nom in entreprise.concurrent_de:
                        score -= 50
            
            # Bonus pour répartition équilibrée des types d'entreprises
            types_par_salle = {}
            for j, s_index in enumerate(chromosome):
                if s_index not in types_par_salle:
                    types_par_salle[s_index] = {'grande': 0, 'moyenne': 0, 'petite': 0}
                types_par_salle[s_index][self.entreprises[j].taille] += 1
            
            for types in types_par_salle.values():
                if abs(types['grande'] - types['moyenne']) <= 1 and abs(types['moyenne'] - types['petite']) <= 1:
                    score += 20
                    
        return score
        
        #Choisit les meilleures solutions par tournoi et Les solutions avec meilleur score ont plus de chances d'être sélectionnées

        
    def selection(self, population: List[List[int]], scores: List[float]) -> List[List[int]]:
        """Sélection par tournoi"""
        nouvelle_population = []
        for _ in range(len(population)):
            tournoi = random.sample(list(enumerate(scores)), 3)
            gagnant = max(tournoi, key=lambda x: x[1])[0]
            nouvelle_population.append(population[gagnant])
        return nouvelle_population

    # Combine deux bonnes solutions pour en créer de nouvelles(Prend une partie de chaque parent)
    def croisement(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Croisement en un point"""
        point = random.randint(1, len(parent1)-1)
        enfant1 = parent1[:point] + parent2[point:]
        enfant2 = parent2[:point] + parent1[point:]
        return enfant1, enfant2
    
    #Modifie aléatoirement certaines solutions Et Aide à explorer de nouvelles possibilités

    def mutation(self, chromosome: List[int]):
        """Mutation par permutation"""
        if random.random() < self.taux_mutation:
            i, j = random.sample(range(len(chromosome)), 2)
            chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
    #Crée une population initiale aléatoire Et Pour chaque génération Évalue chaque solution,Sélectionne les meilleures,Crée de nouvelles solutions par croisement,Applique des mutations 
    def optimiser(self) -> Dict[str, str]:
        # Création de la population initiale
        population = [self.creer_individu() for _ in range(self.taille_population)]
        
        meilleur_score = float('-inf')
        meilleure_solution = None
        
        for generation in range(self.nb_generations):
            # Évaluation
            scores = [self.calculer_fitness(individu) for individu in population]
            
            # Mise à jour de la meilleure solution
            max_score_idx = scores.index(max(scores))
            if scores[max_score_idx] > meilleur_score:
                meilleur_score = scores[max_score_idx]
                meilleure_solution = population[max_score_idx]
            
            # Sélection
            population = self.selection(population, scores)
            
            # Croisement
            nouvelle_population = []
            for i in range(0, len(population), 2):
                if i+1 < len(population):
                    enfant1, enfant2 = self.croisement(population[i], population[i+1])
                    nouvelle_population.extend([enfant1, enfant2])
                else:
                    nouvelle_population.append(population[i])
            
            # Mutation
            for individu in nouvelle_population:
                self.mutation(individu)
            
            population = nouvelle_population

        # Conversion du meilleur chromosome en placement
        placement = {}
        for i, salle_index in enumerate(meilleure_solution):
            placement[self.entreprises[i].nom] = self.salles[salle_index].nom
            
        return placement
        
    # [Le reste de votre code reste identique]

# Exemple d'utilisation
if __name__ == "__main__":
    # Chargement des données depuis les fichiers CSV
    
    entreprises = charger_entreprises(chemin_entreprises)
    salles = charger_salles(chemin_salles)
    
    organisateur = OrganisateurStandsGenetique(entreprises, salles)
    resultat = organisateur.optimiser()
    
    print("\nRésultat du placement :")
    for entreprise, salle in resultat.items():
        print(f"{entreprise} -> {salle}")

    # Convert dictionary to DataFrame
    df = pd.DataFrame(list(resultat.items()), columns=["entreprise", "emplacement"])

    # Save to CSV file
    df.to_csv("Output/resultat_placement.csv", index=False, encoding="utf-8")

    print("\nRésultats sauvegardés dans 'resultat_placement.csv'")
