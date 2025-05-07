import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from typing import Dict, List
from src.models import Entreprise, Salle

def analyser_solution(entreprises: List[Entreprise], salles: List[Salle], 
                      placement: Dict[str, str], emplacements_fixes: Dict[str, str] = None):
    """Analyse détaillée de la solution trouvée"""
    # Créer un mapping inverse
    placement_inverse = defaultdict(list)
    for entreprise_nom, salle_nom in placement.items():
        placement_inverse[salle_nom].append(entreprise_nom)
    
    # Statistiques par salle
    print("\nAnalyse de la solution:")
    print("="*50)
    
    for salle_nom, entreprises_noms in placement_inverse.items():
        print(f"\nSalle: {salle_nom}")
        print("-"*30)
        
        # Trouver l'objet salle
        salle = next((s for s in salles if s.nom == salle_nom), None)
        print(f"Capacité: {salle.position_qualite if salle else 'Inconnue'}")
        print(f"Nombre d'entreprises: {len(entreprises_noms)}")
        
        # Répartition par taille
        tailles = defaultdict(int)
        secteurs = defaultdict(int)
        
        for e_nom in entreprises_noms:
            # Trouver l'objet entreprise
            entreprise = next((e for e in entreprises if e.nom == e_nom), None)
            if entreprise:
                tailles[entreprise.taille] += 1
                secteurs[entreprise.secteur] += 1
        
        print(f"Répartition par taille: {dict(tailles)}")
        print(f"Répartition par secteur: {dict(secteurs)}")
        
        # Afficher la liste détaillée des entreprises avec leur taille et popularité
        print("\nListe des entreprises dans cette salle:")
        print(f"{'Nom':<30} {'Taille':<10} {'Popularité':<10} {'Statut'}")
        print("-"*65)
        
        # Trier les entreprises par taille puis par popularité
        entreprises_details = []
        for e_nom in entreprises_noms:
            entreprise = next((e for e in entreprises if e.nom == e_nom), None)
            if entreprise:
                entreprises_details.append(entreprise)
        
        # Trier par taille (grande > moyenne > petite) puis par popularité (décroissante)
        taille_ordre = {'grande': 0, 'moyenne': 1, 'petite': 2}
        entreprises_details.sort(key=lambda e: (taille_ordre[e.taille], -e.popularite_info))
        
        for e in entreprises_details:
            # Tronquer le nom s'il est trop long
            nom_aff = e.nom[:27] + "..." if len(e.nom) > 30 else e.nom
            
            # Vérifier si c'est un stand fixé
            est_fixe = emplacements_fixes and e.nom in emplacements_fixes and emplacements_fixes[e.nom] == salle_nom
            statut = "stand fixé" if est_fixe else ""
            
            print(f"{nom_aff:<30} {e.taille:<10} {e.popularite_info:<10.2f} {statut}")
        
        # Vérifier les concurrents
        concurrents = []
        for i, e1_nom in enumerate(entreprises_noms):
            e1 = next((e for e in entreprises if e.nom == e1_nom), None)
            if e1:
                for j, e2_nom in enumerate(entreprises_noms[i+1:], i+1):
                    if e2_nom in e1.concurrent_de:
                        concurrents.append((e1_nom, e2_nom))
        
        if concurrents:
            print(f"⚠️  ALERTE: {len(concurrents)} paires de concurrents dans cette salle!")
            for c1, c2 in concurrents:
                print(f"  - {c1} et {c2}")
        else:
            print("✓ Aucun concurrent dans cette salle")
    
    # Statistiques globales
    print("\nStatistiques globales:")
    print("="*50)
    
    total_concurrents = 0
    for salle_nom, entreprises_noms in placement_inverse.items():
        for i, e1_nom in enumerate(entreprises_noms):
            e1 = next((e for e in entreprises if e.nom == e1_nom), None)
            if e1:
                for j, e2_nom in enumerate(entreprises_noms[i+1:], i+1):
                    if e2_nom in e1.concurrent_de:
                        total_concurrents += 1
    
    print(f"Nombre total de paires de concurrents dans la même salle: {total_concurrents}")
    
    # Générer un graphique de répartition
    visualiser_repartition(entreprises, placement_inverse)

def visualiser_repartition(entreprises: List[Entreprise], placement_inverse: Dict[str, List[str]]):
    """Génère un graphique de la répartition des entreprises par salle"""
    # Données pour le graphique
    salles = list(placement_inverse.keys())
    categories = {'grande': [], 'moyenne': [], 'petite': []}
    
    for salle in salles:
        tailles = {'grande': 0, 'moyenne': 0, 'petite': 0}
        for e_nom in placement_inverse[salle]:
            entreprise = next((e for e in entreprises if e.nom == e_nom), None)
            if entreprise:
                tailles[entreprise.taille] += 1
        
        categories['grande'].append(tailles['grande'])
        categories['moyenne'].append(tailles['moyenne'])
        categories['petite'].append(tailles['petite'])
    
    # Créer le graphique
    fig, ax = plt.subplots(figsize=(12, 8))
    
    width = 0.25
    x = np.arange(len(salles))
    
    ax.bar(x - width, categories['grande'], width, label='Grandes', color='#3498db')
    ax.bar(x, categories['moyenne'], width, label='Moyennes', color='#2ecc71')
    ax.bar(x + width, categories['petite'], width, label='Petites', color='#e74c3c')
    
    ax.set_xlabel('Salles')
    ax.set_ylabel('Nombre d\'entreprises')
    ax.set_title('Répartition des entreprises par taille et par salle')
    ax.set_xticks(x)
    ax.set_xticklabels(salles, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("Output/repartition_par_salle.png")
    print("\nGraphique de répartition enregistré dans 'Output/repartition_par_salle.png'")

def tracer_evolution_scores(historique_scores: Dict[str, List[float]]):
    """Trace l'évolution des scores pendant l'optimisation"""
    plt.figure(figsize=(10, 6))
    plt.plot(historique_scores['moyens'], label='Score moyen')
    plt.plot(historique_scores['max'], label='Score maximum')
    plt.xlabel('Génération')
    plt.ylabel('Score')
    plt.title('Évolution des scores pendant l\'optimisation')
    plt.legend()
    plt.grid(True)
    plt.savefig("Output/evolution_scores.png")
    print("Graphique d'évolution des scores enregistré dans 'Output/evolution_scores.png'")

