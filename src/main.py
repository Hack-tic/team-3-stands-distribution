import os
import time
from src.data_loader import charger_entreprises, charger_salles, charger_entreprises_fixes
from src.genetic_algorithm import OrganisateurStandsGenetique
from src.visualization import analyser_solution, tracer_evolution_scores
from src.utils import save_results_to_csv, create_output_directory

# Définir les chemins des fichiers de données
dossier = os.getcwd()
chemin_entreprises = os.path.join(dossier, "Data/entreprises.csv")
chemin_salles = os.path.join(dossier, "Data/salles.csv")
chemin_fixes = os.path.join(dossier, "Data/entreprises_fixes.csv")

def main():
    """Point d'entrée principal de l'application"""
    # Vérifier si le dossier Output existe
    create_output_directory()
    
    # Chargement des données depuis les fichiers CSV
    print("Chargement des données...")
    entreprises = charger_entreprises(chemin_entreprises)
    salles = charger_salles(chemin_salles)
    emplacements_fixes = charger_entreprises_fixes(chemin_fixes)
    
    print(f"Chargement de {len(entreprises)} entreprises et {len(salles)} salles")
    if emplacements_fixes:
        print(f"Chargement de {len(emplacements_fixes)} emplacements fixes")
    
    # Création de l'organisateur et lancement de l'optimisation
    print("\nDémarrage de l'optimisation génétique...")
    organisateur = OrganisateurStandsGenetique(entreprises, salles, emplacements_fixes)
    resultat = organisateur.optimiser(verbose=True)
    
    # Affichage des résultats
    print("\nRésultat du placement :")
    for entreprise, salle in resultat.items():
        if entreprise in emplacements_fixes:
            print(f"{entreprise} -> {salle} (fixe)")
        else:
            print(f"{entreprise} -> {salle}")

    # Analyse détaillée de la solution
    analyser_solution(entreprises, salles, resultat, emplacements_fixes)
    
    # Tracer l'évolution des scores si disponible
    if hasattr(organisateur, 'historique_scores') and organisateur.historique_scores:
        tracer_evolution_scores(organisateur.historique_scores)
    
    # Sauvegarder les résultats dans un fichier CSV
    save_results_to_csv(resultat)

if __name__ == "__main__":
    main()

