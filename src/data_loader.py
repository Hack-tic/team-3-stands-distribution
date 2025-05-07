import os
import pandas as pd
from typing import List, Dict
from src.models import Entreprise, Salle

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

def charger_entreprises_fixes(chemin_fixes: str) -> Dict[str, str]:
    """Charge les données des entreprises à position fixe depuis un fichier CSV"""
    try:
        df = pd.read_csv(chemin_fixes)
        fixes = {}
        
        for _, row in df.iterrows():
            fixes[row['entreprise']] = row['salle']
            
        return fixes
    except FileNotFoundError:
        print(f"Fichier d'entreprises fixes non trouvé: {chemin_fixes}")
        return {}
    except Exception as e:
        print(f"Erreur lors du chargement des entreprises fixes: {str(e)}")
        return {}

