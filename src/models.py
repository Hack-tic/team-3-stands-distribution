from dataclasses import dataclass
from typing import List

@dataclass
class Entreprise:
    """Classe représentant une entreprise participant à l'événement"""
    nom: str
    taille: str  
    popularite_info: float
    secteur: str
    concurrent_de: List[str]

@dataclass
class Salle:
    """Classe représentant une salle pour l'événement"""
    nom: str
    capacite: int
    position_qualite: float

