import io
import sys
import os
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
import pandas as pd
import uvicorn
from contextlib import redirect_stdout
from typing import List, Dict, Optional
import csv

# Import the necessary modules from the project
from src.data_loader import charger_entreprises, charger_salles
from src.genetic_algorithm import OrganisateurStandsGenetique
from src.visualization import analyser_solution, tracer_evolution_scores
from src.utils import save_results_to_csv, create_output_directory
from src.models import Entreprise, Salle

app = FastAPI(
    title="SmartStandOrg API",
    description="API for smart arrangement of stands in an event using genetic algorithms",
    version="1.0.0",
)

class CaptureOutput:
    def __init__(self):
        self.buffer = io.StringIO()
        self.console_output = ""
    
    def __enter__(self):
        sys.stdout = self.buffer
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = sys.__stdout__
        self.console_output = self.buffer.getvalue()
        self.buffer.close()


@app.get("/")
def read_root():
    return {"message": "Welcome to SmartStandOrg API"}


@app.post("/optimize")
async def optimize_stands(
    background_tasks: BackgroundTasks,
    entreprises_file: UploadFile = File(...),
    salles_file: UploadFile = File(...),
    entreprises_fixes_file: Optional[UploadFile] = File(None)
):
    try:
        # Create temporary directory for file uploads
        create_output_directory()
        
        # Save uploaded files temporarily
        temp_entreprises_path = os.path.join("Data", "temp_entreprises.csv")
        temp_salles_path = os.path.join("Data", "temp_salles.csv")
        temp_fixes_path = os.path.join("Data", "temp_entreprises_fixes.csv")
        
        # Write uploaded files to disk
        with open(temp_entreprises_path, "wb") as buffer:
            buffer.write(await entreprises_file.read())
        
        with open(temp_salles_path, "wb") as buffer:
            buffer.write(await salles_file.read())

        # Handle optional entreprises_fixes file
        emplacements_fixes = {}
        if entreprises_fixes_file:
            with open(temp_fixes_path, "wb") as buffer:
                buffer.write(await entreprises_fixes_file.read())
            
            # Load fixed placements
            df_fixes = pd.read_csv(temp_fixes_path)
            for _, row in df_fixes.iterrows():
                emplacements_fixes[row['entreprise']] = row['salle']
        
        # Load data from CSV files
        entreprises = charger_entreprises(temp_entreprises_path)
        salles = charger_salles(temp_salles_path)
        
        # Capture the console output
        with CaptureOutput() as output:
            # Create the organizer and run the optimization
            print("Démarrage de l'optimisation génétique...")
            organisateur = OrganisateurStandsGenetique(entreprises, salles, emplacements_fixes)
            resultat = organisateur.optimiser(verbose=True)
            
            # Display placement results
            print("\nRésultat du placement :")
            for entreprise, salle in resultat.items():
                if entreprise in emplacements_fixes:
                    print(f"{entreprise} -> {salle} (fixe)")
                else:
                    print(f"{entreprise} -> {salle}")

            # Detailed analysis of the solution
            analyser_solution(entreprises, salles, resultat, emplacements_fixes)
            
            # Trace score evolution if available
            if hasattr(organisateur, 'historique_scores') and organisateur.historique_scores:
                tracer_evolution_scores(organisateur.historique_scores)
            
            # Save results to CSV
            result_path = save_results_to_csv(resultat)

        # Prepare the CSV data for response
        df_result = pd.DataFrame(list(resultat.items()), columns=["entreprise", "emplacement"])
        csv_output = io.StringIO()
        df_result.to_csv(csv_output, index=False)
        csv_output.seek(0)
        
        # Clean up temporary files in the background
        background_tasks.add_task(os.remove, temp_entreprises_path)
        background_tasks.add_task(os.remove, temp_salles_path)
        if entreprises_fixes_file:
            background_tasks.add_task(os.remove, temp_fixes_path)
        
        # Return the CSV file as a response
        return {
            "console_output": output.console_output,
            "result": resultat
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)