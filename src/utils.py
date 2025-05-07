import os
import pandas as pd

def create_output_directory():
    """Ensure that the Output directory exists"""
    if not os.path.exists("Output"):
        os.makedirs("Output")
        print("Output directory created")
    return "Output"

def save_results_to_csv(result_dict, filename="resultat_placement.csv"):
    """Save results to a CSV file"""
    # Convert dictionary to DataFrame
    df = pd.DataFrame(list(result_dict.items()), columns=["entreprise", "emplacement"])
    
    # Ensure output directory exists
    output_dir = create_output_directory()
    
    # Save to CSV file
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False, encoding="utf-8")
    
    print(f"\nRésultats sauvegardés dans '{filepath}'")
    return filepath

