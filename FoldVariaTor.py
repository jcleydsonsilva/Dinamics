import os
import argparse
import subprocess

# Step 1: Prepare input list of mutants
def prepare_mutant_list(mutant_list_tsv,work_dir):
    individual_list = os.path.join(work_dir, 'individual_list.txt')
    with open(individual_list, "w") as output:
        cmd = (
            f"tail -n +2 {mutant_list_tsv} | cut -f 1,2 | tr '>' '\\t' | "
            "awk '{print $2 \"A\" $1 $3 \",\" $2 \"B\" $1 $3 \";\"}'"
        )
        mutants = subprocess.check_output(cmd, shell=True, text=True)
        output.write(mutants)
    print(f"Mutant list prepared: {individual_list}")
    return individual_list


# Step 4: Run FoldX to generate mutants
def run_foldx(input_pdb,mutant_list,output_dir):
    directory, filename = os.path.split(input_pdb)
    foldx_command = (
        f"foldx --command=BuildModel --pdb-dir={directory} "
        f"--pdb={filename} --mutant-file={mutant_list} "
        f"--output-file={output_dir}/1_{input_pdb.replace('.pdb','')}_known_mutant_"
    )
    subprocess.run(foldx_command, shell=True, check=True)
    print (foldx_command)
    print("FoldX execution complete.")    

# Main workflow
if __name__ == "__main__":
    # Define paths and environment variables
    parser = argparse.ArgumentParser(
        description="Process files for ligand, protein, and mutants_uniprot."
    )
    parser.add_argument(
        "-m", "--mutations", 
        required=True, 
        type=str, 
        help="Path to the mutations file."
    )
    parser.add_argument(
        "-p", "--protein", 
        required=True, 
        type=str, 
        help="Path to the protein file."
    )

    # Configuração do argparse
    args = parser.parse_args() 
    
    work_dir = os.getcwd()
    input_pdb = os.path.join(work_dir, args.protein)
    mutant_list_tsv = os.path.join(work_dir, args.mutations)
    
    output_dir = os.path.join(work_dir, "outputs_foldx")
    os.makedirs(output_dir, exist_ok=True)
    rotabase_path = os.path.join(work_dir, "rotabase.txt")
    
    try:
        mutant_list = prepare_mutant_list(mutant_list_tsv,work_dir)
        run_foldx(input_pdb,mutant_list,output_dir)

        print(f"All tasks completed. Mutant complexes are in: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error during execution: {e}") 
