import sys
import numpy as np
import pandas as pd
import requests
import time
import subprocess
import os
import argparse
from datetime import datetime
from pathlib import Path
from requests.exceptions import HTTPError

# Set PYTHONPATH to path/to/MutCompute
from src.box_extraction_all_aa_7channel import *
from src.CNN_test_all import *
from src.output_progs import *



def fetch_pdb_file(pdb, dir=Path('../data/pdb_files'), fs=False):

    dir = Path(dir)

    dir.mkdir(0o777, exist_ok=True, parents=True)

    def gen_pdb_from_api(pdb, dir):

        pdb_file = Path('')
        print('Fetching PDB from api...')
        
        try: 
            respond = requests.get(f'https://pdb-redo.eu/db/{pdb}/{pdb}_final_tot.pdb')
            if respond.ok:
                pdb_filename = f'{pdb}_final_tot.pdb'
                pdb_file = dir / pdb_filename

                with pdb_file.open('wt') as f: 
                    f.write(respond.text)

                print('Successfully fetched from PDB-REDO')
            raise HTTPError()

        except Exception as e:
            from sys import stderr
            print(f'PDB redo error: {e}', file=stderr)
            pdb_filename = f'{pdb}.pdb'
            pdb_file = dir / pdb_filename

            respond = requests.get(f'https://files.rcsb.org/download/{pdb_filename}')

            if respond.ok: 
                print('does file exist: ', Path().cwd(), dir.exists(), pdb_file.exists())
                with pdb_file.open('wt') as f:  
                    f.write(respond.text)

            else:
                raise FileNotFoundError(f'Unable to retrieve pdb file from rcsb: {pdb_filename}')

            print('Successfully fetched from RCSB')

        return pdb_file



    def gen_phys_pdbs(pdb_file, generate_all=True):

        pdb_file = Path(pdb_file)

        dir = pdb_file.parent
        
        pqr_file = dir / (pdb_file.stem + '.pqr.pdb')
        sasa_file = dir / (pdb_file.stem + '.sasa.pdb')

        print('PDB file path: ', pdb_file)
        print('PQR file path: ', pqr_file)
        print('SASA file path: ', sasa_file)

        if generate_all: 
            print('Generating PQR and SASA pdb files... ')
            
            print('PQR SOFTWARE OUTPUT:')
            os.system(f'python2 ../dependencies/pdb2pqr-2.1/pdb2pqr.py --ff=parse --chain {pdb_file} {pqr_file}')
            print('\n')
            
            print('SASA SOFTWARE OUTPUT:')
            os.system(f'freesasa --no-warnings --shrake-rupley --depth=atom --hydrogen --format=pdb {pqr_file} > {sasa_file}')
            print('\n')
        
        else:
            if pqr_file.exists() and sasa_file.exists():
                print('PQR and SASA files have been detected in the filesystem: ')
                print('PQR : ', pqr_file)
                print('SASA: ', sasa_file)

            elif pqr_file.exists():
                print('PQR file detected in the filesystem, generating SASA pdb file... ')
                print('PQR : ', pqr_file)
                
                print('SASA SOFTWARE OUTPUT:')
                os.system(f'freesasa --no-warnings --shrake-rupley --depth=atom --hydrogen --format=pdb {pqr_file} > {sasa_file}')
                print('\n')

            elif sasa_file.exists():
                print('SASA file detected in the filesystem, generating PQR pdb file... ')
                print('SASA : ', sasa_file)
                
                print('PQR SOFTWARE OUTPUT:')
                os.system(f'python2 /mutcompute/dependencies/pdb2pqr-2.1/pdb2pqr.py --ff=parse --chain {pdb_file} {pqr_file}')
                print('\n')

            else:
                print('Generating PQR and SASA pdb files... ')
                
                print('PQR SOFTWARE OUTPUT:')
                os.system(f'python2 /mutcompute/dependencies/pdb2pqr-2.1/pdb2pqr.py --ff=parse --chain {pdb_file} {pqr_file}')
                print('\n')
                
                print('SASA SOFTWARE OUTPUT:')
                os.system(f'freesasa --no-warnings --shrake-rupley --depth=atom --hydrogen --format=pdb {pqr_file} > {sasa_file}')
                print('\n')

        return None


    if len(pdb) > 4 and fs == True:
        pdb_file = dir / pdb
        gen_phys_pdbs(pdb_file)
        return pdb_file
    
    elif len(pdb) == 4 and fs == False: 
        pdb_file = gen_pdb_from_api(pdb, dir)
        gen_phys_pdbs(pdb_file)
        return pdb_file

    elif len(pdb) > 4 and fs == False:
        print('Command Line Argument Error')
        print('If --pdb argument is a file (greater than 4 characters), then -f flag must not be set to False.')
        exit(0)

    elif len(pdb) == 4 and fs == True: 
        print('Command Line Argument Error')
        print('If --file is set to True, --pdb argument must be set to a filename and not to 4 character pdb code.')
        exit(0)
    else: 
        print('Command Line Argument Error')
        print('Invalid --pdb and --file argument combination.')
        exit(0)



def generate_all_boxes(pdb_file):

    pdb_file = Path(pdb_file)

    print('generate_all_boxes: ', pdb_file)

    dir = pdb_file.parent
    pdb_filename = pdb_file.stem
    pqr_file = dir / (pdb_filename + '.pqr.pdb')
    sasa_file = dir / (pdb_filename + '.sasa.pdb')

    pdb_id  = pdb_file
    pqr_id  = pqr_file
    sasa_id = sasa_file

    print('PDB id: ', pdb_id)
    structure_pdb = fetch_pdb_structure(str(pdb_file))
    structure_pqr = fetch_pdb_structure(str(pqr_file))
    structure_sasa = fetch_pdb_structure(str(sasa_file))

    xcoord, ycoord, zcoord, p_charge, all_aa_set, atmres_name = extract_coordinates_from_pdb(structure_pqr,'pqr')
    sas, atmres_name_sas = extract_coordinates_from_pdb(structure_sasa,'sasa')
    p_charge_dict = dict(zip(atmres_name, p_charge))
    sas_dict = dict(zip(atmres_name_sas, sas))

    xdata, ydata = featurization(all_aa_set, list(zip(xcoord, ycoord, zcoord)), atmres_name, p_charge_dict, sas_dict)
    ydata = np.concatenate((ydata, np.reshape([0]*np.shape(ydata)[0], [np.shape(ydata)[0],1])), 1)

    mean_matrix = np.load(Path.cwd().parent / 'model/mean_matrix.npy')
    train_std = np.load(Path.cwd().parent / 'model/stdev_matrix.npy')
    xdata -= mean_matrix
    xdata /= train_std

    print('PDB:%s|aaLen:%d|aaExtracted:%d' % (pdb_file,np.shape(all_aa_set)[0],np.shape(ydata)[0]))

    return xdata, ydata



def parse_nn_out(xdata, ydata, weights):
    out = []
    predict_out = train_3DCNN(xdata, ydata[:,2], np.shape(ydata)[0], weights)

    pred = np.array([])
    j = 1
    prob = []
    for i in predict_out:
        for k in i:
            if j % 2 == 0:
                pred = np.concatenate([pred,k])
            else:
                if prob == []:
                    prob = k
                else:
                    prob = np.vstack((prob,k))
            j += 1

    predicted_prob = []
    true_prob = []
    all_prob = []
    wt_prob = []
    for ind, i in enumerate(pred):
        all_prob += [prob[ind]]
        wt_prob += [prob[ind][int(ydata[ind][2])]]
        predicted_prob += [prob[ind][int(i)]]

    for ind, i in enumerate(pred):
        aa_key = ydata[ind][3] + '_' + ydata[ind][1] + ydata[ind][0]
        pdb = ydata[ind][4]
        # prob_wtSub = prob[ind] - wt_prob[ind]
        # notWT_prob = sum([prob[ind][j_ind] for j_ind,j in enumerate(prob_wtSub) if j > 0])
        # if notWT_prob == 0:
        #     notWT_logRatio = 0
        # else:
        #     notWT_logRatio = np.log(notWT_prob / wt_prob[ind])
        # out.append([','.join([pdb, ydata[ind][3], ydata[ind][0], ydata[ind][1], ','.join(map(str, prob[ind]))])])
        out.append([','.join([pdb, ydata[ind][3], ydata[ind][0], ydata[ind][1], ','.join(map(str, all_prob[ind]))])])
    return out



def gen_ensemble_inference(pdb_code=None, dir=Path('../data/pdb_files'), out_dir=Path('../data/inference_CSVs'), fs_pdb=False):

    dir = Path(dir)
    out_dir = Path(out_dir)

    dir.mkdir(0o777, exist_ok=True, parents=True)
    out_dir.mkdir(0o777, exist_ok=True, parents=True)

    # Change directory for all containers to the cwd
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    # Prioritizes argument, if missing checks commandline.
    if not pdb_code and len(sys.argv) > 1: 
        pdb_code = sys.argv[1]

    # Messes custom files that have capitalized letters
    #pdb_code = pdb_code.lower()

    
    pdb_file = fetch_pdb_file(pdb_code, fs=fs_pdb, dir=dir)
    pdb_stem = pdb_file.stem
   
    print('Generating ensemble inference for: {}'.format(pdb_code))

    # Monkey Patching
    #1
    # raise AssertionError('gen_ensemble_inference Assertion Error!')
    # Monkey Patching
    #2
    # time.sleep(5)
    # return True

    x, y = generate_all_boxes(pdb_file)
    
    f1 = parse_nn_out(x, y, Path.cwd().parent / 'model/weights/weights1/weight_3DCNN.zip')
    f2 = parse_nn_out(x, y, Path.cwd().parent / 'model/weights/weights2/weight_3DCNN.zip')
    f3 = parse_nn_out(x, y, Path.cwd().parent / 'model/weights/weights3/weight_3DCNN.zip')

    avgDict, predDict = avgProbs([f1,f2,f3])
    ratio_d1 = calcRatio(f1, predDict)
    ratio_d2 = calcRatio(f2, predDict)
    ratio_d3 = calcRatio(f3, predDict)

    col_names = ['pdb_id','chain_id', 'pos', 'wtAA', 'prAA', 'wt_prob',
                  'pred_prob', 'avg_log_ratio'] + ['pr'+num_to_aa(k) for k in range(20)]
                  
                #   list(map(str,avgDict[aa_id]/3))

    output_list = []
    for i in f1:
        i = i[0].split(',')
        chn_id = i[1]
        pdb_id = i[0] # PDB code only  # Need this so the nn_api can find the file correctly. 
        # pdb_id = i[0]+'_'+i[1] # PDB and chain (How will the nn_api know the chain?)
        aa_id = '_'.join([i[0],i[1],i[2]])
        res = i[2]
        wt_aa = i[3]
        pred_aa = num_to_aa(predDict[aa_id])
        wt_prob = avgDict[aa_id][aa_to_num(wt_aa)]/3
        pred_prob = avgDict[aa_id][predDict[aa_id]]/3
        avgLogRatio = round(np.mean([ratio_d1[aa_id], ratio_d2[aa_id], ratio_d3[aa_id]]), 3)

        output_list.append(
            [
                pdb_id, chn_id, res, wt_aa, pred_aa, wt_prob, pred_prob, 
                avgLogRatio] + list(map(float,avgDict[aa_id]/3))
            
        )
    
        print(','.join([pdb_id, res, wt_aa, pred_aa, str(wt_prob), ','.join(map(str,avgDict[aa_id]/3))]))

    output_df = pd.DataFrame(output_list, columns=col_names)
    output_df['aa_id'] = output_df['pdb_id'] + "_" + output_df['chain_id']

    # Reorder amino acid columns in alphabetical order
    output_df = output_df[
        [
            'aa_id', 'pdb_id', 'chain_id','pos', 'wtAA',
            'prAA', 'wt_prob', 'pred_prob', 'avg_log_ratio',
            'prALA', 'prARG', 'prASN', 'prASP', 'prCYS',
            'prGLN', 'prGLU', 'prGLY', 'prHIS', 'prILE',
            'prLEU', 'prLYS', 'prMET', 'prPHE', 'prPRO',
            'prSER', 'prTHR', 'prTRP', 'prTYR', 'prVAL'
        ]
    ]

    output_filename = out_dir / (pdb_stem + ".csv")
    output_df.to_csv(output_filename)

    return output_df

# 1. Read the file grn_201.csv and use the first 10 columns
def read_data(file_path):
    df = pd.read_csv(file_path, usecols=range(10))
    return df

# 2. Create the function that converts Three-Letter Code to Single-Letter Code
def convert_3_to_1(aa):
    amino_acid_map = {
            'GLY': 'G', 'ALA': 'A', 'LEU': 'L', 'MET': 'M', 'PHE': 'F',
            'TRP': 'W', 'LYS': 'K', 'GLN': 'Q', 'GLU': 'E', 'SER': 'S',
            'PRO': 'P', 'VAL': 'V', 'ILE': 'I', 'CYS': 'C', 'TYR': 'Y',
            'HIS': 'H', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'THR': 'T'}
    return amino_acid_map.get(aa.upper(), None)

# 3. Format the new column according to the table
def format_evalmut(row):
    pos = row['pos']
    wtAA = convert_3_to_1(row['wtAA'])
    prAA = convert_3_to_1(row['prAA'])
    chain = row['chain_id']
    return f"{wtAA}{pos}.{chain}{{{prAA}}}"

# 4. Execute the command and capture the output
def execute_command(evalmut, pdb_path):

    command = f"maestro config.xml {pdb_path} --bu --evalmut='{evalmut}'"
    print(f"Executing: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout

# 5. Process the output to extract data into columns
def process_output(output):
    lines = output.strip().split('\n')
    if len(lines) > 2:  # Ensure there are at least 3 lines
        mutation_line = lines[2]  # Capture the third line
        # Split the line by tabs into columns
        return mutation_line.split('\t')[0:8]
    return [None] * 8  # Return None for all 9 columns if no valid line

def convert_pattern(input_pattern):
    """
    Converte o padrão "R115.A{K}" para "RA115K,RB115K" de forma genérica.

    Args:
        input_pattern (str): Padrão no formato original, como "R115.A{K}".

    Returns:
        str: Padrão convertido, como "RA115K,RB115K".
    """
    # Expressão regular para capturar os componentes do padrão
    match = re.match(r"(\w)(\d+)\.(\w)\{(\w)\}", input_pattern)
    if not match:
        raise ValueError("Padrão de entrada inválido: " + input_pattern)

    original_amino = match.group(1)  # Aminoácido original
    position = match.group(2)       # Posição
    chain = match.group(3)          # Cadeia
    mutated_amino = match.group(4)  # Aminoácido mutado

    # Gerar os padrões convertidos
    converted_pattern = f"{original_amino}{chain}{position}{mutated_amino},{original_amino}B{position}{mutated_amino}"

    return converted_pattern

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Run Ensemble Net on a pdb file stored locally or through an api.')

    parser.add_argument('-p', '--pdb', required=True, 
                        help='REQUIRED. Either a pdb 4 char code to pull from an API or a pdb filename in the --dir path.'
    )

    parser.add_argument('-f', '--isFile', required=True, default=True, type=bool, 
                        help='OPTIONAL. Boolean flag to query filesystem for pdb file (True) or to query api with pdb code (False). '
                        'Default: False'
    )

    parser.add_argument('-n', '--name', required=False, default=f'run', type=str, 
                        help='OPTIONAL. Name of run. '
                        'Default: run'
    )

    parser.add_argument('-d', '--dir', required=False, default='./', type=str, 
                        help='OPTIONAL. Directory path where pdb files are to be queried or saved. '
                        'Default: ./'
    )

    parser.add_argument('--out-dir', required=False, default='./', type=str,
                        help='OPTIONAL. Directory path where output csv will be saved. '
                        'Default: ./'
    )

    args = parser.parse_args()

    print('ARGS: ', args)
    
    start = time.perf_counter()
    status = 'Unsuccessful'


    try:
        gen_ensemble_inference(pdb_code=args.pdb, fs_pdb=args.isFile, dir=args.dir, out_dir=args.out_dir)
        status = 'Successful'

    except Exception as e:
        print(f'An exception occured: {e}')
        now = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        with open(f'logs/err-{args.name}-logs.txt', 'a+') as f: 
            f.write(f'{now} {status:<12} Run PDB: {args.pdb} Error: {type(e)} {e}\n')
        # raise e

    finally:
        time = time.perf_counter() - start
        now = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        with open(f'logs/{args.name}-logs.txt', 'a+') as f:
            f.write(f"{now} {status:<12} Run GPU: {os.environ.get('THEANO_FLAGS','device=CPU')[7:]:<5} PDB: {args.pdb:<36} duration: {time:0.3f}\n")


    print("\n" + "="*100)
    print("[INFO]  Maestro: prediction of change in stability!")
    print("="*100)
    csv_mut = str(args.pdb).replace('.pdb','.csv')
        
    file_path = f"{os.getcwd()}/{csv_mut}" #
    pdb_path = f"{os.getcwd()}/{csv_mut.replace('.csv','.pdb')}"
    pdb_ind = f"{os.getcwd()}/{csv_mut.replace('.csv','_individual_list.txt')}"
    
    fout = open(pdb_ind,'w')
    
    print (file_path)
        
    # Read data
    df = read_data(file_path)
    print("\n" + "="*100)
    print("[INFO]  Maestro: Reading the Mutcomput output!")
    print("="*100)
        
    prob_threshold = 0.80 #todo, make this an argument
    top_n = 100 # todo, make this an argument
    filtered_df = df[(df['avg_log_ratio'] > 0) & (df['pred_prob']>=prob_threshold)].sort_values(by=['pred_prob'], ascending=False).reset_index()
        
    if len(filtered_df) > top_n:
        filtered_df = filtered_df.head(100)

    print("\n" + "="*100)
    print("[INFO]  Convert columns and format evalmut!")
    print("="*100)
    # Convert columns and format evalmut
    filtered_df['evalmut'] = filtered_df.apply(format_evalmut, axis=1)

    # New column to store mutcompute results
    mutation_results = []

    print("\n" + "="*100)
    print("[INFO]  Running Maestro to each mutation!")
    print("="*100)
    # 6. Execute the system command for each evalmut
    for index, row in filtered_df.iterrows():
        evalmut = row['evalmut']
        output = execute_command(evalmut, pdb_path)
        mutation_result = process_output(output)
        result = convert_pattern(evalmut)
        fout.write(f"{result}\n")
        print("\n" + "="*100)
        print("[INFO] Add the Mutation result as a new row to the list!")
        print("="*100)
        # Add the mutation result as a new row to the list
        mutation_results.append(mutation_result)

    # Ensure correct number of columns (adjust if there are 9 values)
    mutation_results_df = pd.DataFrame(mutation_results, columns=['structure', 'seqlength', 'pH', 'mutation','score', 'delta_score', 'ddG', 'ddG_confidence'])

    # Concatenate the DataFrames
    final_df = pd.concat([filtered_df, mutation_results_df], axis=1)
    print("\n" + "="*100)
    print("[INFO] Save the DataFrame with the results!")
    print("="*100)
    # Save the DataFrame with the results
    csv_path = f"{pdb_path.replace('.pdb','_')}mut_maestro.csv"
        
    final_df.to_csv(csv_path, index=False)
    print(f"Results: {csv_path}")