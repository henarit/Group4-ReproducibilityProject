import json
import numpy as np
import pandas as pd
from os import listdir
from os.path import join

# ========= DEFINE DATA PATHS ======================
data_path = '/content/drive/My Drive/DL Reproducibility Project/data'

mirna_data_path = join(data_path, 'miRNA')
rnaseq_data_path = join(data_path, 'rnaseq')
biospec_path = join(data_path, 'pancancer_biospecimen.csv')
clinical_data_path = join(data_path, 'original_clinical_dataset.json')

# ========= Load Biospecimen data ===================
# used to find the connection between barcode-ids 'TCGA-60-2708'
# with the project names: 'PAAD', 'LUSC' etc

csv_file = pd.read_csv(biospec_path)

case_project = pd.DataFrame({'case' : [], 'project' : []})
case_project['case'] = csv_file['barcode'].str.slice(0, 12, 1) 
case_project['project'] = csv_file['project'].str[5:]

# remove duplicates
case_project = case_project.drop_duplicates(ignore_index = True)

# create dictionary with barkodes as keys: 'TCGA-60-2708' 
# and project names as values: 'LUSC'
cp_dict = dict(zip(case_project.case, case_project.project))

# Get project name from case/barcode (12 char)
def case_to_project(case):
  if case in cp_dict:
    return cp_dict[case]
  else:
    return None

# ========= define set of cases =======================
cases = list(set(case_project.case))

def case_list(project='PAAD'):
	return [case for case in cases if case_to_project(case) == project]

# ========= Load RNAseq and miRNA data to RAM ==========

# miRNA
mirna_locs = {} # paths to mirna pkl files
mirna_projects = [] # list of projects that we have data for

for f in listdir(mirna_data_path):
  # the files have names such as: 'blood_ALL-P2_miRNA_data.pkl'
  f_split = f.split(sep='_')
  file_type = f_split[-1][0:-4] # data.pkl --> data

  if file_type == 'data':
    # project names are written in uppercase
    project = [word for word in f_split if word.isupper()][-1]
    # add path to dictionary
    mirna_locs[project] = join(mirna_data_path, f)
    # add project to list of projects with mirna data
    mirna_projects.append(project)

# RNAseq
rnaseq_locs = {} #  paths to rnaseq pkl files
rnaseq_projects = [] # list of projects that we have data for

for f in listdir(rnaseq_data_path):
  # the files have names such as: 'blood_ALL-P2_miRNA_data.pkl'
  f_split = f.split(sep='_')
  file_type = f_split[-1][0:-4] # data.pkl --> data

  if file_type == 'data':
    # project names are written in uppercase
    project = [word for word in f_split if word.isupper()][-1]
    # add path to dictionary
    rnaseq_locs[project] = join(rnaseq_data_path, f)
    # add project to list of projects with mirna data
    rnaseq_projects.append(project)

# Create dataframes of the mirna and rnaseq data (load into RAM)

print('loading RNAseq data ... takes a while')
rnaseq_data_dict = {}
for project in rnaseq_projects:
  # read data, which is stored in a dataframe
  tmp = pd.read_pickle(rnaseq_locs[project], compression='gzip')
  barcodes = tmp.index.str[0:12]
  tmp.set_index(keys=barcodes, inplace=True)
  rnaseq_data_dict[project] = tmp

print('loading miRNA data ... takes a short while')
mirna_data_dict = {}
for project in mirna_projects:
  # read data, which is stored in a dataframe
  tmp = pd.read_pickle(mirna_locs[project], compression='gzip')
  barcodes = tmp.index.str[0:12]
  tmp.set_index(keys=barcodes, inplace=True)
  mirna_data_dict[project] = tmp

# =========== Clinical data ======================

def load_clinical_data(clinical_data_path):
  # Loads all cases from the json file located at
  # Returns a dictionary of clinical data, where the keys
  # are strings such as e.g. 'TCGA-US-A77G'
  # Also returns "lookup-vectors" which can be used to
  # interpret what vital_status = 0 means etc
  with open(clinical_data_path) as f:
        clin = json.load(f)

  clinical_data = {}
  race_lookup = []
  gender_lookup = ['female', 'male']
  vital_status_lookup = ['dead', 'alive']
  disease_lookup = []

  for i in range(len(clin)):
    # gets id in the form: 'TCGA-US-A77G_demographic' (example), 
    # and selects the first 12 characters: 'TCGA-US-A77G'
    case = clin[i]['demographic']['submitter_id'][0:12]

    clinical_data[case] = {}

    # age
    age = clin[i]['demographic']['age_at_index']
    if isinstance(age, int):
      # sometimes the entry is a string such as 'not reported' 
      clinical_data[case]['age'] = age

    # race
    race = clin[i]['demographic']['race'].lower()
    if race not in race_lookup:
      race_lookup.append(race)
    race = race_lookup.index(race)
    clinical_data[case]['race'] = race
    
    # gender
    gender = clin[i]['demographic']['gender'].lower()
    if gender in gender_lookup:
      gender = gender_lookup.index(gender)
      clinical_data[case]['gender'] = gender

    # vital status
    vital_status = clin[i]['demographic']['vital_status'].lower()
    if vital_status in vital_status_lookup:
      vital_status = vital_status_lookup.index(vital_status)
      clinical_data[case]['vital_status'] = vital_status

    # days_to_death, if applicable
    if vital_status == 0 and 'days_to_death' in clin[i]['demographic']:
        days_to_death = clin[i]['demographic']['days_to_death']
        clinical_data[case]['days_to_death'] = days_to_death

    # histologic grade ... HOW SHOULD THIS BE SET? Ignore for now
    # histologic_grade = None
    # clinical_data[case]['histologic_grade'] = histologic_grade

    # disease / project
    project = case_to_project(case)
    if project not in disease_lookup:
      disease_lookup.append(project)
    disease = disease_lookup.index(project)
    clinical_data[case]['disease'] = disease

    lookups = {
        'race_lookup' : race_lookup,
        'gender_lookup' : gender_lookup,
        'vital_status_lookup' : vital_status_lookup,
        'disease_lookup' : disease_lookup
    }

  return clinical_data, lookups

# Load clinical data from file. For each case, we have a dictionary
# with some keys such as 'age', 'gender' etc
# lookups is a dictionary of 'lookup vectors'
print('loading clinical data ... is quick')
clinical_data_dict, lookups = load_clinical_data(clinical_data_path)

# Lookup vectors
race_lookup = lookups['race_lookup']
gender_lookup = lookups['gender_lookup']
vital_status_lookup = lookups['vital_status_lookup']
disease_lookup = lookups['disease_lookup']

def load(case):
  if case in clinical_data_dict:
    return clinical_data_dict[case]
  else:
    return {}

def clinical_data(case):
	patient_data = load(case)
	keys = ["gender", "race", "age", "disease"]
	if not all(key in patient_data for key in keys): return None
	return np.array([patient_data[key] for key in keys])

def clinical_data_expanded(case):
  # in the code they use this name, so we keep it for convenience
  return clinical_data(case)

def vital_status(case):
  # returns vital status for patient/case
	patient_data = load(case)
	if "vital_status" not in patient_data: return None
	return patient_data['vital_status']

def days_to_death(case):
  # returns days to death for patient/case
	patient_data = load(case)
	if "days_to_death" not in patient_data: return None
	return patient_data['days_to_death']

def cancer_type(case):
  # returns the cancer type for a patient/case
  project = case_to_project(case)
  if project not in disease_lookup:
    disease_lookup.append(project)

  return disease_lookup.index(project)

# =========== Fetch gene data =============

def gene_data(case):
  # Returns numpy array of rnaseq data for case
  project = case_to_project(case)

  # Check if case is associated with some project
  if project == None:
    return None

  # Return data if available, otherwise return None
  if project in rnaseq_data_dict and case in rnaseq_data_dict[project].index:
    tmp = np.array(rnaseq_data_dict[project].loc[case])

    # sometimes there are several measurements ... 
    # in that case take the last one ... (choice not mentioned in paper)
    if tmp.ndim > 1:
      return tmp[-1,:]
    else:
      return tmp
  else:
    return None

# =========== Fetch miRNA data =============

def mirna_data(case):
  # Returns numpy array of mirna data for case
  project = case_to_project(case)

  # Check if case is associated with some project
  if project == None:
    return None

  # Return data if available, otherwise return None
  if project in mirna_data_dict and case in mirna_data_dict[project].index:
    tmp = np.array(mirna_data_dict[project].loc[case])

    # sometimes there are several measurements ... 
    # in that case take the last one ... (choice not mentioned in paper)
    if tmp.ndim > 1:
      return tmp[-1,:]
    else:
      return tmp
  else:
    return None

# =========== Fetch slides data =============

# not implemented
def sample_from_slides(case):
  return False




