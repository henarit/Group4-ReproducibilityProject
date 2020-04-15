# ========= Define and Open Clinical Data Paths ======================
clinical_data_path = '/content/drive/My Drive/DL Reproducibility Project/tmp/original_clinical_dataset.json'

with open(clinical_data_path) as f:
  clinical_data = json.load(f)

cases = []

# Get case id for each patient from submitter_id (12 char)
for i in range(len(clinical_data)):
  id = clinical_data[i]['demographic']['submitter_id'][0:12]
  cases.append(id)
print(cases[0:5])

# ========= Load Biospecimen data ===================
# used to find the connection between barcode-ids 'TCGA-60-2708'
# with the project names: 'PAAD', 'LUSC' etc

# Get project name from case/barcode (12 char)
biospec_path = '/content/drive/My Drive/DL Reproducibility Project/MultimodalPrognosis/data/processed/pancancer_biospecimen.csv'
csv_file = pd.read_csv(biospec_path)
csv_file['barcode2'] = csv_file['barcode'].str.slice(0, 12, 1) 

project = []

#find the connection between barcode-ids 'TCGA-60-2708'
# with the project names: 'PAAD', 'LUSC' etc
for case in cases:
  
  a = csv_file[csv_file['barcode2'] == str(case)]['project']
  if len(a) > 0:
    a = a.iloc[0]
    a = a[5:]
    project.append(a)
  else:
    project.append(None)


# IMPLEMENTATION: PART II

range_clinical_cases = np.arange(1, (len(clinical_data))) #starts at 1 because clinical_data[0] should be ignored

#Extracting meaningful clinical info into different dataframes that will be later concantenated
# First, primary diagnosis
primary_diagnosis = {'primary_diagnosis': [clinical_data[i]["diagnoses"][0]['primary_diagnosis'] for i in range_clinical_cases]}
df_primary = pd.DataFrame(primary_diagnosis, columns=['primary_diagnosis'])
# Second, vital status
vital_status = {'vital_status': [clinical_data[i]["demographic"]['vital_status'] for i in range_clinical_cases]}
df_vital = pd.DataFrame(vital_status, columns=['vital_status'])

# Third, days to death
days_to_death = {'days_to_death': [0 for i in range_clinical_cases]}
df_days = pd.DataFrame(days_to_death, columns=['days_to_death'])
for i in range_clinical_cases:
  if "days_to_death" in clinical_data[i-1]["demographic"]:
    x = [clinical_data[i-1]["demographic"]["days_to_death"]]
    df_days.iloc[i-2, df_days.columns.get_loc('days_to_death')] = x[0]

#Concatenating all dataframes into a single one containing, for each patient, primary diagnosis, vital status, days to death
result = pd.concat([df_primary, df_vital, df_days], axis=1, sort=False)


# IMPLEMENTATION: PART III

cancer_sites = ['ACC', 'BRCA', 'CESC', 'DLBC', 'KICH', 'KIRC', 'KIRP', 'OV', 'PCPG', 'PRAD', 'SARC', 'SKCM', 'TGCT', 'THCA', 'THYM', 'UCEC', 'BLCA', 'CHOL', 'COAD', 'ESCA', 'GBM', 'HNSC', 'LGG', 'LIHC','LUAD', 'LUSC', 'MESO', 'PAAD', 'READ', 'STAD', 'UCS', 'UVM']
day_range = np.arange(1,11000) # From day 1 to day 11000, starting at day 1 because at day at time 0 all patients are assumed to be alive
rango = np.arange(0, (len(project)-1)) # From patient 0 to patient 11166
prob_all_stacked = np.arange(1,11000) # It will store the survival probability at each time step, from 1 to 11
for x in cancer_sites:
  days_to_death_array = np.array([]) # It will store the days to death, for each patient case in each cancer site
  for i in rango:
    if (project[i] == x): # If the patient case corresponds to the x cancer site
      days_to_death = result.iloc[i, result.columns.get_loc('days_to_death')]
      days_to_death_array = np.append(days_to_death_array, days_to_death)
      number_patients = len(days_to_death_array) # The number of patient cases for this cancer site, in total
      probability = np.array([]) # It will contain the survival probabilities for this cancer site
  alive = number_patients; # Initial number of patients, which are alive at t = 0
  alive_array = np.zeros(11000) # Array containing the number of alive patients at each time step
  alive_array[0] = number_patients
  for r in day_range: # Loop going from day 1 to day 11000
    dead = 0 # This variable will store the number of dead patients at this timestep (r)
    for j in range(number_patients): 
      if days_to_death_array[int(j)] == r: # It goes through the array containing the days to death of each patient with this cancer site and checks whether they died at timestep r
        dead = dead +1 # If it died at r, it scores
        alive = alive - 1
    alive_array[r] = alive # It updates the number of alive patients at timestep r
    # Computing the survival probability at each time step, using the number of dead at this time step, and the number of alive patients up until this timestep
    if r == 1:
      probability = np.append(probability,(alive_array[0] - dead)/alive_array[0])
    else:
      p2 = (alive_array[r-1] - dead)/alive_array[r-1]
      probability = np.append(probability, probability[r-2]*p2)
  prob_all_stacked = np.vstack([prob_all_stacked, probability]) # Stacking in a matrix all cancer sites for storing purposes

