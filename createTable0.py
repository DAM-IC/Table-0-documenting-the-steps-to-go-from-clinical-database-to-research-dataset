# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:46:55 2024

@author: Frank van Rosmalen, Jip W.T.M. de Kok

This script was used to extract and prepare data from the clinical source
database of the Intensive Care Unit of the Maastricht University Medical
Centre+. For reasons of readability and security, some database-specific 
properties have been generalised in the code. The scientific publication
associated with this code can be found here:
    https://doi.org/10.1016/j.jclinepi.2024.111342

This study was approved by the institutional review board of the MUMC+ (2021-2792)
"""

# Import required packages
import pandas as pd
import numpy as np
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import pyodbc

## First setup a database connection to the clinical source database (a SQL server)
conn = pyodbc.connect('DRIVER={SQL Server};\
                      SERVER=localhost;\
                      DATABASE=mydb')
                          
## Step 1. Start by pulling a list of all historical admissions from the system
qry = """
    SELECT 
        encounterId 
        ,clinicalUnitId 
        ,inTime 
        ,outTime 
        ,transferClinicalUnitId 
        ,admitType 
        ,dischargeDisposition 
        ,dischargeLocation 
        ,isDischarged 
        ,isDeceased 
        ,isTransferred 
        ,hasOpenChart 
    FROM PtCensus
    """               
    
listAll = pd.read_sql(qry, conn)  

## Step 2, remove admission outside of date range: 2013-2023 (so excluding 2023)
listAll = listAll.sort_values(by=['encounterId','inTime'])
# Define the cutoff date & remove rows older or younger than the cutoff date
maxDate = pd.to_datetime('2023-01-01')
minDate = pd.to_datetime('2013-01-01')
listAll = listAll[(listAll['inTime'] < maxDate)]
listAll = listAll[(listAll['inTime'] >= minDate)]

 ## Add date of birth
qry = """
    SELECT 
        encounterId
        ,da.shortLabel as Property 
        , storeTime 
        ,terseForm as DateOfBirth2 
    FROM PtDemographic PT 
        INNER JOIN D_Intervention DI 
            ON DI.interventionId = PT.interventionId  
        INNER JOIN D_Attribute DA  
            ON DA.attributeId = PT.attributeId   
    WHERE di.shortLabel = 'Geb. datum' 
    """
dobTab = pd.read_sql(qry, conn)
dobTab['DateOfBirth2'] = pd.to_datetime(dobTab['DateOfBirth2'],format = '%d-%m-%Y')
# The tables in the clinical source database add a row each time a value is changed
# (for example: when a correction is made). We therefore sort by encounter then by
# date, and only take the last entry as this is the most recent correction. 
# We know that 'backcharting' aka making a correction in retrospect, is only allowed
# by the system within 24 hours. We extract after >24h, so we will have the latest
# version.
dobTab = dobTab.sort_values(by=['encounterId', 'storeTime'])
dobTab = dobTab.drop_duplicates(subset='encounterId', keep='last')
listAll = pd.merge(listAll, dobTab[['encounterId', 'DateOfBirth2']], 
                    on='encounterId', how='left')

## Step 3: Remove 'wrong' patient numbers, test patients 
# Patient numbers can only contains numeric characters, covert to numeric and 
# convert everything with a letter in the text to NaN. It is common practice to
# indicate test patients with a 'number' starting with an X. These get filtered 
# out in this step

# Add patient numbers
qry = """
    SELECT 
        encounterId 
        ,lifetimeNumber 
        ,gender 
    FROM D_encounter
    """
encounterTable = pd.read_sql(qry, conn)
listAll = pd.merge(listAll, encounterTable[['encounterId',
                                            'lifetimeNumber','gender']],
                   on='encounterId', how='left')
listAll['lifetimeNumber'] = pd.to_numeric(listAll['lifetimeNumber'], errors='coerce')

# We have a list of 'bad patient numbers', this list contains test patients (not
# starting with an X as described above)  as well as odd cases where a wrong patient
# number is used. These cases are found because they do not match the standard 
# patient number format 
listOfBadPatientNumbers = pd.read_excel('C:/path/to/exclusiePatienten.xlsx')
# Remove admissions that are in the exclusion list, then remove admissions where
# lifetimeNumber ==NaN or that are not closed yet
listAll = listAll[~listAll['lifetimeNumber'].isin(listOfBadPatientNumbers['Patientnummer'])]
listAll = listAll.dropna(subset=['lifetimeNumber'])
listAll = listAll.dropna(subset=['outTime'])

# Housekeeping:
del encounterTable,maxDate,minDate,qry,listOfBadPatientNumbers

## Step 4: Remove non-adult ICU departments
# clinicalUnitId with corresponding hospital department:
# 12	VMC5
# 17	VMA3
# 5     VIB2
# 14    VEE2
listAll = listAll.loc[~listAll['clinicalUnitId'].isin([17,12])]
# For the NICU/PICU we remove all admission that have at least 1 part of the 
# admission in these wards, since these are not adults
nicuEncounters = listAll.loc[listAll['clinicalUnitId'].isin([14, 5]), 'encounterId']
listAll = listAll.loc[~listAll['encounterId'].isin(nicuEncounters)]

## Step 5: Check if all admissions have at least 1 recorded heart frequency
# Start by pulling all heart rate measurements, an only keep the ones that match
# our current list of admission
qry = """
    SELECT 
        encounterId 
        ,da.shortLabel as Property 
        ,chartTime 
        ,storeTime 
        ,valueNumber 
        , clinicalUnitId 
    FROM PtAssessment PT 
        INNER JOIN D_Intervention DI 
            ON DI.interventionId = PT.interventionId 
        INNER JOIN D_Attribute DA 
            on DA.attributeId = PT.attributeId 
    WHERE di.shortLabel = 'HF' 
    """
hrs = pd.read_sql(qry, conn)  
hrs = hrs.loc[hrs['encounterId'].isin(listAll['encounterId'])]
hrs = hrs.dropna(subset=['valueNumber'])
hrs = hrs.sort_values(by=['encounterId', 'chartTime'])
hrs.reset_index(inplace = True, drop = True)

# Check for 'dangling' measurements, by which we mean dangling in time. Sometimes
# a patient has one single heart frequency measurement long before admission or
# after discharge. This is a measurement of another patient who was connected to
# the same bed. We remove these measurements by ignoring all measurements >18 hours
# away from the 'bulk' of the measurements. 

unEnc = hrs['encounterId'].unique() # Get unique encounter IDs

removeInd = []

# Loop through unique encounter IDs
for thisEnc in unEnc:
    ind = hrs.index[hrs['encounterId'] == thisEnc]
    delta = hrs.loc[ind, 'chartTime'].diff()[1:] > timedelta(hours=18)    
    if len(delta) > 1:
        if delta.iloc[0] == True:
            removeInd.append(ind[0])
        elif delta.iloc[-1] == True:
            removeInd.append(ind[-1])

# Remove rows with indices in removeInd
hrs.drop(removeInd, inplace=True)

hasHr = listAll['encounterId'].isin(hrs['encounterId'])
listAll = listAll[hasHr]
hrs.reset_index(inplace = True,drop=True)
listAll.reset_index(inplace = True,drop=True)

# We add the first and last recorded heart rate datetime to use as start/stop 
# of the admission

# Initialize variables
start = np.full((len(listAll), 1), pd.NaT)
stop = np.full((len(listAll), 1), pd.NaT)
encs = np.full((len(listAll), 1), np.nan)

# Iterate through 'listAll' rows
for k in range(len(listAll)):
    encs[k] = listAll.loc[k, 'encounterId']
    thisIndex = hrs['encounterId'] == listAll.loc[k, 'encounterId']
    start[k] = hrs.loc[thisIndex, 'chartTime'].min()
    stop[k] = hrs.loc[thisIndex, 'chartTime'].max()

# Create 'hrStartStop' DataFrame
hrStartStop = pd.DataFrame({'encounterId': encs.flatten(), 'hrStart': start.flatten(),
                            'hrStop': stop.flatten()})

# Drop duplicates from 'hrStartStop'
hrStartStop = hrStartStop.drop_duplicates()

# Merge 'listAll' with 'hrStartStop' using outer join on 'encounterId'
listAll = listAll.merge(hrStartStop, on='encounterId', how='left')

# Housekeeping:
del qry, delta, ind, hasHr, nicuEncounters, removeInd, thisEnc,unEnc,k,encs,hrStartStop,start,stop,thisIndex

## Step 6: Check if there are admission with <=15 minutes in between, 
# we consider these as the same admission. So far, a transfer to another ICU 
# unit was considered a separate admission, we are now combining these.

# Convert clinicalUnitId and encounterId to strings
listAll['clinicalUnitId'] = listAll['clinicalUnitId'].astype(str)
listAll['encounterId'] = listAll['encounterId'].astype(str)

# Unique list of ICU patients
icuPatients = listAll['lifetimeNumber'].unique()

# Create an empty dataframe to store the results
readmissionListCond = pd.DataFrame(columns=listAll.columns.tolist() + [
    'previousOut', 'timeSinceLast'])

# Sort listAll by lifetimeNumber and hrStart
listAll = listAll.sort_values(by=['lifetimeNumber', 'hrStart'])
listAll.reset_index(inplace = True,drop=True)

# Initialize counters
remove_rows = []

# Iterate over unique ICU patients
for thisPat in icuPatients:
    tmpTab = listAll[listAll['lifetimeNumber'] == thisPat]
    tmpTab.reset_index(inplace = True, drop = True)
    # Check if there are multiple entries for the patient
    if len(tmpTab) > 1:
        # Convert prevOut to a pandas Series
        prevOut = np.array([pd.NaT] + tmpTab['hrStop'].iloc[:-1].tolist())
                 
        # Compute if admissions are within 15 minutes of eachother
        timeRule = (prevOut - tmpTab['hrStart'].tolist()) > pd.Timedelta(minutes=-15)
        timeRule = timeRule.tolist()[1:] + [False]        
        
        # Create start and stop indices to use in the loop
        diff_array = np.diff(np.concatenate(([0], timeRule)))
        start_indices = np.where(diff_array == 1)[0]
        
        diff_array = np.diff(np.concatenate((timeRule,[0])))
        stop_indices = np.where(diff_array == -1)[0]+1
        
        if len(start_indices) != 0: # Only do this if there is a start index
            remove_rows = [] # Reset
            # Loop and combine if admissions within 15 minutes of each other
             
            for startStopInd in range(len(start_indices)): 
                start = start_indices[startStopInd]
                stop = stop_indices[startStopInd]
                encounterId_unique = tmpTab['encounterId'].iloc[start:stop+1].unique()
                clinicalUnitId_unique = tmpTab['clinicalUnitId'].iloc[start:stop+1].unique()
                max_hrStop = tmpTab['hrStop'].iloc[start:stop+1].max()
                dischargeLocation = tmpTab['dischargeLocation'].iloc[stop]
                isDischarged = tmpTab['isDischarged'].iloc[stop]
                isDeceased = tmpTab['isDeceased'].iloc[stop]
                # Combine the data of the 2 admissions
                tmpTab.loc[start, 'encounterId'] = ','.join(encounterId_unique)
                tmpTab.loc[start, 'clinicalUnitId'] = ','.join(clinicalUnitId_unique)
                tmpTab.loc[start, 'hrStop'] = max_hrStop
                tmpTab.loc[start, 'dischargeLocation'] = dischargeLocation
                tmpTab.loc[start, 'isDischarged'] = isDischarged
                tmpTab.loc[start, 'isDeceased'] = isDeceased
                
                remove_rows.extend(range(start + 1, stop+1))
            
            tmpTab = tmpTab.drop(remove_rows)
        # Add information about the time between admission
        if len(tmpTab) > 1:
            tmpTab = tmpTab.sort_values(by='hrStart')
            tmpTab.reset_index(inplace = True, drop = True)
            previousOut = [pd.NaT] + tmpTab['hrStop'].iloc[:-1].tolist()
            timeSinceLast = (pd.to_datetime(previousOut) - pd.to_datetime(
                tmpTab['hrStart'])).astype('timedelta64[ns]')
            tmpTab['previousOut'] = previousOut
            tmpTab['timeSinceLast'] = timeSinceLast

        else:
            tmpTab['previousOut'] = pd.NaT
            tmpTab['timeSinceLast'] = pd.NaT        
        
    else:
        tmpTab['previousOut'] = pd.NaT
        tmpTab['timeSinceLast'] = pd.NaT

    readmissionListCond = pd.concat([readmissionListCond, tmpTab],
                                    ignore_index=True)    
    
## Step 7, check if patients are part of the Dutch national Intensive Care
# Registry (called NICE). We check if one of the following categories is filled 
# out (these are only used in the NICE registry so are empty otherwise). Variable
# definitions are as follows: 
    # herkomstPat = admission origin
    # catPat = Patient category
    # verwspec = admitting medical specialism
    # opnindic = admission indication
# This is done by pulling the information from the table, then sorting by unique
# encounter ID and 'storeTime'. Because only one value can be given for these 
# categories, the latest stored value is the actual value. This way we can account
# for corrections after the initial choise is scored.

qry = """
    SELECT 
        encounterId 
        ,da.shortLabel as Att 
        ,di.shortLabel as Int 
        ,terseForm 
        ,storeTime 
        ,chartTime 
        ,clinicalUnitId 
    FROM PtDemographic PT 
        INNER JOIN D_Intervention DI 
            ON DI.interventionId = PT.interventionId 
        INNER JOIN D_Attribute DA 
            on DA.attributeId = PT.attributeId 
    WHERE DI.longLabel = 'Herkomst' 
    """
herkomstPat = pd.read_sql(qry, conn)
herkomstPat = herkomstPat.sort_values(by=['encounterId', 'storeTime'])
herkomstPat.drop_duplicates(subset = 'encounterId',keep='last', inplace = True,
                            ignore_index = True)

qry = """
    SELECT 
        encounterId 
        ,da.shortLabel as Att 
        ,di.shortLabel as Int 
        ,terseForm 
        ,storeTime 
        ,chartTime 
        ,clinicalUnitId 
    FROM PtDemographic PT 
        INNER JOIN D_Intervention DI 
            ON DI.interventionId = PT.interventionId 
        INNER JOIN D_Attribute DA 
            on DA.attributeId = PT.attributeId 
    WHERE DI.longLabel = 'Categorie patiënt' 
    """
catPat = pd.read_sql(qry, conn)
catPat = catPat.sort_values(by=['encounterId', 'storeTime'])
catPat.drop_duplicates(subset = 'encounterId',keep='last', inplace = True,
                       ignore_index = True)

qry = """
    SELECT 
        encounterId 
        ,da.shortLabel as Att 
        ,di.shortLabel as Int 
        ,terseForm 
        ,storeTime 
        ,chartTime 
        ,clinicalUnitId 
    FROM PtDemographic PT 
        INNER JOIN D_Intervention DI 
            ON DI.interventionId = PT.interventionId 
        INNER JOIN D_Attribute DA 
            on DA.attributeId = PT.attributeId 
    WHERE DI.longLabel = 'Verwijzend specialisme' 
    """
verwSpec = pd.read_sql(qry, conn)
verwSpec = verwSpec.sort_values(by=['encounterId', 'storeTime'])
verwSpec.drop_duplicates(subset = 'encounterId',keep='last', inplace = True,
                         ignore_index = True)

qry = """
    SELECT 
        encounterId 
        ,da.shortLabel as Att 
        ,di.shortLabel as Int 
        ,terseForm 
        ,storeTime 
        ,chartTime 
        ,clinicalUnitId 
    FROM PtDemographic PT 
        INNER JOIN D_Intervention DI 
            ON DI.interventionId = PT.interventionId 
        INNER JOIN D_Attribute DA 
            on DA.attributeId = PT.attributeId 
    WHERE DI.longLabel = 'Opname-indicatie' 
    """
opnIndic= pd.read_sql(qry, conn)
opnIndic = opnIndic.sort_values(by=['encounterId', 'storeTime'])
opnIndic.drop_duplicates(subset = 'encounterId',keep='last', inplace = True, ignore_index = True)

allNice = pd.concat([verwSpec['encounterId'], catPat['encounterId'],
                     herkomstPat['encounterId'], opnIndic['encounterId']]).unique()
allNiceTable = pd.concat([verwSpec, catPat, herkomstPat, opnIndic]).sort_values(by='chartTime')
allNiceTableUn = allNiceTable[['encounterId', 'chartTime', 'clinicalUnitId']].drop_duplicates()

# The following loop checks if admissions have NICE data
firstNice = np.full(len(readmissionListCond), np.nan)
firstNiceAfd = np.full(len(readmissionListCond), np.nan)

# Iterate over rows of 'readmissionListCond'
for k, row in readmissionListCond.iterrows():
    thisEncounter = (row['encounterId'])
    thisEncounter = [int(x) for x in thisEncounter.split(",")]    
    
    thisUnits = row['clinicalUnitId']
    thisUnits = [int(x) for x in thisUnits.split(",")]       
    
    # Check if 'thisEncounter' is not empty
    if thisEncounter:
        # Get all clinical unit IDs for this encounter from 'allNiceTableUn'
        ind = allNiceTableUn['encounterId'].isin(thisEncounter)
        alleNiceClinId = allNiceTableUn.loc[ind, 'clinicalUnitId']
        
        if len(alleNiceClinId) > 0:
            firstNiceAfd[k] = alleNiceClinId.values[0]
            first = np.where(thisUnits == firstNiceAfd[k])[0]
            firstNice[k] = first[0] if len(first) > 0 else -1

# Create new columns for the firstNiceAfd, firstNice, and firstClinicalUnit values
readmissionListCond['firstNiceAfd'] = firstNiceAfd
readmissionListCond['firstNice'] = firstNice
readmissionListCond['firstClinicalUnit'] = readmissionListCond[
    'clinicalUnitId'].apply(lambda x: x[0] if len(x) > 0 else np.nan)

hasNice = ~readmissionListCond['firstNice'].isnull()
hasNoNice = readmissionListCond['firstNice'].isnull()

noNiceTable = readmissionListCond[hasNoNice]
readmissionListCond = readmissionListCond[hasNice]

# Housekeeping:
del qry,ind,k,start,stop, alleNiceClinId, allNice, catPat, clinicalUnitId_unique, \
diff_array, dischargeLocation, encounterId_unique,first,firstNice,firstNiceAfd, \
hasNice, hasNoNice, herkomstPat, icuPatients, isDeceased,isDischarged,listAll, \
    max_hrStop,timeSinceLast,tmpTab,verwSpec,timeRule,thisUnits,thisPat,thisEncounter, \
        stop_indices,startStopInd,start_indices,row,remove_rows,prevOut,opnIndic, \
            previousOut,

## Step 8, check if patients who are not in the NICE registry qualify as ICU
# patient based on treatment. We check for ICU medication

encIDString = ','.join(noNiceTable['encounterId'].explode().unique().astype(str))

qry = """
    SELECT 
        encounterId 
        , DI.shortLabel 
        , DA.shortLabel as attShortLabel
        , verboseForm 
        , chartTime 
        , clinicalUnitId 
    FROM PtMedication PT 
        INNER JOIN D_Intervention DI 
            ON DI.interventionId = PT.interventionId 
        INNER JOIN D_Attribute DA 
            ON DA.attributeId = PT.attributeId 
    WHERE encounterId IN ( """ + encIDString + ")"
   
medication= pd.read_sql(qry, conn)

# Filter to only look for perfusor medication, then filter for ICU-only meds
medication = medication[medication['shortLabel'].str.contains('03. Perfusor medicatie')]
medication = medication[medication['shortLabel'].str.contains(
    '|'.join(['Alteplase', 'Dexmedetomidine', 'Labetalol', 'Midazolam',
              'Propofol', 'Norepinefrine']), case=False)]

# Convert chartTime to datetime
medication['chartTime'] = pd.to_datetime(medication['chartTime'])

# Get unique encounterIds
unMed = medication['encounterId'].unique()
hasNiceCharacteristics = np.zeros(len(noNiceTable), dtype=bool)

noNiceTable.reset_index(inplace = True, drop = True)
# Check for every admission if ICU-medication was administered
for k in range(len(unMed)):
    thisEnc = medication['encounterId'] == unMed[k]
    medTimes = medication.loc[thisEnc, 'chartTime']
    pattern = r'\b{}\b'.format(unMed[k]) # use pattern matching because we are dealing with combined strings
    noNiceIndex = noNiceTable.index[noNiceTable['encounterId'].str.contains(pattern)]    
    for kkk in range(0,len(noNiceIndex)):
        _in = noNiceTable.loc[noNiceIndex[kkk], 'hrStart']
        _out = noNiceTable.loc[noNiceIndex[kkk], 'hrStop']
        if any(medTimes.between(_in, _out)):
            hasNiceCharacteristics[noNiceIndex[kkk]] = True

toAdd = noNiceTable[hasNiceCharacteristics]
noNiceTable = noNiceTable[~hasNiceCharacteristics]
readmissionListCond = pd.concat([readmissionListCond, toAdd])

# Next up we check if there is a registered 'minute volume' from a ventilator.
# This is used as a proxy to check if the patient is on a ventilator. 
encIDString = ','.join(noNiceTable['encounterId'].explode().unique().astype(str))

qry = """
    SELECT 
        encounterId 
        , da.shortLabel as Property 
        , chartTime 
        , storeTime 
        , valueNumber 
        , clinicalUnitId 
    FROM PtAssessment PT 
        INNER JOIN D_Intervention DI 
            ON DI.interventionId = PT.interventionId 
        INNER JOIN D_Attribute DA 
            on DA.attributeId = PT.attributeId 
    WHERE di.shortLabel = 'MV [gem]' 
    AND clinicalUnitId in (3,8,15,13,7,18,16)
    """ + "AND encounterId IN (" + encIDString + ")"
mvs= pd.read_sql(qry, conn)

unMvs = mvs['encounterId'].unique()
hasNiceCharacteristics = np.zeros(len(noNiceTable), dtype=bool)

noNiceTable.reset_index(inplace = True, drop = True)
for k in range(len(unMvs)):
    thisEnc = mvs['encounterId'] == unMvs[k]
    mvTimes = mvs.loc[thisEnc, 'chartTime']
    
    pattern = r'\b{}\b'.format(unMvs[k])
    noNiceIndex = noNiceTable.index[noNiceTable['encounterId'].str.contains(pattern)]    
    for kkk in range(0,len(noNiceIndex)):
        _in = noNiceTable.loc[noNiceIndex[kkk], 'hrStart']
        _out = noNiceTable.loc[noNiceIndex[kkk], 'hrStop']
        if any(mvTimes.between(_in, _out)):
            hasNiceCharacteristics[noNiceIndex[kkk]] = True

toAdd = noNiceTable[hasNiceCharacteristics]
noNiceTable = noNiceTable[~hasNiceCharacteristics]
readmissionListCond = pd.concat([readmissionListCond, toAdd])

# Next we check for APACHE IV diagnosis code
encIDString = ','.join(noNiceTable['encounterId'].explode().unique().astype(str))
qry = """
    SELECT 
        encounterId 
        , da.shortLabel as Property 
        , chartTime 
        , storeTime 
        , verboseForm 
        , clinicalUnitId 
    FROM PtDemographic PT 
        INNER JOIN D_Intervention DI 
            ON DI.interventionId = PT.interventionId 
        INNER JOIN D_Attribute DA 
            on DA.attributeId = PT.attributeId 
    WHERE di.shortLabel = 'APACHE IV Diagnosen' 
    """+ "AND encounterId IN (" + encIDString + ")"
    
apacheIVdiag= pd.read_sql(qry, conn)

unIdsApache = apacheIVdiag['encounterId'].unique()

noNiceTable.reset_index(inplace = True, drop = True)

# Find the latest storeTime for each unique encounterId
latest_storeTimes = apacheIVdiag.groupby('encounterId')['storeTime'].max()

# Filter the DataFrame to keep rows with the latest storeTime for each encounterId
apacheIVdiag = apacheIVdiag.groupby('encounterId').apply(lambda group: group[
    group['storeTime'] == latest_storeTimes[group.name]])
apacheIVdiag = apacheIVdiag[apacheIVdiag['verboseForm'] != 'None']

unApIv = apacheIVdiag['encounterId'].unique()
hasNiceCharacteristics = np.zeros(len(noNiceTable), dtype=bool)
for k in range(len(unApIv)):
    thisEnc = apacheIVdiag['encounterId'] == unApIv[k]    
    pattern = r'\b{}\b'.format(unApIv[k])
    noNiceIndex = noNiceTable.index[noNiceTable['encounterId'].str.contains(pattern)]    
    if any(noNiceIndex):
        hasNiceCharacteristics[noNiceIndex[kkk]] = True
            
toAdd = noNiceTable[hasNiceCharacteristics]
noNiceTable = noNiceTable[~hasNiceCharacteristics]
readmissionListCond = pd.concat([readmissionListCond, toAdd])

# Remove patients without ICU characteristics that stay in department C3 only,
# because we know that, after the previous checks, these patients are non-ICU patients
c3Indices = noNiceTable.index[noNiceTable['clinicalUnitId'] == '16']
noNiceTable.drop(c3Indices,axis=0,inplace=True)

## Step 9:
# The next part of the code happens outside of the script. The result stored in
# 'noNiceTable, is manually checked using the EPD to see if the intensivist was
# the main care provider for this admission. Admissions where the intensivist 
# was not the main care provider are removed and the result is imported in the same 
# form as it was exported (so the columns are equal to noNiceTable)

# The step after this step checks if there are missing values for date of birth,
# or gender, or if the date of birth is too far in the past (we used 1-1-1900 or 
# earlier as 1-1-1900) The admissions with missing data were exported and manually
# checked and corrected, the result of the correction were then loaded back in:

# Load the result of the manual corrections
readmissionListCond = pd.read_csv("C:/path/toreadmissionListAfterManualCorrections.csv")
# CSV tends to split columns, so we are merging them to get the same format
encounter_id_columns = [col for col in readmissionListCond.columns if col.startswith('encounterId')]
readmissionListCond['encounterId'] = readmissionListCond[encounter_id_columns].apply(
    lambda row: ','.join(row.dropna().astype(int).astype(str)), axis=1)
readmissionListCond.drop(columns=encounter_id_columns, inplace=True)

# CSV tends to split columns, so we are merging them to get the same format
clinId_id_columns = [col for col in readmissionListCond.columns if col.startswith('clinicalUnit')]
readmissionListCond['clinicalUnitId'] = readmissionListCond[
    clinId_id_columns].apply(lambda row: ','.join(row.dropna().astype(
        int).astype(str)), axis=1)
readmissionListCond.drop(columns=clinId_id_columns, inplace=True)

## Step 10, remove patients under 18 years of age at admission
readmissionListCond['DateOfBirth2'] = pd.to_datetime(readmissionListCond['DateOfBirth2'])
readmissionListCond['hrStart'] = pd.to_datetime(readmissionListCond['hrStart'])
remove_indices = []
for index, (start, stop) in enumerate(zip(readmissionListCond['DateOfBirth2'],
                                          readmissionListCond['hrStart'])):
    difference_in_years = relativedelta( stop, start).years
    if difference_in_years < 18:
        remove_indices.append(index) 

readmissionListCond = readmissionListCond.drop(remove_indices)

## Step 11: Readmissions after more than 18 hours were considered true readmissions. 
# Readmissions between 15 minutes and 18 hours were checked for a discharge letter,
# and if one was available, labeled as true readmissions. Remaining readmissions
# between 15 minutes and 18 hours without a discharge letter  were manually checked.

# We start by checken who has only 1 admission, we don't have to check these

# Group by 'lifetimeNumber' and count occurrences
grouped_counts = readmissionListCond['lifetimeNumber'].value_counts()
# Extract unique values with only one occurrence
one_admission_lifetime_numbers = grouped_counts[grouped_counts == 1].index.tolist()
# Filter encounters with only one admission
encounters_correct = readmissionListCond[readmissionListCond[
    'lifetimeNumber'].isin(one_admission_lifetime_numbers)]
# Filter encounters requiring further checking
encounters_to_check = readmissionListCond[~readmissionListCond[
    'lifetimeNumber'].isin(one_admission_lifetime_numbers)]

# Next we check which patients have only admissions split by >18 hours, these
# are correct as well
readmissionListCond['timeSinceLast'] = pd.to_timedelta(readmissionListCond['timeSinceLast'])
encounters_to_check['timeSinceLast'] = pd.to_timedelta(encounters_to_check['timeSinceLast'])
# Filter encounters to check for those with timeSinceLast greater than or equal to -18 hours
ltf_to_check = encounters_to_check.lifetimeNumber[encounters_to_check[
    'timeSinceLast'] >= timedelta(hours=-18)]
ltf_to_check.drop_duplicates(inplace = True)
# Find 'lifetimeNumber' values in encounters to check not in ltfToCheck
lft_correct = set(encounters_to_check['lifetimeNumber']).difference(ltf_to_check)
to_add = encounters_to_check[encounters_to_check['lifetimeNumber'].isin(lft_correct)]
encounters_correct = pd.concat([encounters_correct, to_add], ignore_index=True)
encounters_to_check = encounters_to_check[~encounters_to_check['lifetimeNumber'].isin(lft_correct)]

# Next we check if admissions are separated by a discharge letter, if they are
# this means that the next admission is a true admission and should not be considered
# the same admission as the previous admission. We check this on patient level,
# so if a patient has only clearly separated encounters, we move the patient from
# encounters_to_check to encounter_correct

encIDString = ','.join(encounters_to_check['encounterId'].explode().unique().astype(str))
qry = """
    SELECT 
    encounterId
    , DI.shortLabel
    , DA.shortLabel
    , terseForm,chartTime
    , storeTime
    , clinicalUnitId 
    FROM PtIntervention PT 
    INNER JOIN D_Intervention DI ON DI.interventionId = PT.interventionId
    INNER JOIN D_Attribute DA on DA.attributeId = PT.attributeId
    WHERE DI.shortLabel = 'Aanvraag ontslag rapport-(en) (Voorlopig)'
    AND DA.shortLabel = 'Type rapport'
"""    
briefVoorlopig = pd.read_sql(qry, conn)
# Convert chartTime to datetime
briefVoorlopig['chartTime'] = pd.to_datetime(briefVoorlopig['chartTime'])
briefVoorlopig['storeTime'] = pd.to_datetime(briefVoorlopig['storeTime'])
# Find the latest storeTime for each unique encounterId
latest_storeTimes = briefVoorlopig.groupby('encounterId')['storeTime'].max()

# Filter the DataFrame to keep rows with the latest storeTime for each encounterId
briefVoorlopig = briefVoorlopig.groupby('encounterId').apply
(lambda group: group[group['storeTime'] == latest_storeTimes[group.name]])

remainingLifeTimeNumbers = encounters_to_check['lifetimeNumber'].unique()
includeLftNrs = []

for thisNr in remainingLifeTimeNumbers:
    tmpTab = encounters_to_check[encounters_to_check['lifetimeNumber'] == thisNr]
    tmpTab.reset_index(inplace = True, drop = True)
    letterDate = pd.Series([pd.NaT] * len(tmpTab))  # preallocate with NaT

    for kk in range(len(tmpTab) - 1, -1, -1):
        thisEnc = tmpTab.iloc[kk]['encounterId']
        thisEnc_list = thisEnc.split(',')
        if len(thisEnc_list) > 1:
            thisEnc = thisEnc_list[-1]  # Take the last if there are multiple
        thisEnc = int(thisEnc)  # Convert to integer for matching
            
        letterTime = briefVoorlopig.loc[briefVoorlopig['encounterId'] == thisEnc, 'chartTime']
        if not letterTime.empty:
            letterDate.iloc[kk] = letterTime.values[0]

    lDates = pd.Series([pd.NaT] + letterDate.tolist()[:-1])
    previousEncounterClosedWithLetter = lDates < tmpTab['hrStop'].values
    previousEncounterMoreThan18hAway = tmpTab['timeSinceLast'] < timedelta(hours=-18)
    correctEncSplit = previousEncounterClosedWithLetter | previousEncounterMoreThan18hAway
    correctEncSplit.iloc[0] = True

    if correctEncSplit.sum() == len(correctEncSplit):
        # All encounters are correctly split
        includeLftNrs.append(thisNr)

# Include remaining LifeTimeNumbers that are correctly split
to_add = encounters_to_check[encounters_to_check['lifetimeNumber'].isin(includeLftNrs)]
encounters_correct = pd.concat([encounters_correct, to_add], ignore_index=True)
encounters_to_check = encounters_to_check[~encounters_to_check[
    'lifetimeNumber'].isin(includeLftNrs)]

# The remaining encounters_to_check are manually checked and combined if necessary
# We will now combine the result of this check with the rest of the correct encounters

# Load the result of the manual corrections
corrected = pd.read_csv("C:/path/tomanuallyCheckedSplitOrNotData.csv")
# Excel tends to split columns, so we are merging them to get the same format
encounter_id_columns = [col for col in corrected.columns if col.startswith('encounterId')]
corrected['encounterId'] = corrected[encounter_id_columns].apply(
    lambda row: ','.join(row.dropna().astype(int).astype(str)), axis=1)
corrected.drop(columns=encounter_id_columns, inplace=True)

# Excel tends to split columns, so we are merging them to get the same format
clinId_id_columns = [col for col in corrected.columns if col.startswith('clinicalUnit')]
corrected['clinicalUnitId'] = corrected[clinId_id_columns].apply(
    lambda row: ','.join(row.dropna().astype(int).astype(str)), axis=1)
corrected.drop(columns=clinId_id_columns, inplace=True)

# 'corrected' contains some extra columns used for the combining them, we remove
# these extra columns...
common_columns = encounters_correct.columns.intersection(corrected.columns)
corrected= corrected[common_columns]
# ...and add the result to encounters_correct
encounters_correct = pd.concat([encounters_correct, corrected], ignore_index=True)

# Step 12: In the last step we check if there are gaps of more than 18 hours in 
# our data. If these are present we split the admission into separate parts

# Initialize the columns
encounters_correct['maxDiffHr'] = pd.NaT
encounters_correct['hrNrOfMeasurements'] = np.nan
encounters_correct['hrLongDiffs'] = np.nan

table_ids = encounters_correct['encounterId']
hrs_enc_ids = hrs['encounterId']
hrs_chart_time = hrs['chartTime']

# Process each encounter
for k in range(len(encounters_correct)):
    encs_str = table_ids.iloc[k]
    encs = [int(e) for e in encs_str.split(',')]
    this_index = hrs_enc_ids.isin(encs)
    sorted_chart_time = hrs_chart_time[this_index].sort_values()
    hr_diff = sorted_chart_time.diff().dropna()

    encounters_correct.at[k, 'hrNrOfMeasurements'] = len(sorted_chart_time)
    if not hr_diff.empty:
        encounters_correct.at[k, 'maxDiffHr'] = hr_diff.max()
        encounters_correct.at[k, 'hrLongDiffs'] = (hr_diff.abs() > timedelta(hours=18)).sum()
    else:
        encounters_correct.at[k, 'maxDiffHr'] = pd.NaT
        encounters_correct.at[k, 'hrLongDiffs'] = 0
        
# Separate encounters
no_splits_needed = encounters_correct[encounters_correct['hrLongDiffs'] == 0]
splits_needed = encounters_correct[encounters_correct['hrLongDiffs'] == 1]
splits_needed.reset_index(inplace=True)

# Initialize splits_table
splits_table = pd.DataFrame(columns=['lifetimeNumber', 'encounterId', 'hrStart'
                                     'hrStop', 'DateOfBirth2', 'gender',
                                     'isDeceased', 'partStart', 'partStop'])

# Check for gaps >18 hours, and split the admission based on these gaps
for idx in range(len(splits_needed)):
    this_encounters = splits_needed.loc[idx, 'encounterId']
    enc_ids = [int(e) for e in this_encounters.split(',')]
    this_hrs = hrs[hrs['encounterId'].isin(enc_ids)].sort_values(by='chartTime')
    this_hrs.reset_index(inplace=True)
    
    hr_diff = this_hrs['chartTime'].diff()
    diffs = [0] + hr_diff[hr_diff > timedelta(hours=18)].index.tolist() + [len(this_hrs)-1]
    
    for kk in range(len(diffs) - 1):
        start = diffs[kk]
        stop = diffs[kk+1]
        
        new_row = {
            'lifetimeNumber': splits_needed.loc[idx, 'lifetimeNumber'],
            'encounterId': list(this_hrs['encounterId'].iloc[start:stop].unique()),
            'hrStart': splits_needed.loc[idx, 'hrStart'],
            'hrStop': splits_needed.loc[idx, 'hrStop'],
            'DateOfBirth2': splits_needed.loc[idx, 'DateOfBirth2'],
            'gender': splits_needed.loc[idx, 'gender'],
            'isDeceased': splits_needed.loc[idx, 'isDeceased'],
            'partStart': this_hrs['chartTime'].iloc[start],
            'partStop': this_hrs['chartTime'].iloc[stop - 1]
        }
        
        splits_table = pd.concat([splits_table, pd.DataFrame([new_row])],
                                 ignore_index=True)
        
# Add together splits_table and no_splits_needed to arrive at the final set:
common_columns = no_splits_needed.columns.intersection(splits_table.columns)
# Subset DataFrames to only include common columns
no_splits_needed_subset = no_splits_needed[common_columns]
splits_table_subset = splits_table[common_columns]
# Concatenate the subset DataFrames vertically
encounters_correct_and_split = pd.concat([no_splits_needed_subset,
                                          splits_table_subset],
                                         ignore_index=True)