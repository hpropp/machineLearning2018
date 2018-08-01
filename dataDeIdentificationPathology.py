import pandas as pd
import re
import nltk
import sklearn
import pickle
import numpy as np
import xmltodict as xml
import random

file = "/media/hodaya/DISK_IMG/NWH/Pathology.xml"

# convert xml to dictionary using xmltodict
with open(file) as fd:
    doc = xml.parse(fd.read())

# preprocess the data (stripping special characters and lowercasing)
def preprocessing(data):
    preprocessed = []
    for i, val in enumerate(data['dataroot']['Pathology']): # i is report index
        empi, epic_pmrn, mrn_type, mrn,  = val['EMPI'], val['EPIC_PMRN'], val['MRN_Type'], val['MRN']
        num, date, typ = val['Report_Number'], val['Report_Date_Time'], val['Report_Type']
        stat, desc, text = val['Report_Status'], val['Report_Description'], val['Report_Text']
        text = re.sub('\W+', ' ', text).lower().strip() # taking out special characters and lowercasing
        preprocessed.append((empi, epic_pmrn, mrn_type, mrn, num, date, typ, stat, desc, text))
    return preprocessed

# deidentify the data (to protect the patient)
def deidentification(data):
    deidentified = []
    for line in data:
        for i in line:
            # confidential numbers
            i = re.sub(r'\b[0-9]{9}\b', r'EMPI_NUMBER', i) # format of #########
            i = re.sub(r'[0-9]{11}', r'EPIC_PMRN_NUMBER', i) # format of ###########
            i = re.sub(r'\b[0-9]{8}\b', r'MRN_NUMBER', i) # format of ########
            i = re.sub(r'([a-z]{1,2}|[A-Z]{1,2})[0-9]{2}(\s|:|-)[0-9]{3,6}', r'ACCESSION_NUMBER', i) # format of a## ######
            # names (formats)
            i = re.sub(r'(?<=ordering\sprovider\s)[a-z]{3,11}\s[a-z]{4,9}', r'ORDERING_PROVIDER', i) # ordering provider(space)lastname firstname
            i = re.sub(r'[a-z]{2,10}\s[a-z]{1,7}\s[a-z]{3,11}\s(md\b|m\sd\b|ct\sascp\b|ct\b|cnp\b|md\smba\b|np\b|fnp\b|dpm\b|do\b)', r'PHYSICIAN_NAME',i) # firstname m lastname 'md'/'m d'/'ct ascp'/'ct'
            i = re.sub(r'[a-z]{3,10}\s[a-z]{4,10}\s[a-z]\s(md\b|m\sd\b|ct\sascp\b|ct\b|cnp\b|md\smba\b|np\b|fnp\b|dpm\b|do\b)', r'PHYSICIAN_NAME',i) # lastname firstname m 'md'/'m d'/'ct ascp'/'ct'
            i = re.sub(r'[a-z]{3,10}\s[a-z]{3,11}\s(md\b|m\sd\b|ct\sascp\b|ct\b|md\smba\b|np\b|fnp\b|dpm\b|do\b)', r'PHYSICIAN_NAME',i) # firstname lastname 'md'/'m d'/'ct ascp'
            i = re.sub(r'(?<=patient\sname\s)[a-z]{4,11}\s[a-z]{3,10}\s[a-z]\b', r'PATIENT_NAME', i) # patient name(space)lastname firstname m
            i = re.sub(r'(?<=patient\sname\s)[a-z]{4,11}\s[a-z]{4,10}', r'PATIENT_NAME', i) # patient name(space)lastname firstname
            i = re.sub(r'(?<=patient\sname\s)[a-z]\s[a-z]{4,11}\s[a-z]{4,10}\s[a-z]\b', r'PATIENT_NAME', i) # patient name(space)m lastname firstname m
            i = re.sub(r'jane', r'NURSE_NAME', i)
            i = re.sub(r'shifren\sjan|dolloff\sann|klingenstein\sr|tromanhauser\sscott|gryska\sp', r'ORDERING_PROVIDER', i)
            i = re.sub(r'\balicia\b|\bantonia\b|\bamy\b|\bemily\b|\bemily\sy\b|elizabeth\sk|\bcatherine\sa\b|a\salan\ssemine|semine\sa|angela\sc\b|ellen|ellen\sf\b|joan\se\b|christine\s(m|c)|christine|\bdonovan\sm\b|\bkay\b|terri\sl|\blisa\sm', r'NAME', i)
            i = re.sub(r'leaf\sdob\b', r'NAME\sdob', i)
            # check that DATE TIME works

            i = re.sub(r'(?<=grossing\sstaff\s)[a-z]{2,3}\s[a-z]{2,3}',r'GROSSING_STAFF', i) # grossing staff(space)xxx xx/x
            i = re.sub(r'(?<=dictated\sby\s)[a-z]\s[a-z]{6,9}', r'PERSON_DICTATING', i) # informed by f lastname
            i = re.sub(r'(?<=diagnosis\sby\s)[a-z]{4,8}\s[a-z]\s[a-z]{4,9}\s(md\b|m\sd\b|ct\sascp\b|ct\b|cnp\b|md\smba\b|np\b|fnp\b|dpm\b|do\b)', r'PHYSICIAN_NAME', i) # informed by f lastname
            i = re.sub(r'(\bdr\s|\bmr\s|\bms\s|\bmrs\s)[a-z]{4,10}\s[a-z]\s[a-z]{4,8}', r'NAME', i) # dr/mr/ms/mrs firstname m lastname
            i = re.sub(r'(\bdr\s|\bmr\s|\bms\s|\bmrs\s)[a-z]{4,10}\s[a-z]{4,8}', r'NAME', i) # dr/mr/ms/mrs firstname lastname
            i = re.sub(r'(?<=dictated\sby\s)[a-z]{2,3}\s[a-z]{3}', r'PERSON_DICTATING', i) # dictated by xx/x xxx____
            i = re.sub(r'(?<=dictated\sby\s)[a-z]{2}', r'PERSON_DICTATING', i) # dictated by xx____
             #(not include certain words, any #'s, not 2 characters)
            i = re.sub(r'(?<=labeled\s)(?!endo|append|curett|assist|stomac|skin|tissu|specim|design|flex|sentin|body|duoden|epiderm|esoph|consis|fibroi|antrum|polyp|uter|lymph|medic|the|cervi|descend|with|poster|medial|pelv|colon|bowel|tumor|space|vagina|node|mucosa|ileum|wound|midlin|stern|breast|heal|biops|rectum|knuckl|left|right|tube|and)[a-z]{3,10}\s([a-z]|leaf|ellen|mcconney)\s[a-z]{3,11}\b', r'PERSON_LABELING', i) # labeled firstname m lastname
            i = re.sub(r'(?<=labeled\s)(?!endo|append|curett|assist|stomac|skin|tissu|specim|design|flex|sentin|body|duoden|epiderm|esoph|consis|fibroi|antrum|polyp|uter|lymph|medic|the|cervi|descend|with|poster|medial|pelv|colon|bowel|tumor|space|vagina|node|mucosa|ileum|wound|midlin|stern|breast|heal|biops|rectum|knuckl|left|right|tube|and)[a-z]{3,10}\s[a-z]{3,11}\b', r'PERSON_LABELING', i) # labeled firstname lastname
            i = re.sub(r'(?<=labeled\s)[a-z]\s[a-z]{3,10}', r'PERSON_LABELING', i) # labeled f lastname
            # date & time
            i = re.sub(r'([1-9]|[0][1-9]|[1][0-2])[\s|/]([0-9]|0[1-9]|[12][0-9]|3[01])[\s|/]([0-9]{2,4})(?!(\s[0-9]{2}){3,})', r'DATE', i) # m/dd/yy, mm/dd/yy, mm/dd/yyyy (/ or ' ')
            i = re.sub(r'(?<=DATE\s)[0-9]{2}\s[0-9]{2}(?!\s[0-9]{1,})', r'TIME', i) # m/dd/yy ## ##, mm/dd/yy ## ##, mm/dd/yyyy ## ## (/ or ' ')
            i = re.sub(r'[0-9]{1,2}:[0-9]{2}:[0-9]{2}\s(AM|PM)', 'TIME', i)
            # address & telephone
            i = re.sub(r'[0-9]{3,4}\swashington\s(street|st)\s[a-z]{6}\sma\s[0-9]{5}', r'HOSPITAL_ADDRESS', i)
            i = re.sub(r'tel\s617\s243\s6140', r'HOSPITAL_TELEPHONE', i)
            deidentified.append(i)
    # validate you have found all the cases (take 50 reports at random & check them)
    random_reports = random.sample(deidentified, 50)
    for r in range(49):
        report = random_reports[r]
        # are all names gone from reports?
        if re.search(r'[a-z]{2,10}\s[a-z]{1,7}\s[a-z]{3,11}\s(md\b|m\sd\b|ct\sascp\b|ct\b|cnp\b|md\smba\b|np\b|fnp\b|dpm\b|do\b)', report) != None: # firstname m lastname 'md'/'m d'/'ct ascp'/'ct'
            return "A case that wasn't deidentified was caught (1)"
        elif re.search(r'[a-z]{3,10}\s[a-z]{4,10}\s[a-z]\s(md\b|m\sd\b|ct\sascp\b|ct\b|cnp\b|md\smba\b|np\b|fnp\b|dpm\b|do\b)', report) != None: # lastname firstname m 'md'/'m d'/'ct ascp'/'ct'
            return "A case that wasn't deidentified was caught (2)"
        elif re.search(r'[a-z]{3,10}\s[a-z]{3,10}\s(md\b|m\sd\b|ct\sascp\b|ct\b|md\smba\b|np\b|fnp\b|dpm\b|do\b)', report) != None: # firstname lastname 'md'/'m d'/'ct ascp'/'ct'
            return "A case that wasn't deidentified was caught (3)"
        elif re.search(r'(?<=patient\sname\s)[a-z]{4,11}\s[a-z]{4,10}\s[a-z]\b', report) != None: # patient name(space)lastname firstname m
            return "A case that wasn't deidentified was caught (4)"
        elif re.search(r'(?<=patient\sname\s)[a-z]{4,11}\s[a-z]{4,10}', report) != None: # patient name(space)lastname firstname
            return "A case that wasn't deidentified was caught (5)"
        elif re.search(r'(?<=patient\sname\s)[a-z]\s[a-z]{4,11}\s[a-z]{4,10}\s[a-z]\b', report) != None: # patient name(space)m lastname firstname m
            return "A case that wasn't deidentified was caught (6)"
    return deidentified

preprocessed = preprocessing(doc)
#print(preprocessed)
deidentified = deidentification(preprocessed)
print(deidentified)

f = open("/home/hodaya/Downloads/testfile.txt", "w")
for line in deidentified:
    print >>f, line
    print
f.close()
