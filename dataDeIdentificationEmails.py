import pandas as pd
import re
import nltk
import sklearn
import pickle
import numpy as np
import xmltodict as xml
import random

file = "/media/hodaya/DISK_IMG/RPDR/RPDR/LMRNote.xml"

# convert xml to dictionary using xmltodict
with open(file) as fd:
    doc = xml.parse(fd.read())

# preprocess the data (stripping special characters and lowercasing)
def preprocessing(data):
    preprocessed = []
    for i, val in enumerate(data['dataroot']['LMRNote']): # i is report index
        empi, epic_pmrn  = val['EMPI'], val['EPIC_PMRN']
        date, record_id, stat = val['LMRNote_Date'], val['Record_Id'], val['Status']
        cod, inst = val['COD'], val['Institution']
        subj, text = val['Subject'], val['Comments']
        text = re.sub('\W+', ' ', text).lower().strip() # taking out special characters and lowercasing
        preprocessed.append((empi, epic_pmrn, date, record_id, stat, cod, inst, subj, text))
    return preprocessed

# deidentify the data (to protect the patient)
def deidentification(data):
    deidentified = []
    for line in data:
        for i in line:
            if i == None:
                i = 'None'
            # confidential numbers
            i = re.sub(r'\b[0-9]{9}\b', r'EMPI_NUMBER', i) # format of #########
            i = re.sub(r'\b[0-9]{10}\b', r'ACCT_NUMBER', i) # format of #########
            i = re.sub(r'[0-9]{11}', r'EPIC_PMRN_NUMBER', i) # format of ###########
            #i = re.sub(r'\b[0-9]{7,8}\b', r'MRN_NUMBER', i) # format of ########
            i = re.sub(r'([A-Za-z]{1,2})[0-9]{9}', r'ACCESSION_NUMBER', i) # format of a## ######
            i = re.sub(r'([a-z]{1,2}|[A-Z]{1,2})[0-9]{2}(\s|:|-)[0-9]{3,6}', r'ACCESSION_NUMBER', i) # format of a## ######
            # names (formats)
            # name lastname firstname
            # best NAME
            # send DATE
            # manager letter
            # seeing/see/saw firstname lastname
            # #/####, ##/####
            i = re.sub(r'(?<=ordering\sprovider\s)[a-z]{3,11}\s[a-z]{4,9}', r'ORDERING_PROVIDER', i) # ordering provider(space)lastname firstname 
            i = re.sub(r'(?<=ordering\sprovider\s)[a-z]{3,11}\s[a-z]{4,9}m\sdr\smd', r'ORDERING_PROVIDER', i) # ordering provider(space)lastname firstname
            # lastname firstname dr md, lastname firstname m dr, lastname firstname dr,
            i = re.sub(r'[a-z]{2,10}\s[a-z]\s[a-z]{3,11}\s(dr\b|m\sdr\b|dr\smd\b|md\b|m\sd\b|ct\sascp\b|ct\b|cnp\b|md\smba\b|p\b|n\sp\b|np\b|np\sc\smsn\socn\b)', r'PHYSICIAN_NAME',i) # firstname m lastname 'md'/'m d'/'ct ascp'/'ct'
            i = re.sub(r'[a-z]{2,10}\s[a-z]{4,11}\s[a-z]\s(dr\b|m\sdr\b|dr\smd\b|md\b|m\sd\b|ct\sascp\b|ct\b|cnp\b|md\smba\b|p\b|n\sp\b|np\b|np\sc\smsn\socn\b)', r'PHYSICIAN_NAME',i) # lastname firstname m 'md'/'m d'/'ct ascp'/'ct'
            i = re.sub(r'[a-z]{3,10}\s[a-z]{3,10}\s(dr\b|m\sdr\b|dr\smd\b|md\b|m\sd\b|ct\sascp\b|ct\b|md\smba\b|n\sp\b|np\b|np\sc\smsn\socn\b)', r'PHYSICIAN_NAME',i) # firstname lastname 'md'/'m d'/'ct ascp'
            i = re.sub(r'(?<=patient\sname\s)[a-z]{4,11}\s[a-z]{4,10}\s[a-z]\b', r'PATIENT_NAME', i) # patient name(space)lastname firstname m
            i = re.sub(r'(?<=hospital\sname\s)[a-z]{4,11}\s[a-z]{4,10}\s[a-z]', r'PATIENT_NAME', i)
            i = re.sub(r'(?<=patient\sprofile\s)[a-z]{4,11}\s[a-z]\s[a-z]{4,10}\b', r'PATIENT_NAME', i)
            i = re.sub(r'(?<=patient\sname\s)[a-z]{4,11}\s[a-z]{4,10}', r'PATIENT_NAME', i) # patient name(space)lastname firstname 
            i = re.sub(r'(?<=patient\sprofile\s)[a-z]{4,11}\s[a-z]{4,10}', r'PATIENT_NAME', i)
            i = re.sub(r'(?<=patient\sname\s)[a-z]\s[a-z]{4,11}\s[a-z]{4,10}\s[a-z]\b', r'PATIENT_NAME', i) # patient name(sp_______ ace)m lastname firstname m
            i = re.sub(r'(?<=grossing\sstaff\s)[a-z]{2,3}\s[a-z]{2,3}',r'GROSSING_STAFF', i) # grossing staff(space)xxx xx/x
            i = re.sub(r'(?<=dictated\sby\s)[a-z]\s[a-z]{6,9}', r'PERSON_DICTATING', i) # informed by f lastname
            i = re.sub(r'(?<=diagnosis\sby\s)[a-z]{4,8}\s[a-z]\s[a-z]{4,9}\s(md|m\sd|ct\sascp)', r'PHYSICIAN_NAME', i) # informed by f lastname
            i = re.sub(r'(\bdr\s|\bmr\s|\bms\s|\bmrs\s)[a-z]{4,10}\s[a-z]\s[a-z]{4,11}\b', r'NAME', i) # dr/mr/ms/mrs firstname m lastname
            i = re.sub(r'(\bdr\s|\bmr\s|\bms\s|\bmrs\s)[a-z]{4,10}\s(?!report)[a-z]{4,11}\b', r'NAME', i) # dr/mr/ms/mrs firstname lastname
            i = re.sub(r'(\bdr\s|\bmr\s|\bms\s|\bmrs\s)[a-z]{3,11}', r'NAME', i) # dr/mr/ms/mrs lastname
            i = re.sub(r'(\bDr\.\s|\bMr\.\s|\bMs\.\s|\bMrs\.\s)[A-Za-z]{3,11}', r'NAME', i) # dr/mr/ms/mrs lastname
            i = re.sub(r'(?<=dictated\sby\s)[a-z]{2,3}\s[a-z]{3,6}\s', r'PERSON_DICTATING', i) # dictated by xx/x xxx____
            i = re.sub(r'(?<=dictated\sby\s)[a-z]{2,5}', r'PERSON_DICTATING', i) # dictated by xx____
            i = re.sub(r'(?<=dictated\sby\s)[a-z]{2,5}\s[a-z]\s[a-z]{2}\s((dr\b|m\sdr\b|dr\smd\b|md\b|m\sd\b|ct\sascp\b|ct\b|cnp\b|md\smba\b|p\b|n\sp\b|np\b|np\sc\smsn\socn\b))', r'PERSON_DICTATING', i) # dictated by xx____
            i = re.sub(r'(?<=dear\s)[a-z]{4,10}', r'NAME', i)
            i = re.sub(r'(?<=result\smanager\sletter\s)[a-z]{3,10}\s[a-z]\s[a-z]{3,11}', r'NAME', i) # firstname m lastname
            i = re.sub(r'(?<=result\smanager\sletter\s)[a-z]{3,10}\s[a-z]{2,11}', r'NAME', i) # firstname lastname
            i = re.sub(r'(?<=labeled\s)(?!and|uterus|cervix|that|change|chew|endo)[a-z]{3,10}\s[a-z]\s[a-z]{3,11}\b', r'NAME', i) # labeled firstname m lastname
            i = re.sub(r'(?<=labeled\s)(?!and|uterus|cervix|that|change|chew|endo)[a-z]{3,10}\s[a-z]{3,11}\b', r'NAME', i) # labeled firstname lastname
            i = re.sub(r'(?<=from\s).*to.*(?=subject\b)', r'NAME to NAME sent DATE TIME ', i)
            i = re.sub(r'\bms\s(o|ho)\b', r'ms NAME', i)
            i = re.sub(r'thanks\s(db\b|el\b)', r'thanks NAME', i)
            i = re.sub(r'sincerely\s(jan|sheila|omar\sel\sabd\smd)', r'sincerely NAME', i)
            i = re.sub(r'(?<=sincerely\s)(?!yours|report_end|_|nursing\sstaff|thank|for)[a-z]{4,10}\s[a-z]\s[a-z]{4,11}', r'NAME', i)
            i = re.sub(r'(?<=sincerely\s)(?!yours|report_end|_|nursing\sstaff|thank|for)[a-z]{4,10}\s[a-z]{4,11}', r'NAME', i)
            i = re.sub(r'(?<=sincerely\s)(?!yours|report_end|_|nursing\sstaff|thank|for)[a-z]{4,10}', r'NAME', i)
            i = re.sub(r'(?<=signature\srequired\s)(perry\sg\san\sm\sd|shu\slu\sm\sd)', r'NAME', i)
            i = re.sub(r'[a-z]{4,12}\spartners\sorg', r'EMAIL', i)
            i = re.sub(
                r'\bmary\b|\bbreen\b|\bann\b|\bmorris\b|priest\b|\bpellows\b|\bpompeu\b|\blamaster\b|\bkaren|\bmorse\b|\bpartridge\b|\bsheila\b|\bleonard\b|           \
                \bgradone|\bgriffin|\blorna\b|\bstacy\b|\bstacey\b|\bkathy\b|\bchristine|\bjennifer|hume\b|\brourke\b|\bbaeringer\b\bgidget\b|\bbuckley\b|\bcullins\b|          \
                \bkidd\b|\bnancy\b|\bbaldwin|\bshaughnessy|\bdiane\b|\bmarianne\b|\bstempel\b|\bkristin\b|\bbrawders\b|\bmegan\b|\baimee\b|\bsusan\b|\bholly\b|     \
                \bmeredith\b|\bsameh\b|\bmesallum\b|\bjoseph\sfranses\b|\bjulie\b|\blu\sm\sd\b|\blilly\b|\bschwan\b|\brosenwald\b|\bpaola\b|\bbattista\b|             \
                 \blilly\b|\bschwan\b|\brosenwald\b|catherine|bieksha|ayan|jacky|yolanda|conway|donna|\
                \bgail\b|\bsaulnier|\bdenise\b|\blucy\b|\bschumer\b|\barabie\b|\bknight\b|\bharriet\b|\bporter\b|\blauren\b|\bmitchell\b|\btaryn\b', r'NAME', i) # names that don't follow patterns
            i = re.sub(
                r'\bangela\b|\bdeborah|\bwalsh\b|\bjose\b|\bthurman\b|\breed\b|\bkelly\b|\bcarmelina\b|\brachelle\b|\bdebra\b|\bhotz\b|\bbogdanovitch|\bshari\b|\btracy\b|         \
                \bdiana\b|\bshapiro|\bmcdonnell|\bstephanie\b|\bstephen\b|\bsteph\b|\bperry\b|\bortiz\b|\bcheri\b|\brubin\b|\bge\b|\bkimberly\b|\bmcelwain|\bcoombs\b|       \
                \bkeith\b|\banne\b|\bnick\b|\brob\b|\bsimone\b|\bscott\b|\bcarla\b|\bdeanna\b|\bheywood\b|\bcaroline\b|\bbarbara\b|\bjennifer\b|\bjodi\b|\bdoyle\b|             \
                \bsheila\b|\bpatricia\b|\bjudy\b|\bmaryellen\b|\bweinfeld\b|\bstafford\b|\bnielsen\b|\bheydrich\b|\bstacey\b|\bfuerstman\b|\bdiana\b|\bmark\b|\bjody\b|\
                \bsally\b|\blaura\b|\blinda\b|\bannemarie\b|\bkim\b|\bellen\b|\bwebber\b|\bilyse\b|\bsharon\b|\bkathryn\b|\bkathleen\b|j\smastroianni\snp', r'NAME', i)
            # date & time 


            i = re.sub(
                r'((?<!bmi\s[0-9]{2}\s)|(?<![0-9]\s[0-9]\s[0-9]\s[0-9]\s)|(?<![0-9]{2}\s[0-9]\s[0-9]\s[0-9]\s)|(?<![0-9]\s[0-9]{2}\s[0-9]\s[0-9]\s)|(?<![0-9]\s[0-9]\s[0-9]\s[0-9]{2}\s)|           \
                (?<![0-9]{2}\s[0-9]{2}\s[0-9]{2}\s[0-9]{2}\s)|(?<![0-9]\s[0-9]\s[0-9]{2}\s[0-9]\s)|(?<![0-9]{2}\s[0-9]{2}\s[0-9]\s[0-9]\s)|(?<![0-9]{2}\s[0-9]\s[0-9]{2}\s[0-9]\s)|                                   \
                (?<![0-9]{2}\s[0-9]\s[0-9]\s[0-9]{2}\s)|(?<![0-9]\s[0-9]{2}\s[0-9]{2}\s[0-9]\s)|(?<![0-9]\s[0-9]{2}\s[0-9]\s[0-9]{2}\s)|(?<![0-9]\s[0-9]\s[0-9]{2}\s[0-9]{2}\s)|                                   \
                (?<![0-9]{2}\s[0-9]{2}\s[0-9]{2}\s[0-9]\s)|(?<![0-9]\s[0-9]{2}\s[0-9]{2}\s[0-9]{2}\s)|(?<![0-9]{2}\s[0-9]\s[0-9]{2}\s[0-9]{2}\s)|(?<![0-9]{2}\s[0-9]{2}\s[0-9]\s[0-9]{2}s))(\b[1-9]|[0][1-9]|[1][0-2])[\s/]([0-9]|0[1-9]|[12][0-9]|3[01])[\s/]([0-9]{2,4})((?!(\s[0-9]{2}){3,})|(?!\spm|am))', r'DATE', i)
                 # m/dd/yy, mm/dd/yy, mm/dd/yyyy (/ or ' ') 



            i = re.sub(r'(?<=DATE\s)[0-9]{2}\s[0-9]{2}(?!\s[0-9]{1,})', r'TIME', i)
            i = re.sub(r'[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}', r'DATE TIME', i)
            i = re.sub(r'[0-9]{3,4}(am|pm)', 'TIME', i)
            i = re.sub(r'[0-9]{1,2}\s[0-9]{2}(am|pm)', 'TIME', i)
            i = re.sub(r'[0,9]{2}\s[0-9]{2}\s[0-9]{2}\s(am|pm)', 'TIME', i)
            i = re.sub(r'(january|february|march|april|may|june|july|august|september|october|november|december)\s[0-9]{1,2}\s[0-9]{4}', r'DATE', i)
            i = re.sub(r'(january|february|march|april|may|june|july|august|september|october|november|december)\s[0-9]{4}', r'DATE', i)
            i = re.sub(r'(january|february|march|april|may|june|july|august|september|october|november|december)\s[0-9]{1,2}', r'DATE', i)
            i = re.sub(r'(january|february|march|april|may|june|july|august|september|october|november|december)\s[0-9]{1,2}(st|nd|rd|th)', r'DATE', i)
            i = re.sub(r'(?<=result\smanager\sletter\sDATE\s)[a-z]{3,10}\s[a-z]\s[a-z]{3,11}', r'NAME', i) # firstname m lastname
            i = re.sub(r'(?<=result\smanager\sletter\sDATE\s)[a-z]{3,10}\s[a-z]{3,11}', r'NAME', i) # firstname lastname
            # address & telephone
            i = re.sub(r'(?<=result\smanager\sletter\sDATE\sNAME\s).*(?=EMPI_NUMBER|DATE|dear)', r'ADDRESS ', i)
            i = re.sub(r'(?<=result\smanager\sletter\sNAME\s).*(?=EMPI_NUMBER|DATE|dear)', r'ADDRESS ', i)
            i = re.sub(r'[0-9]{1,4}\s([a-z]{3,11}|[a-z]{4,5}\s[a-z]{4,5})\s(road|rd|ave|way|street|st)\s([a-z]{5,8}|west\sroxbury)\sma\s[0-9]{5}', r'HOSPITAL_ADDRESS', i)
            i = re.sub(r'[0-9]{3}\s[0-9]{3}\s[0-9]{4}', r'TELEPHONE', i)
            deidentified.append(i)
    # validate you have found all the cases (take 50 reports at random & check them)
    random_reports = random.sample(deidentified, 50)
    for r in range(49):
        report = random_reports[r]
        # are all names gone from reports?
        if re.search(r'[a-z]{2,10}\s[a-z]{1,7}\s[a-z]{3,11}\s((dr\b|m\sdr\b|dr\smd\b|md\b|m\sd\b|ct\sascp\b|ct\b|cnp\b|md\smba\b|p\b|n\sp\b|np\b|np\sc\smsn\socn\b))', report) != None: # firstname m lastname 'md'/'m d'/'ct ascp'/'ct'
            return "A case that wasn't deidentified was caught (1)"
        elif re.search(r'[a-z]{3,10}\s[a-z]{4,10}\s[a-z]\s((dr\b|m\sdr\b|dr\smd\b|md\b|m\sd\b|ct\sascp\b|ct\b|cnp\b|md\smba\b|p\b|n\sp\b|np\b|np\sc\smsn\socn\b))', report) != None: # lastname firstname m 'md'/'m d'/'ct ascp'/'ct'
            return "A case that wasn't deidentified was caught (2)"
        elif re.search(r'[a-z]{3,10}\s[a-z]{3,10}\s((dr\b|m\sdr\b|dr\smd\b|md\b|m\sd\b|ct\sascp\b|ct\b|cnp\b|md\smba\b|p\b|n\sp\b|np\b|np\sc\smsn\socn\b))', report) != None: # firstname lastname 'md'/'m d'/'ct ascp'/'ct'
            return "A case that wasn't deidentified was caught (3)"
        elif re.search(r'(?<=patient\sname\s)[a-z]{4,11}\s[a-z]{4,10}\s[a-z]\b', report) != None: # patient name(space)lastname firstname m
            return "A case that wasn't deidentified was caught (4)"
        elif re.search(r'(?<=patient\sname\s)[a-z]{4,11}\s[a-z]{4,10}', report) != None: # patient name(space)lastname firstname
            return "A case that wasn't deidentified was caught (5)"
        elif re.search(r'(?<=patient\sname\s)[a-z]\s[a-z]{4,11}\s[a-z]{4,10}\s[a-z]\b', report) != None: # patient name(space)m lastname firstname m
            return "A case that wasn't deidentified was caught (6)"
    return deidentified

print(len(doc['dataroot']['LMRNote']))
preprocessed = preprocessing(doc)
deidentified = deidentification(preprocessed)

f = open("/media/hodaya/DISK_IMG/RPDR/RPDR/testfile2.txt", "w")
for line in deidentified:
    print >>f, line
    print
f.close()

'''
take 10 mrn numbers and find the corresponding emails
take this set and delete the rest, deidentify this set
then write a script to run through the rest of the data and make it learn from the set that's anonymized
adam will run it on his laptop - since he has access
draft an email to hannah and megan about access to the vpn since the anonymization of emails is taking a while
'''
'''
no hepatoegaly extremities no edema results DATE na 143 k 4 0 cl 115 h co2 23 bun 11 cre 0 7 egfr 101 2 egfr 
122 5 glu 81 DATE anion DATE TIME11 wbc 3 1 l rbc 4 28 hgb 13 4 hct 38 1 mcv 89 0 mch 31 3 mchc 35 2 plt clumped 
normal DATE mpv 10 0 rdw DATE TIME11 method automatic neut 46 lymph 49 h mono 4 eos 0 baso 1 bands 10 plt clmps
present DATE aneut 1 42 l alymp 1 51 amons 0 12 aeos a 0 abaso a 0 DATE 2011 htube results DATE alt sgpt 15 ast 
sgot 19 alkp 65 tbili 1 0 dbili 0 DATE 2010 alt sgpt 30 ast sgot 26 alkp 64 tbili 0 4 dbili 0 DATE 2010 chol 142 
trig 39 hdl 55 ldl 79 chol hdl 2 5 rrsk ldl 0 DATE 2011 tsh 1 741 electrocardiogram DATE sinus rhythm at a rate of 
52 bpm 55 degrees intervals 0 14 0 DATE no evidence of chamber enlargement or past myocardial infarction in comparison
'''