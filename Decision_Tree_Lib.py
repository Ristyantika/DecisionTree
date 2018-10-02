# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 11:03:39 2018

@author: ASUS
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# file: bacacsv.py
import csv

with open('data_banknote_authentication.csv','r') as csvfile:
    spamreader = csv.reader(csvfile)
    csvList = []
    csvData = []
    i = 0
    for data in spamreader:
        if len(data) != 0:
            i = i + 1
            if i == 1:
                continue
            else:
                csvList = csvList +[[int(x) for x in data[:4]]]
                csvData = csvData +[data[4]]

csvfile.close()
from sklearn import tree
Atribut = csvList
Class = csvData
clf = tree.DecisionTreeClassifier()
clf = clf.fit(Atribut[:500],Class[:500])


urut = 499
ya = 0
tidak = 0
for cek in Atribut[500:] :
    cekfix = clf.predict([cek])
    print cek
    urut = urut + 1
    '''print urut'''
    #print cekfix, Class[urut]
    if cekfix[0] == Class[urut]:
        ya = ya + 1
    else :
        tidak = tidak + 1
        
print tidak 
print ya

print float(ya/float(ya+tidak)) * 100

    
    
    




