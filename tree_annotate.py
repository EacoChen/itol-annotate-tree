#!/usr/bin/env python3

from Bio import SeqIO
import plotly.express as px
from customise_annote_file import *
import pandas as pd
import os

if not os.path.exists(os.path.join(os.getcwd(),'tree_anno')):
    os.makedirs(os.path.join(os.getcwd(),'tree_anno'))

records = [_ for _ in SeqIO.parse('ko_integrase_nr.anno.fasta','fasta')]

id2len = {_.id:len(_.seq) for _ in records}
out_text = simplebar(id2len, 'length','#9fc5e8')

with open('./tree_anno/all_len.text','w') as f:
    f.write(out_text)


df = pd.read_excel('/users/PMIU0100/chenhy520/whn/host-seed.xlsx')

df.columns = ['PhageName', 'Name', 'Accessionnumeber','size','id']


tax2id = {}
for x,i in enumerate(df.index):
    begin = int(df.iat[i,4].split('-')[0].split('.')[-1])
    end = int(df.iat[i,4].split('-')[-1].split('.')[-1])
    prefix = '.'.join(df.iat[i,4].split('-')[0].split('.')[0:-1])
    id_list = [df.iat[i,1]]
    for j in range(begin,end+1):
        id_list.append(prefix+'.'+str(j))
    tax2id[df.iat[i,0]]=id_list
print(tax2id)

# to color strip
id2sp = {}
for _ in records:
    for k, v in tax2id.items():
        if _.id.replace('_',':') in v:
            id2sp[_.id] = v[0]
            break
print(id2sp)

sp2color = {s:c for s,c in zip(sorted(set(id2sp.values())),px.colors.qualitative.T10+px.colors.qualitative.Alphabet)}

dataset = 'species color strip'
out_text = to_color_strip(id2sp,sp2color,dataset)
with open('./tree_anno/all_species_colorstrip.text','w') as f:
    f.write(out_text)

# to color range
id2sp = {}
for _ in records:
    for k, v in tax2id.items():
        if _.id.replace('_',':') in v:
            id2sp[_.id] = v[0].split('_')[0]
            break
sp2color = {s:c for s,c in zip(sorted(set(id2sp.values())),px.colors.qualitative.T10+px.colors.qualitative.Alphabet)}

out_text = to_color_range(id2sp,sp2color)
with open('./tree_anno/all_species_colorrange.text','w') as f:
    f.write(out_text)



