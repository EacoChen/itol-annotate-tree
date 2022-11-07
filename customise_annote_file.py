# -*- coding: UTF-8 -*-

import pandas as pd
import plotly.express as px
import os
import re
import numpy as np
import matplotlib as mpl

dbpath = os.path.join(os.path.abspath(os.path.dirname(__file__)),'itol_template')


def get_used_sep(text):
    separator = [_ for _ in text.split("\n") if _.startswith("SEPARATOR")]
    assert len(separator) == 1
    separator = separator[0].strip()
    sep = separator.split(" ")[1]
    if sep == "TAB":
        return "\t"
    elif sep == "SPACE":
        return " "
    elif sep == "COMMA":
        return ","


def replacing_params(text, kwarg={}):
    sep= get_used_sep(text)
    for k, v in kwarg.items():
        if k.upper() in ['MARGIN',"STRIP_WIDTH"]:
            row = [_ for _ in text.split('\n') if k.upper() in _]
            assert len(row) == 1
            if row:
                text = text.replace(row[0],sep.join([k.upper(),str(v)])+'\n' )
        else:
            text = text.replace(k, v)
    return text



def find_proteoclass(df):
    phy_class = []
    for index, row in df.iterrows():
        phylum = row['phylum']
        tax = row['tax']
        if phylum == 'Proteobacteria':
            phy_class.append(tax.split(sep='_')[2])
            # phy_class.append(re.match(r'([a-zA-Z0-9]+)_([a-zA-Z0-9]+)_([a-zA-Z0-9]+)\S*',tax).group(3))
        else:
            phy_class.append(phylum)

    df['phy/class'] = phy_class
    return df


def std_legend(title,info,sep=','):

    legend_title = title
    legend_shape = sep.join( ['1'] * len(info) )
    legend_color = sep.join( info.values() )
    legend_label = sep.join( info.keys() )

    legend_text = f"""
LEGEND_TITLE{sep}{legend_title}
LEGEND_SHAPES{sep}{legend_shape}
LEGEND_COLORS{sep}{legend_color}
LEGEND_LABELS{sep}{legend_label}"""

    return legend_text


def deduced_legend2(info2style, infos, same_colors=False, full=False, sep="\t"):
    # for info2style instead of info2color
    
    colors_theme = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
    shapes = []
    labels = []
    colors = []
    for idx, info in enumerate(infos):
        shapes.append(info2style[info].get("shape", "1"))
        labels.append(info2style[info].get("info", info))
        if not same_colors:
            if full:
                colors.append(info2style[info].get("color"))
            else:
                colors.append(info2style[info].get("color", colors_theme[idx]))
        else:
            colors.append(info2style[info].get("color", same_colors))
    legend_text = [
        "FIELD_SHAPES" + sep + sep.join(shapes),
        "FIELD_LABELS" + sep + sep.join(labels),
        "FIELD_COLORS" + sep + sep.join(colors),
    ]
    legend_text = "\n".join(legend_text)
    return legend_text


def to_color_strip(acc2info,info2color,dataset):
    # 该部分可封装
    # 获取访问码与颜色的对应
    acc2color = {acc: info2color[phyla] for acc, phyla in acc2info.items()}

    # data 注释主体输出
    annote_text = '\n'.join(['%s,%s,%s\n' % (acc, color, acc2info[acc])  # 因为注释跟的是tax而不是phylum，所以此处没有封装
                             for acc, color in acc2color.items()])

    owned_info2color = {info:info2color[info] for acc,info in acc2info.items()}

    # legend 形成，以及整个dataset的名字
    legend_text = std_legend(dataset, owned_info2color)
    dataset_label = dataset

    # 打开模板文件
    
    template_text = open(os.path.join(dbpath,'dataset_color_strip_template.txt')).read()

    # 输入到预先修改为format格式的文本文件
    template_text = template_text.format(legend_text=legend_text,
                                         dataset_label=dataset_label)

    # 形成最后输出文件
    out_text = template_text + '\n' + annote_text

    return out_text



def to_color_style(acc2info,info2color,dataset,pos,bg=False):
    # 该部分可封装
    # 获取访问码与颜色的对应
    acc2color = {acc: info2color[phyla] for acc, phyla in acc2info.items()}
    
    if not bg:
        # data 注释主体输出
        annote_text = '\n'.join(['%s,%s,%s,1,normal' % (acc, pos, color)  
                                 for acc, color in acc2color.items()])
    else:
        annote_text = '\n'.join(['%s,%s,#ffffff,1,normal,%s' 
                                 % (acc, pos, color) 
                                 for acc, color in acc2color.items()])

    owned_info2color = {info:info2color[info] for acc,info in acc2info.items()}

    # legend 形成，以及整个dataset的名字
    legend_text = std_legend(dataset, owned_info2color)
    dataset_label = dataset

    # 打开模板文件
    
    template_text = open(os.path.join(dbpath,'dataset_styles_template.txt')).read()

    # 输入到预先修改为format格式的文本文件
    template_text = template_text.format(legend_text=legend_text,
                                         dataset_label=dataset_label)

    # 形成最后输出文件
    out_text = template_text + '\n' + annote_text

    return out_text



def to_color_range(acc2info, info2color, pos='range'):

    acc2color = {acc: info2color[phyla] for acc, phyla in acc2info.items()}
    if pos == 'range':
        anno_text = '\n'.join(['%s,%s,%s,%s' % (acc, pos, color, acc2info[acc])
                              for acc,color in acc2color.items()])
    else:
        anno_text = '\n'.join(['%s,%s,%s,normal,3' % (acc, pos, color)
                              for acc,color in acc2color.items()])
    
    
    template_text = open(os.path.join(dbpath,'colors_styles_template.txt')).read()

    # 形成最后输出文件
    out_text = template_text + '\n' + anno_text

    return out_text


def highlight_outgroup(acc2type):
    og_dict = {}
    for k, v in acc2type.items():
        if re.match(r'.*outgroup$', v):
            og_dict.update({k:v})

    
    template_text = open(os.path.join(dbpath,'colors_styles_template.txt')).read()

    annote_text = '\n'.join(['%s label_background rgba(81,202,76,0.78)\n%s label rgba(0,0,0,1)' %(acc,acc)
                             for acc,outgroup in og_dict.items()])

    out_text = template_text + '\n' + annote_text

    return out_text


def simplebar(acc2len,name,color='#707FD1'):
    
    template_text = open(os.path.join(dbpath,'dataset_simplebar_template.txt')).read()

    bar_color = color
    dataset_label = name

    template_text = template_text.format(bar_color=bar_color,
                                         dataset_label=dataset_label)

    annote_text = '\n'.join(['%s,%d' %(acc,len) for acc,len in acc2len.items()])

    out_text = template_text + '\n' + annote_text

    return out_text


# def to_binary(acc2info, label='binary', sep=','):
#     shape = sep.join(range(1,len(set(list(acc2info.values())))+1))
#     field_labels = sep.join(list(acc2info.values()))

    
#     template_text = open(os.path.join(dbpath,'dataset_binary_template.txt')).read()

#     template_text = template_text.format(shape=shape,
#                                          field_labels=field_labels,
#                                          dataset_label=label)

#     df = pd.DataFrame(acc2info)
#     annotate_text = '\n'.join([sep.join([str(_) for _ in list(row)[1:]])
#                                for row in df.itertuples()])

#     out_text = template_text + '\n' + annotate_text

#     return out_text


# def to_binary_init():
#     os.chdir('/share/home-user/hyc')

#     genes = ['nirK', 'nirS', 'nosZ', 'norB']

#     df_temp = pd.DataFrame()
#     df = pd.DataFrame()

#     for gene in genes:

#         filepath = './work/6_ancestral_reconstruction/1_leaf_state/genome_' + gene + 'criteria.tsv'

#         df_temp = pd.read_table(filepath,sep = '\t')
#         df_temp.loc[df_temp[gene] == gene, [gene]] = 1
#         df_temp.loc[df_temp[gene] == str('No_' + gene), [gene]] = -1
#         if not df.empty:
#             df = pd.merge(df, df_temp, left_on='Genome', right_on='Genome')
#         else:
#             df = df_temp
#             continue

#     # for gene in genes:
#         # df.loc[df[gene] == 0, [gene]] = -1

#     out_text = to_binary(df)

#     with open('./work/tree/annotate/binary_annotate.txt', 'w') as f:
#         f.write(out_text)

#     # return df
    
# def to_binary_ref(ref_ids,
#                   sep = ',',
#                   shape = '2',
#                   field_labels = 'ref',
#                   label = 'reference'):
#     os.chdir('/share/home-user/hyc/db/itol_template')
#     template_text = open('./dataset_binary_template.txt').read()
#     template_text = template_text.format(shape=shape,
#                                          field_labels=field_labels,
#                                          dataset_label=label)
#     annotate_text = '\n'.join(['%s%s1'%(_,sep) for _ in ref_ids])
    
#     out_text = template_text + '\n' + annotate_text
#     return out_text


def get_color(acc2info,phylum=False):
    phyla2color = {p:c for p,c in zip(sorted(set(acc2info.values())),
                                      px.colors.qualitative.T10 + px.colors.qualitative.Alphabet)}
    if phylum:
        phyla2color['Proteobacteria']='#3C6A54'
        phyla2color['Alphaproteobacteria']='#FAD4A2'
        phyla2color['Betaproteobacteria']='#56CEFF'
        phyla2color['Gammaproteobacteria']='#98C995'
        phyla2color['Epsilonproteobacteria']='#52B6DC'
        phyla2color['Deltaproteobacteria']='#DAE8AA'
        phyla2color['Oligoflexia']='#FF8028'
        phyla2color['Candidatus Muproteobacteria'] = '#BD2B1C'
        phyla2color['Zetaproteobacteria'] = '#63680c'
        phyla2color['Actinobacteria']= '#E15F99'
        phyla2color['Bacteroidetes'] = '#DA16FF'
        phyla2color['Chloroflexi'] = '#511CFB'
        phyla2color['Cyanobacteria'] = '#00A08B'
        phyla2color['Euryarchaeota'] = '#FC0080'
        phyla2color['Firmicutes'] = '#B2828D'
        phyla2color['Gemmatimonadetes'] ='#6C7C32'
        phyla2color['Nitrospinae'] = '#862A16'
        phyla2color['Nitrospirae'] = '#A777F1'
        phyla2color['Planctomycetes'] = '#620042'
        phyla2color['Acidithiobacillia'] = '#336666'
        phyla2color['Crenarchaeota']= '#BAB0AC'
        phyla2color['Deinococcus-Thermus']= '#3283FE'
    
    return phyla2color


def to_external_label(acc2info,dataset='label info'):
    
    template_text = open(os.path.join(dbpath,'dataset_text_template.txt')).read()
    template_text = template_text.format(dataset_label = dataset)
    
    annotate_text = '\n'.join(['%s,%s,1,#000000,normal,1' %(k,v) 
                               for k,v in acc2info.items()])
    
    return template_text + '\n' + annotate_text


def label_text(acc2info):
    
    template_text = open(os.path.join(dbpath,'labels_template.txt')).read()
    
    annotate_text = '\n'.join(['%s,%s' %(k,v) 
                               for k,v in acc2info.items()])
    
    return template_text + '\n' + annotate_text


def to_popup(acc2info):
    
    template_text = open(os.path.join(dbpath,'popup_info_template.txt')).read()
    
    annotate_text = '\n'.join(['%s,accession,%s' %(k,v) 
                               for k,v in acc2info.items()])
    
    return template_text + '\n' + annotate_text


def colorFader(c1, c2, mix=0
               ):  # (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1 - mix) * c1 + mix * c2)


# generate_gradient_legend(100,50,0,'#ff0000','#FFFFFF','#0000ff')
def generate_gradient_legend(
    max_val, mid_val, min_val, max_c, mid_c, min_c, num_interval=7
):
    legened_v2color = {}
    if num_interval % 2 == 1:
        remained_i = (num_interval - 1) // 2
        legened_v2color[round(mid_val, 2)] = mid_c
    else:
        remained_i = num_interval // 2

    total_v = max_val - mid_val
    inter_v = int(total_v / remained_i)
    for _p in range(1, remained_i):
        des_v = _p * inter_v + mid_val
        per = _p * inter_v / total_v
        new_color = colorFader(mid_c, max_c, per)
        legened_v2color[round(des_v, 2)] = new_color
    legened_v2color[round(max_val, 2)] = max_c

    total_v = mid_val - min_val
    inter_v = int(total_v / remained_i)
    for _p in range(1, remained_i):
        des_v = _p * inter_v + min_val
        per = _p * inter_v / total_v
        new_color = colorFader(min_c, mid_c, per)
        legened_v2color[round(des_v, 2)] = new_color
    legened_v2color[round(min_val, 2)] = min_c
    return legened_v2color


def color_gradient(
    id2val, dataset_name="Completness", max_val=None, min_val=None, mid_val=50, other_params={}
):
    default_max = "#ff0000"
    default_min = "#0000ff"
    default_mid = "#FFFFFF"

    template_text = open(os.path.join(dbpath,'dataset_gradient_template.txt')).read()
    sep = get_used_sep(template_text)

    all_vals = list(set([v for k, v in id2val.items()]))

    mid_val = np.mean(all_vals) if mid_val is None else mid_val
    max_val = max(all_vals) if max_val is None else max_val
    min_val = min(all_vals) if min_val is None else min_val

    l2colors = generate_gradient_legend(
        max_val, mid_val, min_val, default_max, default_mid, default_min, num_interval=7
    )

    legend_text = f"""
LEGEND_TITLE{sep}{dataset_name}
LEGEND_SHAPES{sep}{sep.join(['1'] * 7)}
LEGEND_COLORS{sep}{sep.join([_[1] for _ in list(sorted(l2colors.items()))])}
LEGEND_LABELS{sep}{sep.join(map(str, [_[0] for _ in list(sorted(l2colors.items()))]))}"""

    annotate_text = "\n".join(
        [f"{label}{sep}{val}" for label, val in id2val.items()])

    text = template_text.format(
        dataset_label=dataset_name,
        legend_text=legend_text,
        color_min=default_min,
        color_max=default_max,
        color_mid=default_mid)
    text = replacing_params(text, other_params)

    return text + "\n" + annotate_text


def to_binary_shape(
    ID2info,
    info2style=None,
    same_color=False,
    dataset_name="Presence/Absence matrix",
    manual_v=[],
    unfilled_other=False,
    other_params={},
    no_legend=False,
    full=False,
):
    # id2info, could be {ID:list/set}
    # info2color: could be {gene1: {shape:square,color:blabla},}
    # None will use default.
    # if turn unfilled_other on, it will not draw the unfilled markers
    #
    template_text = open(os.path.join(dbpath,'dataset_binary_template.txt')).read() + "\n"
    sep = get_used_sep(template_text)
    if not manual_v:
        all_v = list(
            sorted(set([_ for v in ID2info.values() for _ in v if _])))
    else:
        all_v = manual_v

    # if coord_cols:
    #     extra_replace.update({'#SYMBOL_SPACING,10':"SYMBOL_SPACING\t-27"})
    if info2style is None:
        info2style = {k: {} for k in all_v}
    unfilled_label = "-1" if unfilled_other else "0"

    annotate_text = []
    for _ID, vset in ID2info.items():
        row = sep.join(
            [_ID] + ["1" if _ in vset else unfilled_label for _ in all_v])
        annotate_text.append(row)
    annotate_text = "\n".join(annotate_text)

    legend_text = deduced_legend2(
        info2style, all_v, sep=sep, same_colors=same_color,full=full)
    if no_legend:
        real_legend_text = ""
    else:
        real_legend_text = f"LEGEND_TITLE\t{dataset_name}\n" + legend_text.replace(
            "FIELD", "LEGEND"
        )

    template_text = replacing_params(template_text, other_params)
    template_text = template_text.format(
        legend_text=legend_text + "\n" + real_legend_text, dataset_label=dataset_name
    )

    return template_text + "\n" + annotate_text