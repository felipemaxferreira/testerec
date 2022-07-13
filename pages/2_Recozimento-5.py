#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import numpy as np
import pandas as pd
from itertools import product
from itertools import permutations
import itertools
import random
from math import factorial
from datetime import datetime
from itertools import combinations
import warnings
import os
warnings.filterwarnings('ignore')


INFEASIBLE = 1e8

# # CONFIGURACOES AZURE

def get_ordem_pilha(dados):
    saida=pd.DataFrame()
    for idx in dados.Prefixo.unique():
        aux=dados.query('Prefixo == @idx').sort_values(by=['Sufixo'], ascending=False)
        pos=list(range(0,len(aux)))
        aux['Pos']=pos
        saida=pd.concat([saida, aux])
    return saida


def ciclos(array):
    for cic in rec5_ciclo:
        saida=False
        if set(array).issubset(cic):
            return True
    return saida


def get_data():
    df_data = st.session_state.estoque
    #df_data.Volume = df_data.Volume.astype(str)
    #if 'filtro' not in st.session_state:
    #    st.session_state.filtro = []
    #filtrar=st.session_state.filtro
    
    #df_data=df_data.query('Volume not in @filtrar')

    df_ciclo=pd.read_csv('Ciclos_REC5.csv', sep=';',
                          encoding='latin-1', low_memory=False)

    df_ciclo.fillna(0, inplace=True)
    df_ciclo = df_ciclo.applymap(int) ## converte todas as colunas para int
    df_ciclo = df_ciclo.applymap(str) ## converte todas as colunas para string

    rec5_ciclo=[]
    for j in range(df_ciclo.shape[0]):
        cic=df_ciclo.iloc[j][0:1].values[0]
        for i in range (df_ciclo.shape[1]-1):
            if df_ciclo.iloc[j][i+1:i+2].values[0] != '0':
                rec5_ciclo.append([cic, df_ciclo.iloc[j][i+1:i+2].values[0]])

    ciclos_possiveis=[]
    for i in rec5_ciclo:
        ciclos_possiveis = ciclos_possiveis+i
    ciclos_possiveis = list(set(ciclos_possiveis))

    EQUIP = ['REC-5', 'REC-2']
    SITUACAO = ['ESTOCADO']

    df_data=df_data.query('Situacao == @SITUACAO')

    df_data = df_data[['Volume','Esp','Diam','Larg','Ciclo_Rec5','Prod','Peso', 'Limpeza','Agrup_Ciclo',
                       'Prioridade', 'Data_Producao', 'Pilha', 'Obs', 'PA', 'TT']]

    df_data['REC']=df_data['Pilha'].str.slice(0,2)

    if radio_rec != "REC-2 + REC-5":
        df_data=df_data.query('REC == "R5"')
    
    lim_inf_esp=f_esp[0]
    lim_sup_esp=f_esp[1]
    lim_inf_larg=f_larg[0]
    lim_sup_larg=f_larg[1]
    
    df_data=df_data.query('Esp >= @lim_inf_esp and Esp <= @lim_sup_esp and Larg >= @lim_inf_larg and Larg <= @lim_sup_larg')
    
    df_data = df_data.sort_values(by=['Peso', 'Diam', 'Ciclo_Rec5']).copy().reset_index(drop=True)
    df_data.Prioridade.replace(' ', 'INDEFINIDO', inplace=True)

    df_data=df_data.query('Prioridade != "INDEFINIDO"')
    df_data.drop_duplicates(subset=['Volume'], keep='first', inplace=True, ignore_index=False)
    df_data.reset_index(drop=True, inplace=True)

    Pilhas=df_data[['Pilha']]
    Pilhas=Pilhas.query('Pilha == Pilha')
    Pilhas['REC']=Pilhas['Pilha'].str.slice(0,2)
    Pilhas['Prefixo']=Pilhas['Pilha'].str.slice(0,8)
    Pilhas['Sufixo']=Pilhas['Pilha'].apply(lambda x: x[-1:])
    Pilhas.reset_index(drop=True, inplace=True)
    Pilhas.rename(columns={'Pilha':'Cod_Pilha'}, inplace=True)
    Pilhas=get_ordem_pilha(dados=Pilhas)

    df_data[['Cod_Prioridade', 'Desc_Prioridade']] = df_data['Prioridade'].str.split('.', expand=True)
    df_data.Cod_Prioridade = df_data.Cod_Prioridade.astype(str)
    df_data.Cod_Prioridade.fillna('10', inplace = True)
    df_data['Peso_Prioridade'] = df_data['Cod_Prioridade']
    mapping = {'01':-1e5, '02':-1e4, '03':1e3, '04':10, '05':2, '06':1e3, '07':-1e3, '08':1e3, '09':1e3, '10':1e3,
               'INDEFINIDO':1e3
              }
    df_data.replace({'Peso_Prioridade': mapping}, inplace=True)

    df_data['Faixa_Fator'] = 0
    for i in range(len(df_data)):
        if df_data['Esp'][i] >= 0.0 and df_data['Esp'][i] <= 0.39:
            df_data.loc[:,'Faixa_Fator'][i] = 2167
        elif df_data['Esp'][i] >= 0.4 and df_data['Esp'][i] <= 0.49:
            df_data.loc[:,'Faixa_Fator'][i] = 2610
        elif df_data['Esp'][i] >= 0.5 and df_data['Esp'][i] <= 0.59:
            df_data.loc[:,'Faixa_Fator'][i] = 3569
        elif df_data['Esp'][i] >= 0.6 and df_data['Esp'][i] <= 0.69:
            df_data.loc[:,'Faixa_Fator'][i] = 4549
        elif df_data['Esp'][i] >= 0.7 and df_data['Esp'][i] <= 0.99:
            df_data.loc[:,'Faixa_Fator'][i] = 5048
        else:
            df_data.loc[:,'Faixa_Fator'][i] = 5381

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    now = pd.to_datetime(now, format='%Y-%m-%d %H:%M:%S')
    df_data['Antiguidade'] = now-pd.to_datetime(df_data['Data_Producao'].astype(str), format='%d/%m/%Y')
    df_data['Antiguidade_Horas'] = df_data['Antiguidade'] / np.timedelta64(1, 'h')
    df_data['Prazo_Antiguidade'] = 0

    for i in range(len(df_data)):
        if df_data['Limpeza'][i] == 'EXTRA_LIMPO':
            df_data['Prazo_Antiguidade'] = 72
        else:
            df_data['Prazo_Antiguidade'] = 120

    df_data['Critico_Antiguidade']=df_data['Antiguidade_Horas'] >= df_data['Prazo_Antiguidade']
    df_data['Critico_Antiguidade']=df_data['Critico_Antiguidade'].replace({False : 0})
    df_data['Critico_Antiguidade']=df_data['Critico_Antiguidade'].replace({True : -1e5})

    df_data=df_data.merge(Pilhas, left_on=df_data.Pilha, right_on=Pilhas.Cod_Pilha, how='left')

    df_data = df_data[['Volume', 'Esp', 'Diam', 'Larg', 'Ciclo_Rec5', 'Prod', 'Peso', 'Faixa_Fator', 'Limpeza',
                       'Prioridade', 'Peso_Prioridade', 'Antiguidade', 'Agrup_Ciclo', 'Antiguidade_Horas', 
                       'Critico_Antiguidade', 'Pilha', 'REC_x', 'Pos', 'Obs', 'PA', 'TT']]
    df_data.rename(columns={'REC_x':'REC'}, inplace=True)

    df_data.Pos.fillna(0, inplace=True)
    df_data.REC.fillna('R5', inplace=True)

    #set_pilha = list(range(0, pos_pilha))
    if pos_pilha == 'TOPO':
        df_data=df_data.query('Pos == 0')
    
    if agrupamento != "TODOS":
        df_data=df_data.query('Agrup_Ciclo == @agrupamento')
        
    df_data.reset_index(drop=True, inplace=True)

    df_data['Ciclo_Rec5'] = df_data['Ciclo_Rec5'].astype(int)
    df_data['Ciclo_Rec5'] = df_data['Ciclo_Rec5'].astype(str)

    return df_data, rec5_ciclo


# # Constraints

def calcula_fator_compressao(data, dominio):

    data=data.sort_values(by=['Diam'])

    if len(data)==4:
        data['Posicao'] = [1, 2, 3, 4]
        data['Fator'] = (data.Faixa_Fator * data.Peso/data.Larg) - data.Peso - ((4 - data.Posicao) * 0.6)
        data['Acumulado'] = ([0, sum(data['Peso'][0:1].values)+0.6,
                                 sum(data['Peso'][0:2].values)+1.2,
                                 sum(data['Peso'][0:3].values)+1.8])
        data=data.sort_values(by=['Diam'])
        data['Posicao'] = ["TOPO", "Pos.3", "Pos.2", "BASE"]
    else:
        data['Posicao'] = [1, 2, 3]
        data['Fator'] = (data.Faixa_Fator * data.Peso/data.Larg) - data.Peso - ((3 - data.Posicao) * 0.6)
        data['Acumulado'] = ([0, sum(data['Peso'][0:1].values)+0.6,
                                 sum(data['Peso'][0:2].values)+1.2])
        data=data.sort_values(by=['Diam'])
        data['Posicao'] = ["TOPO", "Pos.2", "BASE"]

    data.Fator = data.Fator.round(2)
    data.Acumulado = data.Acumulado.round(2)

    return data


def constraint_diferenca_diametro_externo(data, dominio=4):
    c=list(combinations(range(0,4), 2))
    saida=[]
    for i in c:
        saida.append(x[i[0]]-x[i[1]])
    if sum(1 for i in saida if i >= 200) > 0:
        return False, saida
    else:
        return True, saida


def constraint_diferenca_diametro_abaixo(candidate, dominio=4):
    if dominio == 4:
        saida=[candidate[0]-candidate[1], candidate[1]-candidate[2], candidate[2]-candidate[3]]
        if sum(1 for i in saida if i > 100) > 0:
            return False, saida
        else:
            return True, saida
    elif dominio == 3:
        saida=[candidate[0]-candidate[1], candidate[1]-candidate[2]]
        if sum(1 for i in saida if i > 100) > 0:
            return False, saida
        else:
            return True, saida
    else:
        return False


def constraint_altura(solucao):
    convectores = (len(solucao)-1) * 60
    altura = max_altura - convectores - sum(solucao['Larg'])
    return altura


def constraint_ciclo(solucao):
    if sum(1 for x in list(solucao['Ciclo_Rec5']) if x == '134') > 1:
        return False
    else:
        ciclo = ciclos(list(solucao['Ciclo_Rec5']))
        return ciclo


def constraint_fator_compressao(solucao):
    fatores = fator_compressao(solucao, dominio=len(solucao))
    return fatores


def constraint_peso(solucao):
    peso = (95 - sum(solucao['Peso']))
    return peso


def constraint_prazo(solucao):
    prazo = sum(solucao['Peso_Prioridade'])
    return prazo


def constraint_complemento(solucao):
    complemento = (list(solucao['Larg']))
    return sum(1 for comp in complemento if comp <= 1099)


def constraint_limpeza(solucao):
    if set(list(solucao['Limpeza'])) == set(['ND', 'EXTRA_LIMPO']):
        return False
    elif set(list(solucao['Limpeza'])) == set(['ND', 'EXTRA_LIMPO', 'LIMPO_90']):
        return False
    else:
        return True


# ### verificar ciclo 134 somente na base

def constraint_posicao_134(data):
    if '134' in list(data['Ciclo_Rec5']):
        if list(data['Ciclo_Rec5'])[-1] == '134':
            return True
        else:
            return False
    else:
        return True


def constraint_bi(solucao):
    lista=list(solucao['Obs'])
    if lista.count('BI:>10mm') > 1:
        return False
    else:
        if 'BI:>10mm' in lista:
            if lista.index('BI:>10mm') == 0:
                return True
            else:
                return False
        else:
            return True


# se retorno = 0 - OK, se returno = 1 - nao factivel
def constraint_fator_compressao(data):
    data=data.sort_values(by=['Diam'])
    if len(data)==4:
        data['Posicao'] = [1, 2, 3, 4]
        data['Peso_acumulado'] = ([0, sum(data['Peso'][0:1].values)+0.6,
                                      sum(data['Peso'][0:2].values)+1.2,
                                      sum(data['Peso'][0:3].values)+1.8])
        data['Fator'] = (data.Faixa_Fator * data.Peso/data.Larg) - data.Peso - ((4 - data.Posicao) * 0.6)
    else:
        data['Posicao'] = [1, 2, 3]
        data['Peso_acumulado'] = ([0, sum(data['Peso'][0:1].values)+0.6,
                                      sum(data['Peso'][0:2].values)+1.2])
        data['Fator'] = (data.Faixa_Fator * data.Peso/data.Larg) - data.Peso - ((3 - data.Posicao) * 0.6)

    if sum(data['Fator'] < data['Peso_acumulado']) != 0: # se algum caso for True entao = 1
        return 1

    data=data.sort_values(by=['Diam'])

    if constraint_posicao_134(data) == False:
        return 1

    if not constraint_bi(data):
        return 1

    diff=list(np.diff(list(data['Diam'])).round(2))
    result=sum(1 for x in diff if x < -4.0)
    return result


# se retorno = 0 - OK, se returno = 1 - nao factivel
def constraint_fator_compressao_ampliado(data):
    if len(data)==4:
        data['Posicao'] = [1, 2, 3, 4]
        data['Peso_acumulado'] = ([0, sum(data['Peso'][0:1].values)+0.6,
                                      sum(data['Peso'][0:2].values)+1.2,
                                      sum(data['Peso'][0:3].values)+1.8])
        data['Fator'] = (data.Faixa_Fator * data.Peso/data.Larg) - data.Peso - ((4 - data.Posicao) * 0.6)
    else:
        data['Posicao'] = [1, 2, 3]
        data['Peso_acumulado'] = ([0, sum(data['Peso'][0:1].values)+0.6,
                                      sum(data['Peso'][0:2].values)+1.2])
        data['Fator'] = (data.Faixa_Fator * data.Peso/data.Larg) - data.Peso - ((3 - data.Posicao) * 0.6)

    if sum(data['Fator'] < data['Peso_acumulado']) != 0: # se algum caso for True entao = 1
        return 1

    if constraint_posicao_134(data) == False:
        return 1

    if not constraint_bi(data):
        return 1

    diff=list(np.diff(list(data['Diam'])).round(2))
    result=sum(1 for x in diff if x < -4.0)
    return result


def get_candidate(solucao):
    if len(solucao) == 4:
        return df_data[(df_data.index==solucao[0])+(df_data.index==solucao[1])+
                       (df_data.index==solucao[2])+(df_data.index==solucao[3])]
    else:
        return df_data[(df_data.index==solucao[0])+(df_data.index==solucao[1])+(df_data.index==solucao[2])]


def constraint_diferenca_larguras(data):
    larguras=list(data['Larg'])
    larguras.sort()
    return larguras[-1]-larguras[0]


def funcao_custo(solucao):
    prazo, peso, altura = 0, 0, 0

    data=get_candidate(solucao)

    altura = constraint_altura(data) ### ALTURA
    if altura < 0:
        return INFEASIBLE

    complemento = constraint_complemento(data) ### Qtde de complementos > 1
    if complemento > max_complementos:
        return INFEASIBLE

    ### PESO
    if sum(data['Peso']) >= 84 or sum(data['Peso']) < 55:
        return INFEASIBLE

    ciclo = constraint_ciclo(data) ### CICLO
    if ciclo == False:
        return INFEASIBLE

    if constraint_fator_compressao(data) > 0: ### FATOR COMPRESSAO
        return INFEASIBLE

    if constraint_limpeza(data) == False:
        return INFEASIBLE

    prazo = constraint_prazo(data) ### PRAZO
    antiguidade = sum(data['Critico_Antiguidade'])

    ### DIFERENCA LARGURAS
    if constraint_diferenca_larguras(data) > 500:
        return INFEASIBLE

    pesos=list(data['Peso'])
    qtde_pesados = sum(1 for p in pesos if p >= 20) + 1
    peso = (60 - sum(data['Peso']))*(100000/qtde_pesados)

    return altura + prazo + peso + antiguidade


# # Otimização

def get_combinations(data, n_pesado, n_medio, n_leve, p1, p2, p3):
    combs=[]
    count=0

    leve_i=p1[0]
    leve_f=p1[1]

    medio_i=p2[0]
    medio_f=p2[1]

    pesado_i=p3[0]
    pesado_f=p3[1]

    idx1=list(data.query('Peso >= @leve_i and Peso < @leve_f').index)
    idx2=list(data.query('Peso >= @medio_i and Peso < @medio_f').index)
    idx3=list(data.query('Peso >= @pesado_i and Peso <= @pesado_f').index)

    pesado=list(combinations(idx3, r=int(n_pesado)))
    medio=list(combinations(idx2, r=int(n_medio)))
    leve=list(combinations(idx1, r=int(n_leve)))

    for p in pesado:
        for m in medio:
            for l in leve:
                elemento=list(p) + list(m) + list(l)
                combs.append([count, elemento])
                count+=1

    return combs


def get_full_combs(data, rep):
    size=data.shape[0]
    limit=300
    
    if rep == 4:
        limit = 200
        
    if size > limit:
        idx=random.sample(list(data.index), 200)
        combs=list(combinations(idx, r=rep))
    else:
        combs=list(combinations(list(list(data.index)), r=rep))
    
    saida=[]
    for i in range(len(combs)):
        saida.append([i, combs[i]])
        
    return saida


def get_spread(data):
    mapa=[]
    ponto=data
    ponto=ponto[2]
    mapa.append(list(range(ponto+1, ponto+101)))
    mapa.append(list(range(ponto-100, ponto)))
    return mapa


def optimization(data, geracoes=10, sample_size=100000, best_filter=20, dominio=4, sequencia=[2,1,1]):
    saida=[]
    filtro=[]
    
    if tipo == 'Seleção parametros':
        if dominio==4:
            combs=get_combinations(data, n_pesado=sequencia[0], n_medio=sequencia[1], n_leve=sequencia[2], 
                                   p1=p1, p2=p2, p3=p3)
        else:
            combs=get_combinations(data, n_pesado=sequencia[0], n_medio=sequencia[1], n_leve=sequencia[2], 
                                   p1=p1, p2=p2, p3=p3)
    else:
        combs=get_full_combs(data, rep=n_rolos)

    sample_size=int(len(combs)*0.1)
    if sample_size > 100000:
        sample_size = 100000

    amostra=random.sample(combs, sample_size)
    
    for comb in amostra:
        custo=funcao_custo(solucao=comb[1])
        if custo < INFEASIBLE:
            saida.append([custo, comb[1], comb[0]])
    
    for epocas in range(0, geracoes):
        saida.sort()
        filtro.sort()
        melhores=saida[0:best_filter]  
        for best in melhores:
            candidate=best[1]
            if candidate not in filtro:
                filtro.append(candidate)
                spread=get_spread(data=best)
                for item in spread:
                    for i in item:
                        if i < len(combs) and i >= 0:
                            saida.append([funcao_custo(solucao=combs[i][1]), combs[i][1], combs[i][0]])
    saida.sort()
    solucao=[combinacao for combinacao in saida if combinacao[0] < INFEASIBLE]
    
    return solucao


def compare_solutions(solucao):
    x=solucao[0][1]
    saida=[]
    saida.append(x)
    for comb in solucao:
        if len(set(x) & set(comb[1])) == 0:
            saida.append(comb[1])
            x=x+comb[1]
    return saida


def show_values(indice):
    fields=['Volume','Esp','Diam','Larg','Ciclo_Rec5','Prod','Peso','Faixa_Fator','Agrup_Ciclo',
            'Prioridade', 'Antiguidade', 'Limpeza', 'Pilha', 'Pos','Obs']
    return df_data.query('index in @indice')[fields]


def saida_arquivo(options):
    count=0
    size=len(options[0])
    saida = pd.DataFrame()
    opcao = pd.DataFrame()
    for i in options:
        count+=1
        opcao = calcula_fator_compressao(data=show_values(i).sort_values(by='Diam'), dominio=size)
        opcao['Opcao'] = count
        altura=sum(opcao['Larg'])+(len(opcao)-1)*60
        opcao['Esp'] = opcao['Esp'].astype(str)
        opcao['Peso'] = opcao['Peso'].round(1)
        opcao['Diam'] = opcao['Diam'].astype(str)
        opcao['Opcao'] = opcao['Opcao'].astype(str)
        opcao['Obs'] = opcao['Obs'].astype(str)
        opcao['Pos'] = opcao['Pos'] + 1
        opcao['Pos'] = opcao['Pos'].astype(str)
        opcao['Pos'] = opcao['Pos'] + '°'        
        opcao.Obs.replace('nan', '', inplace=True)
        opcao.loc["Total"] = opcao.sum(numeric_only=True).round(2)
        opcao.fillna("", inplace=True)
        opcao.at['Total', 'Volume'] = "TOTAL"
        saida = pd.concat([saida, opcao],ignore_index=False)
        saida.Prioridade.replace('10. Antecipado Produção', '10. Antecipado Prod', inplace=True)
        saida.Prioridade.replace('05. Atraso maior que 30 dias em relação ao PCA', '05. Atraso > 30 dias', inplace=True)
    return saida


def execute(df_data):
    data = datetime.now().strftime("%d-%m-%Y")
    hora = datetime.now().strftime("%H")
    prefixo=data+"-"+hora+"-horas"

    resultado=pd.DataFrame()

    df_data=df_data.sort_values(by='Peso')
    df_data.reset_index(drop=True, inplace=True)

    sequencia=[[pesados, medios, leves]]

    if sum([pesados, medios, leves]) == 4:
        for s4 in sequencia:
            solucao=optimization(data=df_data, geracoes=10, sample_size=1000, best_filter=20, dominio=4, sequencia=s4)
            if len(solucao) > 0:
                break

        combs=[]
        if len(solucao) > 0:
            combs=compare_solutions(solucao)

        if len(combs) > 0:
            resultado = saida_arquivo(options=combs[0:10])
            resultado.set_index('Volume', inplace=True)

        if len(combs) > 0:
            apaga=resultado.index.unique()
            df_data=df_data.query('Volume not in @apaga').reset_index(drop=True)
            df_data=df_data.sort_values(by='Peso')
            df_data.reset_index(drop=True, inplace=True)

    elif sum([pesados, medios, leves]) == 3:
        for s3 in sequencia:
            solucao=optimization(data=df_data, geracoes=10, sample_size=1000, best_filter=20, dominio=3, sequencia=s3)
            if len(solucao) > 0:
                break

        combs=[]
        if len(solucao) > 0:
            combs=compare_solutions(solucao)

        if len(combs) > 0:
            resultado = saida_arquivo(options=combs[0:10])
            resultado.set_index('Volume', inplace=True)
    else:
        return resultado

    return resultado


def get_larguras(df_data):
    stats=pd.DataFrame(df_data['Larg'].describe().round(2)).T
    stats.drop(columns=['std'], inplace=True)
    stats.columns=['Quantidade', 'Media', 'Minimo', '25%', '50%', '75%', 'Maximo']
    stats.index=['Larguras']
    stats['< 1100'] = df_data.query('Larg < 1100').shape[0]
    stats.Quantidade = stats.Quantidade.astype(int)
    return stats

def get_ciclos(df_data):
    ciclos=pd.DataFrame(df_data['Agrup_Ciclo'].value_counts())
    ciclos.rename(columns={'Agrup_Ciclo':'Qtde'}, inplace=True)
    return ciclos


def get_analise_prioridade(data):
    data.Prioridade.replace('05. Atraso maior que 30 dias em relação ao PCA', '05. Atraso > 30 dias', inplace=True)
    saida=pd.DataFrame(df_data['Prioridade'].value_counts())
    return saida


def get_leves(data):
    fields=['Volume', 'Esp', 'Diam', 'Larg', 'Ciclo_Rec5', 'Prod', 'Peso', 'Limpeza', 'Agrup_Ciclo', 'Prioridade',  'Pilha']

    data.Larg = data.Larg.astype(int)
    return data.sort_values(by='Peso')[fields].head(10).set_index('Volume')

def get_pesos(data):
    aux=[]
    leves=data.query('Peso < 15')
    medios=data.query('Peso >= 15 and Peso < 20')
    pesados=data.query('Peso >= 20')

    if leves.shape[0] > 0:
        aux.append(['Leves', len(leves), min(leves['Peso']), max(leves['Peso']), np.mean(leves['Peso'])])
    else:
        aux.append(['Leves', 0, 0, 0, 0])
    if medios.shape[0] > 0:
        aux.append(['Medios', len(medios), min(medios['Peso']), max(medios['Peso']), np.mean(medios['Peso'])])
    else:
        aux.append(['Medios', 0, 0, 0, 0])
    if pesados.shape[0] > 0:
        aux.append(['Pesados', len(pesados), min(pesados['Peso']), max(pesados['Peso']), np.mean(pesados['Peso'])])
    else:
        aux.append(['Pesados', 0, 0, 0, 0])

    return pd.DataFrame(aux, columns=["Peso", "Qtde", "Minimo", "Maximo", "Media"]).set_index('Peso')


def get_pilha(data):
    Pilhas=data[['Pilha']]
    Pilhas=Pilhas.query('Pilha == Pilha')
    Pilhas['REC']=Pilhas['Pilha'].str.slice(0,2)
    Pilhas['Prefixo']=Pilhas['Pilha'].str.slice(0,8)
    Pilhas['Sufixo']=Pilhas['Pilha'].apply(lambda x: x[-1:])
    Pilhas.reset_index(drop=True, inplace=True)
    Pilhas.rename(columns={'Pilha':'Cod_Pilha'}, inplace=True)
    Pilhas=get_ordem_pilha(dados=Pilhas)
    saida=[]
    for i in Pilhas.query('REC == "R2"').Pos.unique():
        posicao = str(i+1)+"ª Posicão"
        qtde = len(Pilhas.query('Pos == @i'))
        if qtde > 0:
            saida.append([posicao, qtde])
    
    saida=pd.DataFrame(saida, columns=['Posicao','Qtde']).set_index('Posicao')
    saida.index.name = 'Pilhas'
    return saida

def get_stats_larguras(data):    
    largura=[]
    largura.append([data.shape[0]])
    largura.append([min(data.Larg)])
    largura.append([max(data.Larg)])
    largura.append([len(data.query('Larg < 1100'))])
    largura.append([len(data.query('Larg >= 1300'))])
    return pd.DataFrame(largura, index=['Total Rolos', 'Largura Minima', 'Largura Maxima', 
                                        'Qtde Largs < 1100', 'Qtde Largs >= 1300'], columns=['Valores'])

def get_pa(data):
    saida = pd.DataFrame(data.PA.value_counts())
    saida.rename(columns={'PA':'PA: Qtde Rolos'}, inplace=True)
    return saida

def get_tt(data):
    saida = pd.DataFrame(data.query('Limpeza == "EXTRA_LIMPO"').TT.value_counts())
    saida.rename(columns={'TT':'TT: EXTRA LIMPO'}, inplace=True)
    return saida

def convert_df(df):
    return df.to_csv(sep=';').encode('utf-8')


st.set_page_config(page_title="Otimizador", page_icon="5️⃣", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.big-font {
    font-size:16px !important;
}
</style>
""", unsafe_allow_html=True)

#st.markdown('<p class="big-font">Otimizador Recozimento 5</p>', unsafe_allow_html=True)
        
with st.sidebar:
    cols = st.columns((1, 1))
    tipo = st.radio("Tipo de rodada", ('Seleção parametros', 'Sem restrições - REC5', 'Sem restrições - REC5 + REC2'))
    
    preview = st.button(label="♻️ Preview Estoque")
    submitted = st.button(label="☠️ Executar Cargas")
    
    #if 'filtro' in st.session_state:
    #    if st.button('Salvar Rodada'):
    #        st.write(st.session_state.filtro)
    #    if st.button('Limpar Rodada'):
    #        st.session_state.filtro = []
    
with st.expander("Seleção de Pesos", expanded=False):
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            pesados = st.number_input('Rolos Pesados', min_value=0, max_value=4, value=2, step=1)
            medios = st.number_input('Rolos Médios', min_value=0, max_value=4, value=1, step=1)
            leves = st.number_input('Rolos Leves', min_value=0, max_value=4, value=1, step=1)            
            
        with col2:
            p3 = st.slider('Intervalo Pesados (tons)', 20, 27, (20, 27))
            p2 = st.slider('Intervalo Médios (tons)', 15, 20, (15, 20))
            p1 = st.slider('Intervalo Leves (tons)', 5, 15, (5, 15))     
            
with st.expander("Parametros Recozimento-5", expanded=True): 
    #filtros = st.form(key="Recozimento-5", clear_on_submit=False)
    #with filtros:
    with st.container():
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            radio_rec = st.selectbox("Estoque", ("REC-5", "REC-2 + REC-5")) 
        with col2:
            agrupamento = st.selectbox('Grupo:', ('TODOS', 'EM', 'QC', 'IF/EEP-CC', 'EEP', 'EP'))
            f_larg = st.slider('LARGURA', 500, 1700, (600, 1600)) 
        with col3:
            max_complementos = st.number_input('Máximo Complementos', min_value=0, max_value=4, value=1, step=1)
        with col4:
            pos_pilha = st.selectbox('Posicão Pilha:', ('TODOS', 'TOPO'))
            f_esp = st.slider('ESPESSURA', 0.15, 5.0, (0.3, 5.0)) 
        with col5:
            max_altura = st.number_input('Altura Máxima', min_value=0, max_value=5010, value=4910, step=1)              
        

if preview:
    df_data, rec5_ciclo = get_data()
    ciclos = get_ciclos(df_data)
    pesos = get_pesos(df_data)
    prio = get_analise_prioridade(df_data)
    sts_pilhas = get_pilha(df_data)
    pa = get_pa(df_data)
    tt = get_tt(df_data)
    stats_largs = get_stats_larguras(df_data)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.table(stats_largs)
        st.table(ciclos)
        if sts_pilhas.shape[0] > 0:
            st.table(sts_pilhas)        
    with col2:
        st.table(pesos.style.format(subset=['Minimo', 'Maximo', 'Media'], formatter="{:.3f}"))
        st.table(tt)
    with col3:
        st.table(pa)
        st.table(prio)

if submitted:
    INFEASIBLE = 1e8
    if tipo != 'Seleção parametros':
        pos_pilha = 'TODOS'
        agrupamento = 'TODOS'
        f_larg = (600, 1600)
        f_esp = (0.3, 5.0)
        max_complementos = 4
        max_altura = 4910
        if tipo == 'Sem restrições - REC5 + REC2':
            radio_rec = "REC-2 + REC-5"
        else:
            radio_rec = "REC-5"
        
        df_data, rec5_ciclo = get_data()
        
        n_rolos=4
        saida4=execute(df_data)
        
        if saida4.shape[0] > 0:
            fora=list(saida4.index)
            df_data=df_data.query('Volume not in @fora')
            df_data.reset_index(drop=True, inplace=True)
        
        n_rolos=3
        saida3=execute(df_data)
        
        saida=pd.concat([saida4, saida3])
    else:
        df_data, rec5_ciclo = get_data()
        saida=execute(df_data)
    
    if saida.shape[0] > 0:
        st.success(f"Cargas sugeridas ! 🤔")
        saida.rename(columns={'Agrup_Ciclo':'Grupo', 'Ciclo_Rec5':'Ciclo', 'Opcao':'SEQ'}, inplace=True)
        saida=saida[['Esp','Diam','Larg','Ciclo','Prod','Peso','Grupo','Prioridade','Limpeza','Pilha','Pos','Obs','SEQ']]
        saida.Larg = saida.Larg.astype(int)
        st.table(saida.style.format(subset=['Peso'], formatter="{:.2f}"))
        download=convert_df(df=saida)
        st.download_button(label='📥 Baixar Rodada', data=download, file_name='df_test.csv')
        st.session_state.filtro = list(saida.query('index != "TOTAL"').index)
    else:
        st.error(f"❌ Sem resultados")
        
        