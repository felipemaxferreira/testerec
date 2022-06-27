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
    arq='06-05-2022-03-horas-DT-ENGENHARIA-Otimizador.CSV'
    #arq='26-04-2022-03-horas-DT-ENGENHARIA-Otimizador.CSV'
    df_data=pd.read_csv(arq, sep=";", header=0, encoding='latin-1')

    df_data.rename(columns={'Ult Eqpt':'Ult_Eq',
                            'Equip Atual':'Equip_Atual',
                            'Situação Processo':'Situacao_Processo',
                            'SituaÃ§Ã£o Processo':'Situacao_Processo',
                            'Prod.':'Prod',
                            'Esp. Saída':'Esp',
                            'Esp. SaÃ­da':'Esp',
                            'Larg. Saída':'Larg',
                            'Larg. SaÃ­da':'Larg',
                            'Agrup.Ciclo':'Agrup_Ciclo',
                            'Ciclo REC5':'Ciclo',
                            'Classificação da Prioridade':'Prioridade',
                            'ClassificaÃ§Ã£o da Prioridade':'Prioridade',
                            'Dm Material (Poleg)':'Diam',
                            'Desc. Limpeza':'Limpeza',
                            'DT_PRODUCAO':'Data_Producao',
                            'Cód. Local Completo':'Pilha',
                            'CÃ³d. Local Completo':'Pilha',
                            'Obs. Volume':'Obs'
                           }, inplace=True)

    df_ciclo = pd.read_csv('Ciclos_REC5.csv', sep=';', encoding='latin-1', low_memory=False)

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

    EQUIP = ['CPC REC5', 'CPC REC234']
    SITUACAO = ['APROGRAMAR']

    df_data=df_data.query('Ciclo in @ciclos_possiveis and Situacao_Processo == @SITUACAO and Equip_Atual == @EQUIP')

    df_data = df_data[['Volume','Esp','Diam','Larg','Ciclo','Prod','Peso', 'Limpeza','Agrup_Ciclo',
                       'Equip_Atual', 'Prioridade', 'Data_Producao', 'Pilha', 'Obs']]

    df_data['REC']=df_data['Pilha'].str.slice(0,2)

    if radio_rec != "REC-2 + REC-5":
        df_data=df_data.query('REC == "R5"')

    df_data['Peso'] = df_data['Peso'].astype(str)
    df_data['Peso'] = df_data['Peso'].apply(lambda x: x.replace(',','.'))
    df_data['Peso'] = df_data['Peso'].astype(float)

    df_data['Esp'] = df_data['Esp'].astype(str)
    df_data['Esp'] = df_data['Esp'].apply(lambda x: x.replace(',','.'))
    df_data['Esp'] = df_data['Esp'].astype(float)

    df_data['Larg'] = df_data['Larg'].astype(str)
    df_data['Larg'] = df_data['Larg'].apply(lambda x: x.replace(',','.'))
    df_data['Larg'] = df_data['Larg'].astype(float)

    df_data['Diam'] = df_data['Diam'].astype(str)
    df_data['Diam'] = df_data['Diam'].apply(lambda x: x.replace(',','.'))
    df_data['Diam'] = df_data['Diam'].astype(float)

    df_data = df_data.sort_values(by=['Peso', 'Diam', 'Ciclo']).copy().reset_index(drop=True)
    df_data.Prioridade.replace(' ', 'INDEFINIDO', inplace=True)

    df_data=df_data.query('Prioridade != "INDEFINIDO"')
    df_data.drop_duplicates(subset=['Volume'], keep='first', inplace=True, ignore_index=False)
    df_data.reset_index(drop=True, inplace=True)

    Pilhas=df_data[['Pilha']]
    Pilhas=Pilhas.query('Pilha == Pilha')
    Pilhas['REC']=df_data['Pilha'].str.slice(0,2)
    Pilhas['Prefixo']=df_data['Pilha'].str.slice(0,8)
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
    df_data['Antiguidade'] = now-pd.to_datetime(df_data['Data_Producao'].astype(str), format='%Y-%m-%d %H:%M:%S')
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

    df_data = df_data[['Volume', 'Esp', 'Diam', 'Larg', 'Ciclo', 'Prod', 'Peso', 'Faixa_Fator', 'Limpeza',
                       'Equip_Atual', 'Prioridade', 'Peso_Prioridade', 'Antiguidade', 'Agrup_Ciclo',
                       'Antiguidade_Horas', 'Critico_Antiguidade', 'Pilha', 'Obs', 'REC_x', 'Pos']]
    df_data.rename(columns={'REC_x':'REC'}, inplace=True)

    df_data.Pos.fillna(0, inplace=True)
    df_data.REC.fillna('R5', inplace=True)

    pos_pilha=[0, 1]
    df_data=df_data.query('Pos in @pos_pilha')
    df_data.reset_index(drop=True, inplace=True)

    df_data['Ciclo'] = df_data['Ciclo'].astype(int)
    df_data['Ciclo'] = df_data['Ciclo'].astype(str)

    df_data['Peso'] = df_data['Peso'] / 1000
    df_data['Diam'] = df_data['Diam'] * 25.4
    df_data['Diam'] = df_data['Diam'].round(2)

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
    convectores = (len(solucao)-1) * 59.5
    altura = 4910 - convectores - sum(solucao['Larg'])
    return altura


def constraint_ciclo(solucao):
    if sum(1 for x in list(solucao['Ciclo']) if x == '134') > 1:
        return False
    else:
        ciclo = ciclos(list(solucao['Ciclo']))
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
    if '134' in list(data['Ciclo']):
        if list(data['Ciclo'])[-1] == '134':
            return True
        else:
            return False
    else:
        return True


def constraint_bi(solucao):
    lista=list(solucao['Obs'])
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
    if complemento > 1:
        return INFEASIBLE

    ### PESO
    if sum(data['Peso']) >= 84 or sum(data['Peso']) < 55:
        return INFEASIBLE

    ciclo = constraint_ciclo(data) ### CICLO
    if ciclo == False:
        return INFEASIBLE

    if constraint_fator_compressao(data) > 0: ### FATOR COMPRESSAO
        return INFEASIBLE
    #if tentativas(solucao) > 0:
    #    return INFEASIBLE

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

def get_combinations(data, n_pesado, n_medio, n_leve):
    combs=[]
    count=0

    div=data.Peso.quantile([0.33, 0.67])
    lim1=div[0.33]
    lim2=div[0.67]
    idx1=list(data.query('Peso <= @lim1').index)
    idx2=list(data.query('Peso > @lim1 and Peso < @lim2').index)
    idx3=list(data.query('Peso >= @lim2').index)
    pesado=list(combinations(idx3, r=n_pesado))
    medio=list(combinations(idx2, r=n_medio))
    leve=list(combinations(idx1, r=n_leve))

    for p in pesado:
        for m in medio:
            for l in leve:
                elemento=list(p) + list(m) + list(l)
                combs.append([count, elemento])
                count+=1

    return combs

def get_combinations_2(data, n_pesado, n_medio, n_leve, p1, p2, p3):
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

    pesado=list(combinations(idx3, r=n_pesado))
    medio=list(combinations(idx2, r=n_medio))
    leve=list(combinations(idx1, r=n_leve))

    for p in pesado:
        for m in medio:
            for l in leve:
                elemento=list(p) + list(m) + list(l)
                combs.append([count, elemento])
                count+=1

    return combs

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

    if dominio==4:
        combs=get_combinations_2(data, n_pesado=sequencia[0], n_medio=sequencia[1], n_leve=sequencia[2], p1=p1, p2=p2, p3=p3)
    else:
        combs=get_combinations_2(data, n_pesado=sequencia[0], n_medio=sequencia[1], n_leve=sequencia[2], p1=p1, p2=p2, p3=p3)

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
    fields=['Volume','Esp','Diam','Larg','Ciclo','Prod','Peso','Faixa_Fator','Agrup_Ciclo','Equip_Atual',
            'Prioridade', 'Antiguidade', 'Limpeza', 'Obs', 'Pilha', 'Pos']
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

        opcao.loc["Total"] = opcao.sum(numeric_only=True).round(2)
        opcao.fillna("", inplace=True)
        opcao.at['Total', 'Volume'] = "TOTAL"
        saida = pd.concat([saida, opcao],ignore_index=False)
        saida.Prioridade.replace('10. Antecipado ProduÃ§Ã£o', '10. Antecipado', inplace=True)
        saida.Prioridade.replace('02. CrÃ­tico', '02. Critico', inplace=True)
        saida.Prioridade.replace('05. Atraso maior que 30 dias e', '05. Atraso > 30 dias', inplace=True)
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


def get_estatisticas(df_data):
    stats=pd.DataFrame(df_data['Larg'].describe().round(2)).T
    stats.drop(columns=['std','count'], inplace=True)
    stats.columns=['Media', 'Minimo', '25%', '50%', '75%', 'Maximo']
    stats.index=['Larguras']
    stats['< 1100'] = df_data.query('Larg < 1100').shape[0]
    ciclos=pd.DataFrame(df_data['Ciclo'].value_counts()).T
    ciclos.rename(columns={'Ciclo':'Quantidade'}, inplace=True)
    return stats, ciclos


def get_analise_pesos(data):
    div=data.Peso.quantile([0.33, 0.67])
    lim1=div[0.33]
    lim2=div[0.67]
    leves=data.query('Peso <= @lim1')
    medios=data.query('Peso > @lim1 and Peso < @lim2')
    pesados=data.query('Peso >= @lim2')
    leves=pd.DataFrame(leves.Peso.describe()).T
    medios=pd.DataFrame(medios.Peso.describe()).T
    pesados=pd.DataFrame(pesados.Peso.describe()).T
    total=pd.DataFrame(data.Peso.describe()).T
    pesos=pd.concat([leves,medios,pesados,total])
    pesos.index=['Leves', 'Medios', 'Pesados', 'TOTAL']
    pesos=pesos[['count', 'mean', 'min', 'max']]
    pesos.columns=['Quantidade', 'Media', 'Minimo', 'Maximo']
    return pesos[['Quantidade', 'Minimo', 'Maximo', 'Media']]


def get_analise_prioridade(data):
    data.Prioridade=data.Prioridade.replace({'10. Antecipado ProduÃ§Ã£o':'10. Antecipado',
                                             '05. Atraso maior que 30 dias e':'05. Atraso maior 30 dias',
                                             '01. CrÃ\xadtico Parada de linha':'01. Critico Parada de linha',
                                             '02. CrÃ\xadtico':'02. Critico',
                                             '04. ExportaÃ§Ã£o':'04. Exportacao'})
    return pd.DataFrame(df_data['Prioridade'].value_counts())


def get_leves(data):
    fields=['Volume', 'Esp', 'Diam', 'Larg', 'Ciclo', 'Prod', 'Peso',
           'Limpeza', 'Agrup_Ciclo', 'Equip_Atual',
           'Prioridade',  'Pilha', 'Obs']
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


def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(sep=';').encode('utf-8')


st.set_page_config(page_title="Otimizador", page_icon="5️⃣", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.big-font {
    font-size:25px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Otimizador Recozimento 5</p>', unsafe_allow_html=True)

with st.sidebar:
    cols = st.columns((1, 1))
    radio_rec = st.selectbox("Estoque", ("REC-5", "REC-2 + REC-5"))
    #option = st.selectbox('Quantidade Rolos:', ('4-Rolos', '3-Rolos'))

    pesados= cols[0].number_input('Rolos Pesados', min_value=0, max_value=4, value=2, step=1)
    p3 = cols[1].slider('Intervalo Pesados (tons)', 20, 27, (21, 23))
    medios = cols[0].number_input('Rolos Médios', min_value=0, max_value=4, value=1, step=1)
    p2 = cols[1].slider('Intervalo Médios (tons)', 15, 20, (17, 19))
    leves = cols[0].number_input('Rolos Leves', min_value=0, max_value=4, value=1, step=1)
    p1 = cols[1].slider('Intervalo Leves (tons)', 5, 15, (10, 13))
    max_complementos = cols[0].number_input('Máximo Complementos', min_value=0, max_value=4, value=1, step=1)
    #opcoes = st.slider('Qtde Sugestoes', 0, 30, 10)

    df_data, rec2_ciclo = get_data()
    lista=get_leves(data=df_data)
    menores=list(lista.index)
    #st.dataframe(lista, 2000, 1000)
    #options = st.multiselect('What are your favorite colors', menores, [menores[0]])

#st.write('Selecionado:', option, 'para', radio_rec)

form = st.form(key="annotation", clear_on_submit=False)

with form:
    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
    with col1:
        preview = st.form_submit_button(label="Preview")
    with col2:
        submitted = st.form_submit_button(label="Executar")
    with col3:
        rolos_leves = st.form_submit_button(label="Leves")

if rolos_leves:
    df_data, rec5_ciclo = get_data()
    lista=get_leves(data=df_data)
    #menores=list(lista.index)
    st.table(lista.style.format(subset=['Larg', 'Peso', 'Esp', 'Diam'], formatter="{:.2f}"))
    #options = st.multiselect('What are your favorite colors', menores, [menores[0]])

if preview:
    cols = st.columns((1, 1))
    df_data, rec5_ciclo = get_data()
    stats, ciclos = get_estatisticas(df_data)
    st.table(stats.style.format(subset=['Media', 'Minimo', '25%', '50%', '75%', 'Maximo'], formatter="{:.2f}"))
    cols[1].dataframe(ciclos)
    pesos = get_pesos(df_data)
    st.table(pesos)#.style.format(subset=['Quantidade'], formatter="{:.0f}"))
    prio=get_analise_prioridade(df_data)
    cols[0].table(prio)

if submitted:
    # dados_retornados = funcao rodar_modelo_obter_df_de_resultado(text_param, num_param, date_param)
    INFEASIBLE = 1e8
    df_data, rec5_ciclo = get_data()
    saida=execute(df_data)

    if saida.shape[0] > 0:
        st.success(f"Cargas sugeridas ! 🤔")
        saida=saida[['Esp','Diam','Larg','Ciclo','Prod','Peso','Agrup_Ciclo','Prioridade','Limpeza','Obs','Posicao','Opcao']]
        saida.rename(columns={'Agrup_Ciclo':'Grupo'}, inplace=True)
        st.table(saida)#.style.format(subset=['Larg', 'Peso'], formatter="{:.1f}"), width=700, height=768)
        download=convert_df(df=saida)
        st.download_button(label='📥 Download',
                                data=download,
                                file_name= 'df_test.csv')
    else:
        st.error(f"❌ Sem resultados")
