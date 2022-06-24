#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import streamlit as st
import time
import random
import itertools
import warnings
warnings.filterwarnings('ignore')
from itertools import combinations
from itertools import product
from math import factorial
from datetime import datetime
from random import sample


def get_data():
    arq='DT-ENGENHARIA-Otimizador.CSV'
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
                            'Ciclo REC2':'Ciclo',
                            'Classificação da Prioridade':'Prioridade',
                            'ClassificaÃ§Ã£o da Prioridade':'Prioridade',
                            'Dm Material (Poleg)':'Diam',
                            'Desc. Limpeza':'Limpeza',
                            'DT_PRODUCAO':'Data_Producao',
                            'Cód. Local Completo':'Pilha',
                            'CÃ³d. Local Completo':'Pilha',
                            'Obs. Volume':'Obs'
                           }, inplace=True)


    LIMITE_LARG = 4810 - 180
    #INFEASIBLE = 1e8
    possiveis_ciclos = ['R', 'N', 'W', 'I', 'Z', 'E', 'D', 'A', 'Y', 'X']
    EQUIP = ['CPC REC234', 'CPC REC5']
    SITUACAO = ['APROGRAMAR']

    df_data=df_data.query('Situacao_Processo == @SITUACAO and Equip_Atual == @EQUIP and Ciclo in @possiveis_ciclos')


    df_ciclo = pd.read_csv('CICLO_LETRAS.csv', sep=';',
                           encoding='latin-1', low_memory=False)
    df_ciclo.drop(columns=['CICLO'], inplace=True)


    # # Tratamento dos dados

    # INICIO ###############


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

    rec2_ciclo=get_list_ciclo(df_ciclo)


    df_data['Ciclo'] = df_data['Ciclo'].str.strip()


    df_data = df_data.sort_values(by=['Esp', 'Peso', 'Diam', 'Ciclo']).copy().reset_index(drop=True)


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
    mapping = {'01':-1e5,'02':-1e4,'03':1e4,'04':10,'05':2,'06':1e4,'07':-1e3,'08':1e4,
               '09':1e4,'10':1e4,'INDEFINIDO':1e4}
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

    df_data['Critico_Antiguidade'] = df_data['Antiguidade_Horas'] >= df_data['Prazo_Antiguidade']


    df_data=df_data.merge(Pilhas, left_on=df_data.Pilha, right_on=Pilhas.Cod_Pilha, how='left')


    df_data = df_data[['Volume','Esp','Diam','Larg','Ciclo','Prod','Peso','Faixa_Fator', 'Limpeza',
                       'Prazo_Antiguidade',
                       'Agrup_Ciclo','Equip_Atual', 'Prioridade', 'Peso_Prioridade', 'Antiguidade',
                       'Antiguidade_Horas',
                       'Critico_Antiguidade', 'Pilha', 'Obs', 'Pos', 'REC']]


    df_data.Pos.fillna(0, inplace=True)
    df_data.REC.fillna('R2', inplace=True)


    pos_pilha=[0]
    df_data=df_data.query('Pos in @pos_pilha and REC == "R2"')
    df_data.reset_index(drop=True, inplace=True)


    df_data['Critico_Antiguidade']=df_data['Critico_Antiguidade'].replace({False : 0})
    df_data['Critico_Antiguidade']=df_data['Critico_Antiguidade'].replace({True : -1e5})


    df_data.Pilha = df_data.Pilha.astype(str)


    df_data['Peso'] = df_data['Peso'] / 1000
    df_data['Diam'] = df_data['Diam'] * 25.4
    df_data['Diam'] = df_data['Diam'].round(2)


    # ## Constraints

    data = datetime.now().strftime("%d-%m-%Y")
    hora = datetime.now().strftime("%H")
    prefixo=data+"-"+hora+"-horas"


    # ### Dominio = 4 rolos

    df_data=df_data.sort_values(by='Peso')
    df_data.reset_index(drop=True, inplace=True)

    return df_data, rec2_ciclo


def get_list_ciclo(data):
    ciclo=[]
    for i in range(len(data)):
        x = data[i:i+1].values[0]
        ciclo.append(list(x.astype(str)))
    return ciclo


def get_ordem_pilha(dados):
    saida=pd.DataFrame()
    for idx in dados.Prefixo.unique():
        aux=dados.query('Prefixo == @idx').sort_values(by=['Sufixo'], ascending=False)
        pos=list(range(0,len(aux)))
        aux['Pos']=pos
        saida=pd.concat([saida, aux])
    return saida


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


def constraint_diferenca_diametro_abaixo(candidate, dominio=4):
    if dominio == 4:
        saida=[candidate[0]-candidate[1], candidate[1]-candidate[2], candidate[2]-candidate[3]]
        if sum(1 for i in saida if i > 4) > 0:
            return False, saida
        else:
            return True, saida
    elif dominio == 3:
        saida=[candidate[0]-candidate[1], candidate[1]-candidate[2]]
        if sum(1 for i in saida if i > 4) > 0:
            return False, saida
        else:
            return True, saida
    else:
        return False


def ciclos(array):
    for cic in rec2_ciclo:
        saida=False
        if set(array).issubset(cic):
            return True
    return saida


def constraint_altura(solucao):
    convectores = (len(solucao)-1) * 59.5
    altura = 4810 - convectores - sum(solucao['Larg'])
    return altura


def constraint_ciclo(solucao):
    ciclo = ciclos(list(solucao['Ciclo']))
    return ciclo


def constraint_fator_compressao(solucao):
    fatores = fator_compressao(solucao, dominio=len(solucao))
    return fatores


def constraint_peso(solucao):
    peso = sum(solucao['Peso'])
    return peso


def constraint_prazo(solucao):
    prazo = sum(solucao['Peso_Prioridade'])
    return prazo


def constraint_complemento(solucao):
    complemento = (list(solucao['Larg']))
    return sum(1 for comp in complemento if comp <= 1099)


def constraint_pilha(solucao):
    pos = sum(solucao['Pos_Pilha'])
    return pos


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

    if not constraint_bi(data):
        return 1

    data=data.sort_values(by=['Diam'])

    diff=list(np.diff(list(data['Diam'])).round(2))
    result=sum(1 for x in diff if x < -4.0)
    return result


def get_candidate(solucao):
    if len(solucao) == 4:
        return df_data[(df_data.index==solucao[0])+(df_data.index==solucao[1])+
                       (df_data.index==solucao[2])+(df_data.index==solucao[3])]
    else:
        return df_data[(df_data.index==solucao[0])+(df_data.index==solucao[1])+(df_data.index==solucao[2])]


def constraint_enobrecimento(data):
    return len(data['Agrup_Ciclo'].unique())


def constraint_pesados(data):
    pesos=list(data['Peso'])
    if sum(1 for p in pesos if p >= 20) > 1:
        return 1
    else:
        return 0


def constraint_diferenca_larguras(data):
    larguras=list(data['Larg'])
    larguras.sort()
    return larguras[-1]-larguras[0]


# ## Função de custo

def funcao_custo(solucao):
    antiguidade, prazo, peso, altura = 0, 0, 0, 0
    data=df_data.loc[solucao]

    ### ALTURA
    altura = constraint_altura(data)
    if altura < 0:
        return INFEASIBLE

    ### complemento
    complemento = constraint_complemento(data)
    if complemento > max_complementos:
        return INFEASIBLE

    ### CICLO
    ciclo = constraint_ciclo(data)
    if ciclo == False:
        return INFEASIBLE

    ### ENOBRECIMENTO
    if constraint_enobrecimento(data) > 1:
        return INFEASIBLE

    ### DIFERENCA LARGURAS
    if constraint_diferenca_larguras(data) > 500:
        return INFEASIBLE

    ### FATOR COMPRESSAO
    if constraint_fator_compressao(data) > 0:
        return INFEASIBLE

    pesos=60-sum(data['Peso'])

    posicao=sum(list(np.array(data['Pos'])**2))

    ### PRAZO
    prazo = constraint_prazo(data)
    antiguidade = sum(data['Critico_Antiguidade'])

    return (altura*100)+prazo+peso*1000+antiguidade+posicao*1000000


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


# # Otimização
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
        for seq in sequencia:
            solucao=optimization(data=df_data, geracoes=10, sample_size=1000, best_filter=20, dominio=4, sequencia=seq)
            if len(solucao) > 0:
                break


        combs=[]
        if len(solucao) > 0:
            combs=compare_solutions(solucao)

        if len(combs) > 0:
            resultado = saida_arquivo(options=combs[0:10])
            resultado.set_index('Volume', inplace=True)

    elif sum([pesados, medios, leves]) == 3:
        for seq in sequencia:
            solucao=optimization(data=df_data, geracoes=10, sample_size=1000, best_filter=20, dominio=3, sequencia=seq)
            if len(solucao) > 0:
                break
    else:
        return resultado

    combs=[]
    if len(solucao) > 0:
        combs=compare_solutions(solucao)

    if len(combs) > 0:
        resultado = saida_arquivo(options=combs[0:10])
        resultado.set_index('Volume', inplace=True)

    return resultado


def get_estatisticas(df_data):
    stats=pd.DataFrame(df_data['Larg'].describe().round(2)).T
    stats.drop(columns=['std','count'], inplace=True)
    stats.columns=['Media', 'Minimo', '25%', '50%', '75%', 'Maximo']
    stats.index=['Larguras']
    stats['< 1100'] = df_data.query('Larg < 1100').shape[0]
    ciclos=pd.DataFrame(df_data['Agrup_Ciclo'].value_counts())
    ciclos.rename(columns={'Agrup_Ciclo':'Quantidade'}, inplace=True)
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
    leves=data.query('Peso < 15')
    medios=data.query('Peso >= 15 and Peso < 20')
    pesados=data.query('Peso >= 20')
    aux=[['Leves', len(leves), min(leves['Peso']), max(leves['Peso']), np.mean(leves['Peso'])],
         ['Medios', len(medios), min(medios['Peso']), max(medios['Peso']), np.mean(medios['Peso'])],
         ['Pesados', len(pesados), min(pesados['Peso']), max(pesados['Peso']), np.mean(pesados['Peso'])]]
    return pd.DataFrame(aux, columns=["Peso", "Qtde", "Minimo", "Maximo", "Media"]).set_index('Peso')

def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(sep=';').encode('utf-8')


st.set_page_config(page_title="Otimizador", page_icon="🐞", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.big-font {
    font-size:25px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Otimizador Recozimento 2</p>', unsafe_allow_html=True)

with st.sidebar:
    cols = st.columns((1, 1))

    pesados= cols[0].number_input('Rolos Pesados', min_value=0, max_value=4, value=2, step=1)
    p3 = cols[1].slider('Intervalo Pesados (tons)', 20, 27, (21, 23))
    medios = cols[0].number_input('Rolos Médios', min_value=0, max_value=4, value=1, step=1)
    p2 = cols[1].slider('Intervalo Médios (tons)', 15, 20, (17, 19))
    leves = cols[0].number_input('Rolos Leves', min_value=0, max_value=4, value=1, step=1)
    p1 = cols[1].slider('Intervalo Leves (tons)', 5, 15, (10, 13))
    max_complementos = cols[0].number_input('Máximo Complementos', min_value=0, max_value=4, value=1, step=1)

    df_data, rec2_ciclo = get_data()
    lista=get_leves(data=df_data)
    menores=list(lista.index)

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
    df_data, rec2_ciclo = get_data()
    lista=get_leves(data=df_data)
    #menores=list(lista.index)
    st.dataframe(lista, 2000, 1000)
    #options = st.multiselect('What are your favorite colors', menores, [menores[0]])

if preview:
    cols = st.columns((1, 1))
    df_data, rec2_ciclo = get_data()
    stats, ciclos = get_estatisticas(df_data)
    st.table(stats.style.format(subset=['Media', 'Minimo', '25%', '50%', '75%', 'Maximo'], formatter="{:.2f}"))
    cols[1].table(ciclos)
    pesos = get_pesos(df_data)
    st.table(pesos)#.style.format(subset=['Quantidade'], formatter="{:.0f}"))
    prio=get_analise_prioridade(df_data)
    cols[0].table(prio)

if submitted:
    INFEASIBLE = 1e8
    df_data, rec2_ciclo = get_data()
    saida=execute(df_data)

    if saida.shape[0] > 0:
        st.success(f"Cargas sugeridas ! 🤔")
        saida=saida[['Esp','Diam','Larg','Ciclo','Prod','Peso','Agrup_Ciclo','Prioridade','Limpeza','Obs','Posicao','Opcao']]
        saida.rename(columns={'Agrup_Ciclo':'Grupo'}, inplace=True)
        st.table(saida)#.style.format(subset=['Larg', 'Peso'], formatter="{:.1f}"))
        download=convert_df(df=saida)
        st.download_button(label='📥 Download',
                                data=download,
                                file_name= 'df_test.csv')
    else:
        st.error(f"❌ Sem resultados")
