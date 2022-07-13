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
import os

def get_data():
    df_data = st.session_state.estoque

    LIMITE_LARG = 4810 - 180
    #INFEASIBLE = 1e8
    possiveis_ciclos = ['R', 'N', 'W', 'I', 'Z', 'E', 'D', 'A', 'Y', 'X']
    EQUIP = ['REC-2', 'REC-5']

    df_data=df_data.query('Ciclo_Rec2 == Ciclo_Rec2 and Situacao == "ESTOCADO"')

    df_ciclo = pd.read_csv('CICLO_LETRAS.csv', sep=';',
                            encoding='latin-1', low_memory=False)
    df_ciclo.drop(columns=['CICLO'], inplace=True)

    df_data['REC']=df_data['Pilha'].str.slice(0,2)
    df_data=df_data.query('REC == "R2"')

    # # Tratamento dos dados

    # INICIO ###############

    rec2_ciclo=get_list_ciclo(df_ciclo)

    df_data['Ciclo_Rec2'] = df_data['Ciclo_Rec2'].str.strip()
    
    lim_inf_esp=f_esp[0]
    lim_sup_esp=f_esp[1]
    lim_inf_larg=f_larg[0]
    lim_sup_larg=f_larg[1]
    
    df_data=df_data.query('Esp >= @lim_inf_esp and Esp <= @lim_sup_esp and Larg >= @lim_inf_larg and Larg <= @lim_sup_larg')
    
    df_data = df_data.sort_values(by=['Esp', 'Peso', 'Diam', 'Ciclo_Rec2']).copy().reset_index(drop=True)

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
    df_data['Antiguidade'] = now-pd.to_datetime(df_data['Data_Producao'].astype(str), format='%d/%m/%Y')
    df_data['Antiguidade_Horas'] = df_data['Antiguidade'] / np.timedelta64(1, 'h')
    df_data['Prazo_Antiguidade'] = 0

    for i in range(len(df_data)):
        if df_data['Limpeza'][i] == 'EXTRA_LIMPO':
            df_data['Prazo_Antiguidade'] = 72
        else:
            df_data['Prazo_Antiguidade'] = 120

    df_data['Critico_Antiguidade'] = df_data['Antiguidade_Horas'] >= df_data['Prazo_Antiguidade']


    df_data = df_data[['Volume','Esp','Diam','Larg','Ciclo_Rec2','Prod','Peso','Faixa_Fator', 'Limpeza',
                       'Prazo_Antiguidade', 'Agrup_Ciclo', 'Prioridade', 'Peso_Prioridade', 'Antiguidade', 
                       'Antiguidade_Horas', 'Critico_Antiguidade', 'Pilha', 'Obs', 'PA', 'TT']]

    df_data.reset_index(drop=True, inplace=True)

    df_data['Critico_Antiguidade']=df_data['Critico_Antiguidade'].replace({False : 0})
    df_data['Critico_Antiguidade']=df_data['Critico_Antiguidade'].replace({True : -1e5})

    df_data=df_data.merge(Pilhas, left_on=df_data.Pilha, right_on=Pilhas.Cod_Pilha, how='left')

    df_data = df_data[['Volume', 'Esp', 'Diam', 'Larg', 'Ciclo_Rec2', 'Prod', 'Peso', 'Faixa_Fator', 'Limpeza',
                       'Prioridade', 'Peso_Prioridade', 'Antiguidade', 'Agrup_Ciclo', 'Antiguidade_Horas', 
                       'Critico_Antiguidade', 'Pilha', 'REC', 'Pos', 'Obs', 'PA', 'TT']]

    df_data.Pos.fillna(0, inplace=True)
    df_data.REC.fillna('R2', inplace=True)
    
    #set_pilha = list(range(0, pos_pilha))
    if pos_pilha == 'TOPO':
        df_data=df_data.query('Pos == 0')
    
    if agrupamento != "TODOS":
        df_data=df_data.query('Agrup_Ciclo == @agrupamento')    

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
    convectores = (len(solucao)-1) * 60
    altura = max_altura - convectores - sum(solucao['Larg'])
    return altura


def constraint_ciclo(solucao):
    ciclo = ciclos(list(solucao['Ciclo_Rec2']))
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
        saida.append([i, list(combs[i])])
        
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


# # Otimização
def show_values(indice):
    fields=['Volume','Esp','Diam','Larg','Ciclo_Rec2','Prod','Peso','Faixa_Fator','Agrup_Ciclo',
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
    return pd.DataFrame(df_data['Prioridade'].value_counts())

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
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(sep=';').encode('utf-8')


st.set_page_config(page_title="Otimizador", page_icon="2️⃣", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.big-font {
    font-size:16px !important;
}
</style>
""", unsafe_allow_html=True)

#st.markdown('<p class="big-font">Otimizador Recozimento 2</p>', unsafe_allow_html=True)

with st.sidebar:
    cols = st.columns((1, 1))
    
    tipo = st.radio("Tipo de rodada", ('Seleção parametros', 'Sem restrições'))
    
    preview     = st.button(label="♻️ - Preview Estoque")
    submitted   = st.button(label="☠️ - Executar Cargas") 

with st.expander("Seleção de Pesos", expanded=False):
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            #pos_pilha = st.selectbox('Posicão Pilha:', ('TODOS', 'TOPO'))
            pesados = st.number_input('Rolos Pesados', min_value=0, max_value=4, value=2, step=1)
            medios = st.number_input('Rolos Médios', min_value=0, max_value=4, value=1, step=1)
            leves = st.number_input('Rolos Leves', min_value=0, max_value=4, value=1, step=1)            
            
        with col2:
            #agrupamento = st.selectbox('Grupo:', ('TODOS', 'EM', 'QC', 'IF/EEP-CC', 'EEP', 'EP'))
            p3 = st.slider('Intervalo Pesados (tons)', 20, 27, (20, 27))
            p2 = st.slider('Intervalo Médios (tons)', 15, 20, (15, 20))
            p1 = st.slider('Intervalo Leves (tons)', 5, 15, (5, 15))     
    
with st.expander("Parametros Recozimento-2", expanded=True):
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            max_complementos = st.number_input('Máximo Complementos', min_value=0, max_value=4, value=1, step=1)    
            f_esp = st.slider('Espessura', 0.15, 5.0, (0.3, 5.0))
        with col2:
            pos_pilha = st.selectbox('Posicão Pilha:', ('TODOS', 'TOPO'))
        with col3:
            max_altura = st.number_input('Altura Máxima', min_value=0, max_value=5010, value=4810, step=1)    
            f_larg = st.slider('Largura', 500, 1700, (600, 1600))
            #max_sugestoes = st.number_input('Qtde Sugestões', min_value=0, max_value=20, value=10, step=1)
        with col4:
            agrupamento = st.selectbox('Grupo:', ('TODOS', 'EM', 'QC', 'IF/EEP-CC', 'EEP', 'EP'))           
            

if preview:
    df_data, rec2_ciclo = get_data()
    ciclos = get_ciclos(df_data)
    pesos = get_pesos(df_data)
    prio = get_analise_prioridade(df_data)
    sts_pilhas = get_pilha(df_data)
    pa = get_pa(df_data)
    stats_largs = get_stats_larguras(df_data)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.table(stats_largs)
        st.table(ciclos)
        if sts_pilhas.shape[0] > 0:
            st.table(sts_pilhas)        
    with col2:
        st.table(pesos.style.format(subset=['Minimo', 'Maximo', 'Media'], formatter="{:.3f}"))
        st.table(prio)
    with col3:
        st.table(pa)

if submitted:
    INFEASIBLE = 1e8
    st.write(tipo)
    if tipo != 'Seleção parametros':
        pos_pilha = 'TODOS'
        agrupamento = 'TODOS'
        f_larg = (600, 1600)
        f_esp = (0.3, 5.0)
        max_complementos = 4
        max_altura = 4810
        
        df_data, rec2_ciclo = get_data()
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
        df_data, rec2_ciclo = get_data()
        saida=execute(df_data)

    if saida.shape[0] > 0:
        st.success(f"Cargas sugeridas ! 🤔")
        saida=saida[['Esp','Diam','Larg','Ciclo_Rec2','Prod','Peso','Agrup_Ciclo','Prioridade','Limpeza',
                     'Posicao', 'Pilha', 'Pos', 'Obs', 'Opcao']]
        saida.rename(columns={'Agrup_Ciclo':'Grupo', 'Ciclo_Rec2':'Ciclo', 'Opcao':'SEQ'}, inplace=True)
        saida.Larg = saida.Larg.astype(int)
        st.table(saida.style.format(subset=['Peso'], formatter="{:.2f}"))
        download=convert_df(df=saida)
        st.download_button(label='📥 Download', data=download, file_name= 'df_test.csv')
    else:
        st.error(f"❌ Sem resultados")
