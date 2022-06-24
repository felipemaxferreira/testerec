#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime


import warnings
warnings.filterwarnings('ignore')

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

    EQUIP = ['CPC REC5', 'CPC REC234']
    SITUACAO = ['APROGRAMAR']

    df_data=df_data.query('Ciclo == Ciclo and Situacao_Processo == @SITUACAO and Equip_Atual == @EQUIP')

    df_data = df_data[['Volume','Esp','Diam','Larg','Ciclo','Prod','Peso', 'Limpeza','Agrup_Ciclo',
                       'Equip_Atual', 'Prioridade', 'Data_Producao', 'Pilha', 'Obs', 'Ciclo REC2']]

    df_data['REC']=df_data['Pilha'].str.slice(0,2)
    #df_data=df_data.query('REC == "R5"')

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


    df_data=df_data.merge(Pilhas, left_on=df_data.Pilha, right_on=Pilhas.Cod_Pilha, how='left')

    df_data = df_data[['Volume', 'Esp', 'Diam', 'Larg', 'Ciclo', 'Prod', 'Peso', 'Limpeza',
                       'Equip_Atual', 'Prioridade', 'Data_Producao', 'Agrup_Ciclo',
                       'Pilha', 'Obs', 'REC_x', 'Pos', 'Ciclo REC2']]
    df_data.rename(columns={'REC_x':'REC'}, inplace=True)

    df_data.reset_index(drop=True, inplace=True)

    df_data['Peso'] = df_data['Peso'] / 1000
    df_data['Diam'] = df_data['Diam'] * 25.4
    df_data['Diam'] = df_data['Diam'].round(2)

    df_data['Prioridade']=df_data['Prioridade'].replace(
                                                {'10. Antecipado Produção':'10. Antecipado',
                                                 '10. Antecipado ProduÃ§Ã£o':'10. Antecipado',
                                                 '05. Atraso maior que 30 dias e':'05. Atraso > 30 dias',
                                                 '01. Crítico Parada de linha':'01. Parada de linha',
                                                 '02. Crítico':'02. Critico',
                                                 '04. Exportação':'04. Exportacao',
                                                 '01. CrÃ\xadtico Parada de linha':'01. Parada de linha',
                                                 '02. CrÃ\xadtico':'02. Critico',
                                                 '04. ExportaÃ§Ã£o':'04. Exportacao'})

    return df_data


def get_larguras(df_data):
    stats=pd.DataFrame(df_data['Larg'].describe().round(2)).T
    stats.drop(columns=['std','count'], inplace=True)
    stats.columns=['Media', 'Minimo', '25%', '50%', '75%', 'Maximo']
    stats.index=['Larguras']
    stats['< 1100'] = df_data.query('Larg < 1100').shape[0]
    return stats


def get_estoque_ciclos(df_data):
    ciclos=pd.DataFrame(df_data['Agrup_Ciclo'].value_counts())
    ciclos.rename(columns={'Agrup_Ciclo':'Quantidade'}, inplace=True)
    return ciclos


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
                                             '05. Atraso maior que 30 dias e':'05. Atraso > 30 dias',
                                             '01. CrÃ\xadtico Parada de linha':'01. Parada de linha',
                                             '02. CrÃ\xadtico':'02. Critico',
                                             '04. ExportaÃ§Ã£o':'04. Exportacao'})
    return pd.DataFrame(df_data['Prioridade'].value_counts())


def get_leves(data):
    fields=['Volume', 'Esp', 'Diam', 'Larg', 'Ciclo', 'Prod', 'Peso',
           'Limpeza', 'Agrup_Ciclo', 'Equip_Atual', 'Prioridade',  'Pilha', 'Obs']
    return data.sort_values(by='Peso')[fields].head(10).set_index('Volume')


#df_data.set_index('Volume')[['Esp', 'Diam', 'Larg', 'Ciclo', 'Ciclo REC2', 'Prod', 'Peso', 'Limpeza', 'Equip_Atual',
#                             'Prioridade', 'Agrup_Ciclo', 'Pilha', 'Obs', 'Data_Producao']]

st.set_page_config(page_title="Estoque", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

#st.write("# Welcome to Streamlit! 👋")

st.sidebar.success("Select a demo above.")

df_data = get_data()
#st.bar_chart(df_data.set_index('Volume')['Peso'])
st.table(df_data.set_index('Volume')[['Esp', 'Diam', 'Larg', 'Ciclo', 'Ciclo REC2', 'Prod', 'Peso', 'Limpeza', 'Equip_Atual',
                             'Prioridade', 'Agrup_Ciclo', 'Pilha', 'Obs', 'Data_Producao']])
