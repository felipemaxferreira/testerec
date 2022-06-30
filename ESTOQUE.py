#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def eda(data):
    data.rename(columns={'Dt Prod':'Data_Producao',
                         'Nr Rolo':'Volume',
                         'Situação':'Situacao',
                         'Cic':'Ciclo_Rec2',
                         'CR5':'Ciclo_Rec5',
                         'Agrup':'Agrup_Ciclo',
                         'Observação':'Obs',
                         'Endereço':'Pilha',
                         'Classificação da prioridade':'Prioridade',
                         'Limp Sup':'Limpeza'
                        }, inplace=True)

    colunas=['Data_Producao',
             'Volume',
             'Situacao',
             'Larg',
             'Diam', 
             'Peso',
             'Esp',
             'Prod',
             'Ciclo_Rec2',
             'Ciclo_Rec5',
             'Agrup_Ciclo',
             'Limpeza',
             'Obs',
             'Pilha',
             'Prioridade']

    data=data[colunas]

    data.Peso=data.Peso/1000
    data.Volume=data.Volume.astype(str)

    return data.reset_index(drop=True)

st.set_page_config(page_title="Estoque", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

data = st.file_uploader("Wybierz pliki CSV:",type = 'csv', accept_multiple_files=True)
DFL=pd.DataFrame()
for file in data:
   Table = pd.read_csv(file, sep=';', encoding='latin-1', thousands='.', decimal=',')
   DF1 = pd.DataFrame(Table)
   DFL = pd.concat([DFL,DF1], sort=False)

form = st.form(key="annotation", clear_on_submit=False)

with form:
    preview = st.form_submit_button(label="Carregar")

if preview:
    DFL=eda(DFL)
    st.session_state['estoque'] = DFL.reset_index(drop=True)
    st.write(pd.DataFrame(DFL))

st.sidebar.success("Select a demo above.")
