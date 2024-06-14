import os
import shutil
import csv
import numpy as np
import model_lib as ml

class SujeitoMoove:
    def __init__(self, nome):
        self.nome = nome
        self.janelas = {}
        self.diagnostic = -1

    def set_janelas(self, tabela_moove, atividade):
        tabela_suj = tabela_moove.loc[tabela_moove['subject_name'] == self.nome].iloc[:, 11:].sample(frac=1).reset_index(drop=True)
        if(not tabela_suj.empty):
            self.diagnostic = tabela_suj['LABEL'].loc[0]

            self.janelas[atividade] = tabela_suj.drop('LABEL', axis=1).to_numpy()


atividades = ['static', 'gait', 'toegait']
tam_janelas = 3
sensores = ['LSK', 'RUL', 'Trunk']
igualar_janelas = False
normalizar = True
tipo = 'xgboost'
particao = 'all'
excluir_suj_treino = True
remove_all = True

nome_tabela = f"moove_table_all_subjects_processed_{tam_janelas}s.csv"

tabela_raw = ml.tratar_tabela_csv(nome_tabela, acc=True, gyro=(particao == 'all'))

nomes_sujeitos = tabela_raw['subject_name'].unique()
sujeitos = [SujeitoMoove(nome) for nome in nomes_sujeitos]

modelos_all = []
if(excluir_suj_treino == False):
    modelos_all = ml.load_models()


modelos_atividades = {}
tabelas_atividade = {}
fatores_norm = {}

for atividade in atividades:
    tabela_atividade, fator_norm = ml.dados_treinamento(tabela_raw, atividade, sensores, normalizar)
    tabelas_atividade[atividade] = tabela_atividade
    fatores_norm[atividade] = fator_norm
    for sujeito in sujeitos:
        sujeito.set_janelas(tabela_atividade, atividade)

    if(excluir_suj_treino == False):
        modelo_atividade = [model for model in modelos_all if model.eq(tipo, 
                                                            atividade, 
                                                            tam_janelas, 
                                                            sensores, igualar_janelas, 
                                                            particao,
                                                            normalizar) == True] #atividade
    
    
        modelos_atividades[atividade] = modelo_atividade[0]



print('Análise Moove: Validação de modelos')
print('  Modelos testados:')
print(f'     Topologia: {tipo}')
print(f"     Tam. Janela: {tam_janelas}s")
print(f"     Normalizar: {normalizar}")
print(f"     Igualar Janelas: {igualar_janelas}")
print(f"     Sensores: {', '.join(sensores)}")
print(f"     Atividades: {', '.join(atividades)}")
print(f"     Particao: {particao}")


print('  Método de treinamento:')
if(excluir_suj_treino):
    if(remove_all):
        print('     Dados do sujeito removidos do treinamento')
    else:
        print(f'     30\% dos dados do sujeito usados na validação, se possível')
else:
    print('     Dados do sujeito usados no treinamento')

print('--------------------------##----------------------------')

for sujeito in sujeitos:
    print(f'Sujeito: {sujeito.nome}')
    print(f"Diagnostico: {'TEA' if sujeito.diagnostic == 1 else 'DT'}")
    print('Resposta dos modelos:')

    for atividade in sujeito.janelas:
        X_in = sujeito.janelas[atividade]
        modelo_atividade = None
        lim = 0
        if(excluir_suj_treino == False):
            modelo_atividade = modelos_atividades[atividade]
        else:
            tabela_atividade = tabelas_atividade[atividade]
            tabela_filtrada = tabela_atividade.loc[tabela_atividade['subject_name'] != sujeito.nome].reset_index(drop = True)

            X_add, y_add = np.array([]), None

            if(remove_all == False):
                size = X_in.shape[0]
                lim = round(size * 0.3)
                if(lim != 0):
                    X_in, X_add = X_in[:lim, :], X_in[lim:, :]
             

            modelo_atividade = ml.modelo_filtrado(tabela_filtrada, tipo, fatores_norm[atividade],
                                                  atividade, tam_janelas, sensores, igualar_janelas,
                                                  particao, normalizar, sujeito.nome,
                                                  X_add, sujeito.diagnostic)
            

        y = modelo_atividade.modelo.predict(X_in)

        diag_mean = np.mean(y)
        diag = round(diag_mean)

        print(f"      Modelo {atividade}: {diag_mean:.2f} -> {'TEA' if diag == 1 else 'DT'} - Janelas: {y.shape[0] if lim == 0 else lim}/{sujeito.janelas[atividade].shape[0]} - Sum: {sum(y)}")

    print('--------------------------FIMSUJEITO------------------------')

