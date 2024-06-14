import pandas as pd
import sys
import numpy as np
import sklearn.model_selection
import sklearn.metrics 
import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import os
import pickle
from joblib import dump, load


##Filtro sobre a tabela original na atividade e sensores de interesse
#  -> Opção de nomalização
#todo: implementar a opção de fdr_sort



def dados_treinamento(tabela_moove, atividade, sensores, normalizar=False, fdr_sort=False):
    ativ_table = tabela_moove.loc[tabela_moove['task_name'] == atividade]

    dados_s = pd.DataFrame()
    dados_info = 0
    dados_label = 0

    info = False

    for sensor in sensores:
        tabela_sensor = ativ_table[ativ_table['sensor_position'] == sensor].reset_index(drop=True)

        if(info == False):
            dados_info = tabela_sensor.iloc[:, :11]
            dados_label = pd.DataFrame({'LABEL': (tabela_sensor['subject_type'].values == 'TEA').astype(int)})
            info = True

        tabela_sensor = tabela_sensor.iloc[:, 11:].add_suffix(' ' + sensor)
        fator_norm = []

        dados_s = pd.concat([dados_s, tabela_sensor],axis=1)  

    if(normalizar):
            fator_norm = [dados_s.min().to_numpy(), dados_s.max().to_numpy()]

            dados_s = (dados_s - fator_norm[0])/(fator_norm[1] - fator_norm[0]) 

    
    dados_s = pd.concat([dados_info, dados_s,dados_label],axis=1)


    return dados_s.sample(frac=1).reset_index(drop=True), fator_norm

## Função de análise da distribuição TEA-DT em um dataset

def divisao_classes(dados_t, igualar_janelas,arquivo=sys.stdout):
    index_tea, index_dt = dados_t['subject_type'] == "TEA", dados_t['subject_type'] == "DT" 

    num_tea, num_dt = index_tea.sum(), index_dt.sum() 

    num_i_tea = len((dados_t.loc[index_tea])['subject_name'].unique())
    num_i_dt  = len((dados_t.loc[index_dt])['subject_name'].unique()) 

    print('\t Divisao TEA - DT:\n', file=arquivo)
    print(f"\t\tJanelas TEA: {num_tea}", file=arquivo)
    print(f"\t\tJanelas DT : {num_dt}\n\t\t  --> Prop. Janelas: {round(num_tea/(num_tea + num_dt), 2)}\n\t\t", file=arquivo) 
    print(f"\t\tNum. Ind. TEA: {num_i_tea}\n\t\tNum. Ind. DT : {num_i_dt}\n\t\t\t", file=arquivo)
    print(f"\t\t  --> Prop. IND.   : {round(num_i_tea/(num_i_tea + num_i_dt), 2)}\n", file=arquivo)
    

    if(igualar_janelas == True):
        
        print("Padronização de Janelas Ativado.\n", file=arquivo)

        index_dt = [indice for indice, valor in enumerate(index_dt) if valor]
        index_tea = [indice for indice, valor in enumerate(index_tea) if valor]

        if(num_tea > num_dt):
            dados_dt = dados_t.loc[index_dt].reset_index(drop=True)
            dados_tea = dados_t.loc[index_tea[0:num_dt]].reset_index(drop=True)
            print(f"Numero de Janelas por classe: {num_dt}\n", file=arquivo)
        elif(num_tea < num_dt):
            dados_dt = dados_t.loc[index_dt[0:num_tea]].reset_index(drop=True)
            dados_tea = dados_t.loc[index_tea].reset_index(drop=True)
            print(f"Numero de Janelas por classe: {num_tea}\n", file=arquivo)

        dados_t = pd.concat([dados_dt, dados_tea], axis=0).reset_index(drop=True)
        
        index_tea, index_dt = dados_t['LABEL'] == 1, dados_t['LABEL'] == 0
        num_tea, num_dt = index_tea.sum(), index_dt.sum() 

        num_i_tea = len((dados_t.loc[index_tea])['subject_name'].unique())
        num_i_dt  = len((dados_t.loc[index_dt])['subject_name'].unique())

        print('\t Divisao TEA - DT Pós Padronização:\n', file=arquivo)
        print(f"\t\tJanelas TEA: {num_tea}", file=arquivo)
        print(f"\t\tJanelas DT : {num_dt}\n\t\t  --> Prop. Janelas: {round(num_tea/(num_tea + num_dt), 2)}\n\t\t", file=arquivo) 
        print(f"\t\tNum. Ind. TEA: {num_i_tea}\n\t\tNum. Ind. DT : {num_i_dt}\n\t\t\t", file=arquivo)
        print(f"\t\t  --> Prop. IND.   : {round(num_i_tea/(num_i_tea + num_i_dt), 2)}\n", file=arquivo)

        

    return dados_t.sample(frac=1).reset_index(drop=True), [num_tea, num_dt]
## Cálculo de métricas importantes

def metricas_classificador(y_real, y_pred):

    f1_scr = sklearn.metrics.f1_score(y_real, y_pred)
    acc    = sklearn.metrics.accuracy_score(y_real, y_pred)

    cf = sklearn.metrics.confusion_matrix(y_real, y_pred)

    tn, fp, fn, tp = cf.ravel()

    precision = tp/(tp + fp) #Ou sensibilidade
    recall = tp/(tp + fn)
    specificity = tn/(tn + fp)

    return cf, [f1_scr, acc, precision, recall, specificity]

## 

def print_metrics(metricas, arquivo=sys.stdout):
    nome_metricas = ['F1 Score', 'Accuracy', 'Precision', 'Recall', 'Specificity']
    max_key_length = max(len(key) for key in nome_metricas)

    if(np.isnan(np.sum(metricas))):
        print('Erro em todas as folds! Sem resultados.\n', file=arquivo)
    else:
        for index, nome in enumerate(nome_metricas):
            print(f"\t  {nome.ljust(max_key_length)}: {metricas[0][index]:.2f} +- {metricas[1][index]:.2f}", file=arquivo)

## Implementação do kfolds
#todo: implementar a opção de retornar também o resultado de treinamento
#todo:      ||        ||   de printar o desempenho de cada fold

def kfold_cv(dados_atividade, modelo = 'log_reg', kfold=5, print_folds=False, print_train=False, arquivo=sys.stdout):
    dados_atividade = dados_atividade.sample(frac=1).reset_index(drop=True).iloc[:, 11:]
    
    y = dados_atividade['LABEL'].to_numpy()
    X = dados_atividade.drop('LABEL', axis=1).to_numpy()
    

    kf = sklearn.model_selection.KFold(n_splits=kfold)
    kf.get_n_splits(X)

    metrics_train = []
    metrics_test = []

    cf_test = []
    cf_train = []

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index] 
        y_train, y_test = y[train_index], y[test_index]
        
        if(modelo == 'xgboost'):
            classificador = XGBClassifier(objective='binary:logistic')
        elif(modelo == 'svm'):
            classificador = svm.NuSVC(gamma="auto")
        elif(modelo == 'log_reg'):
            classificador = LogisticRegression(max_iter=1000) 


        try:
            classificador.fit(X_train, y_train)

            yhat_train = classificador.predict(X_train)
            yhat_test = classificador.predict(X_test)

            
            cf_test, metric_test_f = metricas_classificador(y_test, yhat_test)
            metrics_test.append(metric_test_f)

            if(print_train == True):
                cf_train, metric_train_f = metricas_classificador(y_train, yhat_train)
                metrics_train.append(metric_train_f)

            if(print_folds == True):
                print('Teste', file= arquivo)
        except:
            print('\tErro de parâmetro no fold! Checar distribuição ou dimensionamento dos dados', file = arquivo)

    folds_completas = len(metrics_test)
    desempenho_test = [np.nan, np.nan]
    if(folds_completas != 0):
        desempenho_test = [np.mean(metrics_test, axis=0), np.std(metrics_test, axis=0)]
    

    return [desempenho_test, folds_completas]

##

#class ModelSettings:


class ModeloMoove:
    def __init__(self, tipo, modelo, fator_norm):
        self.tipo = tipo
        self.atividade = 0
        self.tam_janela = 0
        self.sensores = 0
        self.igualar_janelas = 0
        self.particao = 0
        self.modelo = modelo
        self.normalizado = 0
        self.fator_norm = fator_norm
        self.excluded = None

    def set_info(self, atividade, tam_janela, sensores, igualar_janelas, particao, normalizado, excluded=None):
        self.atividade = atividade
        self.tam_janela = tam_janela
        self.sensores = sensores
        self.igualar_janelas = igualar_janelas
        self.particao = particao
        self.normalizado = normalizado
        if(excluded != None):
            self.excluded = excluded

    def eq(self, tipo, atividade, tam_janela, sensores, igualar_janelas, particao, normalizado):
        return(((self.tipo == tipo) and
               (self.atividade == atividade) and
               (self.tam_janela == tam_janela) and
               (self.sensores == sensores) and
               (self.igualar_janelas == igualar_janelas) and
               (self.particao == particao) and
               (self.normalizado == normalizado)))

def gerar_modelo_moove(dados_atividade, modelo_tipo, fator_norm, X_add=np.array([]), diag=None):
    dados_atividade = dados_atividade.sample(frac=1).reset_index(drop=True).iloc[:, 11:]
    
    y = dados_atividade['LABEL'].to_numpy()
    X = dados_atividade.drop('LABEL', axis=1).to_numpy()

    if((len(X_add) != 0) and (diag !=None)):
        dados = np.concatenate([X, np.transpose([y])], axis=1)
        y_add = np.array([diag]*X_add.shape[0])
        dados_add = np.concatenate([X_add, np.transpose([y_add])], axis=1)
        dados = np.concatenate([dados, dados_add], axis=0)
        np.random.shuffle(dados)
        X = dados[:, :-1]
        y = dados[:, -1]
    
    
    if(modelo_tipo == 'xgboost'):
        classificador = XGBClassifier(objective='binary:logistic')
    elif(modelo_tipo == 'svm'):
        classificador = svm.NuSVC(gamma="auto")
    elif(modelo_tipo == 'log_reg'):
        classificador = LogisticRegression(max_iter=1000) 
    
    classificador.fit(X, y)

    modelo = ModeloMoove(modelo_tipo, classificador, fator_norm)

    return modelo
##

def tratar_tabela_csv(nome_tabela, acc=True, gyro=True):
    table = pd.read_csv(nome_tabela, index_col=False)

    tabela_acc = table.loc[table['channel_group'] == "ACC"].reset_index(drop=True)
    tabela_final = tabela_acc.iloc[:, :11]


    if(acc == True):
        tabela_acc = tabela_acc.iloc[:, 11:].reset_index(drop=True).add_suffix(' ACC')
        tabela_final = pd.concat([tabela_final, tabela_acc], axis=1)
        #tabela_final = tabela_acc

    if(gyro == True):
        tabela_gyro = table.loc[table['channel_group'] == "GYRO"].iloc[:, 11:].reset_index(drop=True).add_suffix(' GYRO')
        tabela_final = pd.concat([tabela_final, tabela_gyro], axis=1)



    return tabela_final.reset_index(drop=True)


def relatorio_moove(table, tam_janela, channel_comb,name_suffix="", tipo_arquivo=0, tipo_modelo=1, sensores=['LSK', 'RUL', 'Trunk'], exec_all=True, n_fold=5, igualar_janelas=True, normalizar=True, excluded=None):

    tipos_arquivo = [['janelasUnif', '']],
                    #['SOI mínimo','_nunif'], 
                    #['min_seg_q','_min_m1stq'], 
                    #['SOI mínimo','_min_natg5']]


    #taq = 1

    #arquivo = 'moove_table_alls_jan5s_fc1=0_fc2=10_janelas_nunif' + tipos_arquivo[taq][1] + '.csv'

    #table = pd.read_csv(nome_tabela, index_col=False)

    #todo: mais modelos?
    modelos = ['xgboost', 'svm', 'log_reg']

    modelo = modelos[tipo_modelo] 


    #todo: Outras possibilidades de trincas
    #sensores = ['LSK', 'RUL', 'Trunk']

    #exec_all = True

    atividades = table['task_name'].unique()
    if(exec_all == False):
        atividades = ['static', 'gait', 'toegait']

    rel_dir = "./relatorios/"
    nome_arquivo_out = rel_dir + f"relatorio_moove_{tipos_arquivo[tipo_arquivo][0][0]}{tam_janela}s_modelSel_{modelo}_{sensores[0]}_{sensores[1]}_{sensores[2]}_igualar{igualar_janelas}_normalizar={normalizar}_{name_suffix}.txt"

    #----------------------'----------------------------------------------------------------------------------#

    #n_fold = 5

    #igualar_janelas = False

    with open(nome_arquivo_out, 'w') as arquivo:
        print("Moove - Model Selection print", file=arquivo)
        print('\n----------------------------------------------------\n\n', file=arquivo)
        print(f" Modelo: {modelo}\n Sensores = {sensores[0]}, {sensores[1]}, {sensores[2]}", file=arquivo)


        for atividade in atividades:

            print('\n----------------------------------------------------\n\n', file=arquivo)
            print('Atividade: ' + atividade + ':\n', file=arquivo)

            dados_atividade, fator_norm = dados_treinamento(table, atividade, sensores, normalizar)

            dados_atividade, janelas = divisao_classes(dados_atividade, arquivo=arquivo, igualar_janelas=igualar_janelas)

            desempenho_ativ, folds_completas = kfold_cv(dados_atividade, modelo, kfold=n_fold, arquivo=arquivo)

            if(folds_completas != n_fold):
                print(f"Erro em uma ou mais folds!!!\nFolds completas: {folds_completas}\n", file=arquivo)

            print_metrics(desempenho_ativ, arquivo=arquivo)
            print("\n", file=arquivo)

            particao = 0
            if((channel_comb[0] == True) and (channel_comb[1] == True)):
                particao = "all"
            elif(channel_comb[0] == True):
                particao = "acc"
            else:
                particao = "gyro"

            modelo_moove = gerar_modelo_moove(dados_atividade, modelo, fator_norm)
            modelo_moove.set_info(atividade, tam_janela, sensores, igualar_janelas, particao, normalizar)

            dir_modelos = "./modelos_t/"

            dump(modelo_moove, f"{dir_modelos}_{modelo}_{atividade}_{''.join(sensores)}_igJan={igualar_janelas}_{particao}_normalizar={normalizar}_excluded={excluded}.joblib")

    print("Relatório completo.\n Arquvio salvo: " + nome_arquivo_out)


def modelo_filtrado(dados_filtrados, tipo_modelo, fator_norm,
                    atividade, tam_janela, sensores, igualar_janelas,
                    particao, normalizar, excluido,
                    X_add=np.array([0]), diag=None):

    modelo_moove = gerar_modelo_moove(dados_filtrados, tipo_modelo, fator_norm, X_add, diag)
    modelo_moove.set_info(atividade, tam_janela, sensores, igualar_janelas, particao, normalizar, excluido)

    return modelo_moove

def load_models():
    dir_modelos = "./modelos_t/"
    modelos = []

    for f in os.listdir(dir_modelos):
        f_dir = dir_modelos + f
        modelos.append(load(f_dir))

    return modelos