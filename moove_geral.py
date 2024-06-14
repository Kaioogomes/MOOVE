import model_lib as ml
import pandas as pd
#--------------------------------------------------------------------------------------------------------#




#channel_vec = [[True, True], [True, False], [False, True]]
channel_vec = [[True, True], [True, False]]
arquivos = [["moove_table_all_subjects_processed_5s.csv", 5], ["moove_table_all_subjects_processed_3s.csv", 3]]
#arquivos = [["../moove_table_sois_teste.csv"]]


for arquivo in arquivos:
    for channel_combination in channel_vec:
        # suffix = arquivo[1]
        suffix = ""

        if(channel_combination[0] == True):
            suffix += "ACC"
        if(channel_combination[1] == True):
            suffix += "GYRO"

        tabela = ml.tratar_tabela_csv(arquivo[0], channel_combination[0], channel_combination[1])

        modelos = [0,1]

        for modelo in modelos:
            ml.relatorio_moove(tabela, tam_janela=arquivo[1], channel_comb=channel_combination,tipo_modelo=modelo, exec_all=False, igualar_janelas=False, normalizar=True)

