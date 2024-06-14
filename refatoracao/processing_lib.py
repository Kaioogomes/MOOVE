import os
import shutil
import csv
import numpy as np
#import model_lib as ml
#import pandas as pd
# from scipy.signal import butter, filtfilt
# import util
# import pandas as pd

#----------------------------------#---------------------------------------


diretorio_fonte = './input_files/'
diretorio_processados = './processed_files/'
diretorio_corrompidos = './corrupt_files/'

atividades_interesse = ['static', 'gait', 'toegait']

#----------------------------------#---------------------------------------


class SignalInfo:
    def __init__(self, device, atividade, particao):
        self.device = device
        self.atividade = atividade
        self.particao = particao

class RawSignal:
    def __init__(self, sinal, device, atividade, particao):
        #self.nome_individuo = nome_individuo
        self.raw = sinal
        self.info = SignalInfo(device, atividade, particao)

class SinalMoove:
    def __init__(self):
        self.acc = 0
        self.gyro = 0
        self.info = 0
    def set_acc(self, acc_signal):
        self.acc = acc_signal
    def set_gyro(self, gyro_signal):
        self.gyro = gyro_signal
    def set_info(self, info):
        self.info = info

#----------------------------------#---------------------------------------

#TODO -> Indentificação automática dos campos de RawSignal
def read_file(f_dir):
    raw_signals_mat = []

    with open(f_dir, 'r') as arquivo:
        reader = csv.reader(arquivo, delimiter='\t')
        for row in reader:
            raw_signals_mat.append(row)
            #print(row)

    return raw_signals_mat



def read_aquisitions(mv=False):
    signals_raw = {}

    for f in os.listdir(diretorio_fonte):
        f_dir = diretorio_fonte + f
        
        campos = f.split(sep='_')
        
        nome = campos[0]
        device = campos[1]
        ativ_alias = campos[2]
        particao = campos[3]

        file_mat = np.array(read_file(f_dir))
        file_mat = file_mat.astype(float)

        raw_signal = RawSignal(file_mat, device, ativ_alias, particao)
        #proc_signals = tratar_sinais(raw_signals)
        #proc_signals = raw_signals

        #sinal_moove = SinalMoove(proc_signals, device, ativ_alias, particao)
        if nome not in signals_raw:
            signals_raw[nome] = [raw_signal]
        else:
            signals_raw[nome].append(raw_signal)

        if mv == True: 
            shutil.move(f_dir, diretorio_processados)

    return signals_raw


def correct_order(sinal_raw):   
    sinal_moove = SinalMoove()

    info = sinal_raw.info

    sinal_moove.set_info(info)

    if(info.particao == "all"):

        #acc_index = [0,1,2,6,7,8,12,13,14]
        acc_index = [6,7,8,12,13,14,0,1,2]

        #gyro_index = [3,4,5,9,10,11,15,16,17]
        gyro_index = [9,10,11,15,16,17,3,4,5]


        sinal_moove.set_acc(sinal_raw.raw[:, acc_index])
        sinal_moove.set_gyro(sinal_raw.raw[:, gyro_index]/8)
    else:
        sig_index = [3,4,5,6,7,8,0,1,2]
        
        if(info.particao == "acc"):
            sinal_moove.set_acc(sinal_raw.raw[:, sig_index])
        else:
            sinal_moove.set_gyro(sinal_raw.raw[:, sig_index]/8)

    return sinal_moove


def basic_features(signal):
    return np.concatenate([np.min(signal, axis=0),
                          np.max(signal,axis=0), 
                          np.mean(signal,axis=0), 
                          np.std(signal, axis=0)], axis=None)

def extract_features(signal):
    signal_diff = np.diff(signal, axis=0)
    signal_int = np.cumsum(signal, axis=0)

    aux_matrix = np.concatenate([signal[0:-1], signal_diff, signal_int[0:-1]], axis=1)

    return basic_features(aux_matrix)

def extract_input_matrix(sinal_moove, tam_janela_s):
    tam_janela = 100*tam_janela_s
    particao = sinal_moove.info.particao
    num_janelas = 0

    if(particao == "gyro"):
        num_janelas = sinal_moove.gyro.shape[0] // tam_janela
    else:
        num_janelas = sinal_moove.acc.shape[0] // tam_janela

    janelas = []

    for i in range(num_janelas):
        if(particao == "all"):
            janelas.append(np.concatenate([extract_features(sinal_moove.acc[i*num_janelas:(i+1)*num_janelas, :]),
                                           extract_features(sinal_moove.gyro[i*num_janelas:(i+1)*num_janelas, :])],
                                           axis=None))
        elif(particao == "acc"):
            janelas.append(extract_features(sinal_moove.acc[i*tam_janela:(i+1)*tam_janela, :]))
        else:
            janelas.append(extract_features(sinal_moove.gyro[i*tam_janela:(i+1)*tam_janela, :]))


    return np.array(janelas)


def mapear_atividade(ativ_alias):
    if('static' in ativ_alias):
        return 'static'
    elif('toegait' in ativ_alias):
        return 'toegait'
    elif('gait' in ativ_alias):
        return 'gait'

    return '?'

