import processing_lib as pl
import model_lib as ml

def classify_frames(raw_signals, modelos, settings):
    
    acumulado_all = {"moove": [], "delsys": []}
    acumulado_acc = {"moove": [], "delsys": []}

    frames = {}
    resultado = {}
    
    for rsignal in raw_signals:
        psignal = pl.correct_order(rsignal)
        info = rsignal.info

        if info.atividade not in frames:
            frames[info.atividade] = {}
        if info.particao not in frames[info.atividades]:
            frames[info.atividade][info.particao] = {}

        X_input = pl.extract_input_matrix(psignal, settings.tam_janela)

        modelo_esp = [modelo for modelo in modelos if modelo.settings == settings]
        modelo_esp = modelo_esp[0]

        if(settings.normalizar):
            fator_norm = modelo_esp.fator_norm
            X_input = (X_input - fator_norm[0])/(fator_norm[1] - fator_norm[0])

        y_pred = modelo_esp.modelo.predict(X_input)

        if info.device not in frames[info.atividade][info.particao]:
            frames[info.atividade][info.particao][info.device] = y_pred

        