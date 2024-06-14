class Modelo:
    def __init__(self, modelo, tipo_modelo, atividade, particao):
        self.modelo = modelo
        self.tipo_modelo = tipo_modelo
        self.atividade = atividade
        self.particao = particao


class SinalMoove:
    def __init__(self, sinal, atividade, particao):
        self.sinal = sinal
        self.atividade = atividade
        self.particao = particao


class Sujeito:
    def __init__(self, nome):
        self.nome = nome
        self.sinais = []
    def add_signal(self, sinal):
        self.sinais.append(sinal)