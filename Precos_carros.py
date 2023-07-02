import torch
import jovian # estudar
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt #estudar
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

carro = 'Precos_carro.csv'
dataframe = pd.read_csv(carro)


seu_nome = 'João Aires' # no mínimo 5 caractéres
def custom_dataset(dataframe, rand_str):
    datafra = dataframe.copy(deep=True)
    #responsável por excluir uma porcentagem das linhas do dataframe.

    datafra = datafra.sample(int(0.95 * len(datafra)), random_state = int(ord(rand_str[0])))

    #responsável por dimensionar os valores da coluna "Ano" do dataframe.
    datafra.Ano = datafra.Ano * ord(rand_str[1])/100

    # responsável por dimensionar os valores da coluna "Precp_venda" do dataframe
    datafra.Preco_Venda = datafra.Preco_Venda * ord(rand_str[2])/100

    # responsavel por remover uma coluna especifica do dataframe
    if ord(rand_str[3]) % 2 == 1:
        datafra = datafra.drop(['Nome_carro'], axis=1)
    return datafra

datafra = custom_dataset(dataframe, seu_nome)

print(datafra.head())

input_colu = ["Ano","Preco_Atual","Kms_percorridos","Proprietario"]
categorica_colu = ["Tipo_de_combustivel","Vendedor_Tipo","Transmisscao"]
output_colu = ["Preco_Venda"]

def dataframe_matriz (datafra):
    # Faça uma cópia do dataframe original
    datafra1 = dataframe.copy(deep=True)

    #Converter colunas categóricas não numéricas em números
    for col in categorica_colu:
        datafra1[col] = datafra1[col].astype('category').cat.codes

    # Extraia entradas e saídas como matrizes numpy
    input_matrizes = datafra1[input_colu].to_numpy()
    alvo_matriz = datafra1[output_colu].to_numpy()

    return input_matrizes, alvo_matriz

input_matrizes, alvo_matriz = dataframe_matriz(datafra)

print(input_matrizes, alvo_matriz)
print('\n')

input = torch.Tensor(input_matrizes)
alvo = torch.Tensor(alvo_matriz)

dataset = TensorDataset(input, alvo)

treinam_ds, valida_ds = random_split(dataset, [241, 60])

tam_lotes = 128

carrega_treina = DataLoader(treinam_ds, tam_lotes, shuffle= True)
carrega_valida = DataLoader(valida_ds, tam_lotes)

tam_input = len(input_colu)
tam_output = len(output_colu)

class ModeloCarro(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(tam_input, tam_output)

    def forward(self, xb):
        out = self.linear(xb)
        return out

    def training_step(self, lotes):
        input, alvo = lotes
        out = self(input)
        loss = F.l1_loss(out, alvo)
        return loss

    def validation_step(self, lotes):
        input, alvo = lotes
        out = self(input)
        loss = F.l1_loss(out, alvo)
        return {'val_loss': loss.detach()}

    def validacao_fim(self, output):
        lotes_loss = [x['val_loss'] for x in output]
        epoch_loss = torch.stack(lotes_loss).mean()
        return {'val_loss': epoch_loss.item()}

    def epoca_end(self, epoca, resultado, num_epocas):
        if (epoca + 1) % 20 == 0 or epoca == num_epocas-1:
            print('Época [{}], val_loss: {:.2f}'.format(epoca+1, resultado['val_loss']))
            print('\n')

model = ModeloCarro()

list(model.parameters())

# Algoritmo de avaliação
def avalicao(model, carrega_valida ):
    outputs = [model.validation_step(lotes) for lotes in carrega_valida]
    return model.validacao_fim(outputs)

#Algoritmo de ajuste
def ajustar(epocas, lr, model, carrega_treina, carrega_valida, opt_func = torch.optim.SGD):
    historico = []
    otimizador = opt_func(model.parameters(), lr)
    for epoca in range(epocas):
        #fase de treino
        for lotes in carrega_treina:
            loss = model.training_step(lotes)
            loss.backward()
            otimizador.step()
            otimizador.zero_grad()
            #fase de validação
        resultado = avalicao(model, carrega_valida)
        model.epoca_end(epoca, resultado, epocas)
        historico.append((resultado))
    return historico

#Verifique o valor inicial que val_loss tem

resultado = avalicao(model, carrega_valida)
print(resultado)

# Ajustando
epocas = 90
lr = 1e-8
historico1 = ajustar(epocas, lr, model, carrega_treina, carrega_valida)

# Treine repetidamente até ter um 'bom' val_loss
epocas = 20
lr = 1e-9
historico1 = ajustar(epocas, lr, model, carrega_treina, carrega_valida)

# Algoritmo de previsão
def predict_single(input, alvo, model, dataset, dataframe):
    inputs = input.unsqueeze(0)
    predictions = model(inputs)
    prediction = predictions[0].detach().item()
    index = dataset.tensors[0].tolist().index(input.tolist())
    nome_carro = dataframe.iloc[index]['Nome_carro']
    print('\n')
    print('Carro: {}'.format(nome_carro))
    print('Entrada: {}'.format(input))
    print('Alvo: {:.2f}'.format(alvo.item()))
    print('Predição: {:.2f}'.format(prediction))

# Testando o modelo com algumas amostras
input, alvo = valida_ds[0]
predict_single(input, alvo, model, dataset, dataframe)

input, alvo = valida_ds[10]
predict_single(input, alvo, model, dataset, dataframe)