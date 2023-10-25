# 💼 • Tracking System

Sistema de rastreio e identificação de pessoas.

## 🌱 • Como contribuir

### ⚙️ • Setup

Execute pelo terminal o script pela raiz do projeto: `./scripts/setup.sh`

### 🔧 • Instalar nova biblioteca no projeto

Execute o script pela raiz do projeto: `./scripts/install.sh <Biblioteca>`

### ☕ • Executar

Execute o script pela raiz do projeto: `./scrips/execute.sh`. Utilize os seguintes argumentos:

- `--use_pre_trained`: `True` or `False`. Descrição: faz o modelo utilizar o melhor treinamento já realizado.
- `--model`: Indica o modelo a ser utilizado. Por padrão será o `yolov8n.pt`
- `--dataset`: Indica o dataset a ser utilizado. Por padrão será o `data-crowd-humans.pt`
- `--epochs`: Indica quanta épocas o modelo será treinado. O modelo só é treinado quando o parâmetro `--use-pre-trained` não é passado.
- `mode`: `image`, `video` ou `stream`: Indica o modo que o modelo será executado para detecção.
- `video_name` e `image_name`: Nome do arquivo para ser utilizado.

## 🧑 • Colaborades

- Gabriel da Silva Morishita Garbi
- Bruno Seki Schenberg
- Leonardo Santos Rocha

## 🖥️ • Especificações

O projeto está sendo construido utilizando `Linux` e `Cuda`. Por favor, verifique as dependências do projeto no `requirements.txt`
