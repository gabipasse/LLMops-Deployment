# Use uma imagem base oficial do Python
FROM python:3.12

# Defina variáveis de ambiente para não solicitar entrada interativa
ENV DEBIAN_FRONTEND=noninteractive

# Crie e defina o diretório de trabalho
WORKDIR /app

# Copie o arquivo requirements.txt para o diretório de trabalho
COPY requirements.txt .

# Instale as dependências do MLflow e outras dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copie o restante do código da aplicação para o contêiner
COPY . .

# Exponha a porta que o MLflow usará
EXPOSE 5000

# Comando para iniciar o servidor MLflow
CMD ["mlflow", "server", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "./mlflow-artifacts", "--app-name", "basic-auth", "--port", "5000"]
