# Hoje vou iniciar uma análise de transcriptoma espacial de retina.
# Meu objetivo inicial é tentar encontrar nesse dataset regiões com maior degeneração, avaliando a relação RPE x morte celular
# -------------- Instação de pacotes necessários, e importar ------------------------
pip install merlin
pip install scanpy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import merlin
import scanpy as sc
import os

# Checar work directory

os.getcwd()
os.chdir("/Users/barbaradalmaso/Desktop/RPE-spatial/")

# Download dos datasets
adata = sc.read_h5ad("VA45_integrated.h5ad") # Em vars tem os genes. Como exportar um df corretamente?

# Criar DataFrame da matriz de expressão
expression_df = pd.DataFrame(
    adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
    index=adata.obs.index,  # Índice das células
    columns=adata.var.index  # Nomes dos genes
)
expression_df = pd.concat([adata.obs, expression_df], axis=1) # Integrar df com metadados

# Selecionar dados de cada amostra, escolher região com melhores tecidos # regiões de 0 a 7 e sample id de DV1-3 e TN1-3
filtered_df = expression_df[(expression_df['sampleid'] == 'TN3') & (expression_df['region'] == 0)] # O melhor

# Criar o scatter plot para essa região
plt.figure(figsize=(12, 10))
sns.scatterplot(
    x='center_x', y='center_y', hue='subclass', 
    data=filtered_df, palette='tab20', s=10, alpha=0.7, legend=False
)
plt.title('Distribuição Espacial das Subclasses de Células - Região 3', fontsize=16)
plt.xlabel('Coordenada X')
plt.ylabel('Coordenada Y')
plt.gca().invert_yaxis()

plt.show()

# Fazer uma análise exploratória nos meus datasets:
