############# TCC DE MBA EM DATA SCIENCE & ANALYTICS #############
###################### UNIVERSIDADE DE SÃO PAULO #################
###################### ESTUDANTE BARBARA DALMASO #################

# In[0.2]: Importar pacotes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import os
import seaborn as sns
from scipy.io import mmread

# In[0.3]: Ajustar work-directory
os.getcwd()
os.chdir("/Users/barbaradalmaso/Desktop/RPE-spatial/Dados/")

# In[1.0]: Download de datasets
# Inicialmente, vou fazer o download de dois tipos importantes de datasets 
# 1. scRNA-seq (adata): Kuchroo, M., DiStasio, M., Song, E. et al. Single-cell analysis reveals inflammatory interactions driving macular degeneration. Nat Commun 14, 2589 (2023). https://doi.org/10.1038/s41467-023-37025-7
# 2. scRNA-seq espacial (sdata): Choi, J., Li, J., Ferdous, S. et al. Spatial organization of the mouse retina at single cell resolution by MERFISH. Nat Commun 14, 4929 (2023). https://doi.org/10.1038/s41467-023-40674-3

# This file contains the raw MERFISH count matrix for six samples with 500 gene features. The "sampleid" column represents the unique sample ID, while the "region" column corresponds to the tissue section ID. The "majorclass" and "subclass" columns indicate annotated retinal cell types. Finally, the "center_x" and "center_y" columns provide the coordinates of the cell centers.
sdata = sc.read_h5ad("VA45_integrated.h5ad") # https://zenodo.org/records/8144355 


##################### Processamento de dados de spatial scRNA-seq de retina saudável #############################
# Criar DataFrame da matriz de expressão
expression_df = pd.DataFrame(
    sdata.X.toarray() if hasattr(sdata.X, "toarray") else sdata.X,
    index=sdata.obs.index,  # Índice das células
    columns=sdata.var.index  # Nomes dos genes
)
expression_df = pd.concat([sdata.obs, expression_df], axis=1) # Integrar df com metadados
gene_list = expression_df.columns[11:]  # Seleciona apenas os nomes das colunas, que contém os nomes dos genes

# Selecionar dados de cada amostra, escolher região com melhores tecidos: regiões de 0-7 e sample id de DV1-3 e TN1-3
filtered_df = expression_df[(expression_df['sampleid'] == 'TN3') & (expression_df['region'] == 0)] # O melhor

# Criar o scatter plot para essa região
plt.figure(figsize=(50, 50))
sns.scatterplot(
    x='center_x', y='center_y', hue='subclass', 
    data=filtered_df, palette='tab20', s=30, alpha=0.7, legend=False
)
plt.title('Distribuição Espacial das Subclasses de Células', fontsize=16)
plt.xlabel('Coordenada X')
plt.ylabel('Coordenada Y')
plt.gca().invert_yaxis()

plt.show()

##################### Processamento de dados de scRNA-seq de retina saudável e degenerada #############################
# Primeiro eu preciso da lista de amostras para exportar os dados e saber quem é controle ou AMD
path = "/Users/barbaradalmaso/Desktop/RPE-spatial/Dados//GSE221042_RAW/metadata-sc.csv"
metadatasc = pd.read_csv(path, sep=';')
metadatasc['file_name'] = metadatasc['file_name'] + "_matrix.mtx.gz" # Adicionar a extensão para posterior valor
file_name = metadatasc['file_name'].values # Extrair os nomes dos arquivos

# Agora vou exportar os dataframes em forma de loop, para selecionar os dados de pacientes saudáveis e com AMD
# Utilizaremos como base os file_names obtidos anteriormente, e a lista de 500 genes utilizada no sequenciamento espacial
dataframes = [] # Dataframe vazio
path = "/Users/barbaradalmaso/Desktop/RPE-spatial/Dados//GSE221042_RAW/"

# Processar arquivos um a um
for file in metadatasc['file_name']:
    file_path = path + file  # Constrói o caminho completo
    print(f"Lendo arquivo: {file_path}")
    
    # Ler o arquivo .mtx como matriz esparsa
    matrix = mmread(file_path)
    
    # Verificar se é COO e converter para formato CSC ou CSR (que são indexáveis)
    sparse_matrix = matrix.tocsr()
    
    # Converter em DataFrame
    df = pd.DataFrame.sparse.from_spmatrix(sparse_matrix)
    
    # Filtrar apenas os genes que estão em gene_list
    df_filtered = df[gene_list]
    
    # Adicionar à lista
    dataframes.append(df_filtered)

# Concatenar tudo no final
adata = pd.concat(dataframes, ignore_index=True)


!!!!!!!!!!!!!!!!!!!!!!!!!!!!PARA 
RESOLVER -> MUDAR GENES DA NOMENCLATURA DE CAMUNDONGO PARA HUMANO!!!!!!!!!!!!!!!!!!!

