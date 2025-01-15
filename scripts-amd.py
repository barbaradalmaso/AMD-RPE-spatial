############# TCC DE MBA EM DATA SCIENCE & ANALYTICS #############
###################### UNIVERSIDADE DE SÃO PAULO #################
###################### ESTUDANTE BARBARA DALMASO #################

# In[0.1]: Importar pacotes
# pip install numpy pandas matplotlib scanpy seaborn scipy anndata scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import os
import seaborn as sns
from scipy.io import mmread
from scipy.sparse import csr_matrix
import anndata as ad
from sklearn.decomposition import PCA
from scipy.stats import pearsonr


# In[0.2]: Ajustar work-directory
os.getcwd()
os.chdir("/Users/barbaradalmaso/Desktop/AMD-RPE-spatial/Dados/")

# In[1.0]: Download de datasets - scRNA-seq espacial
# Inicialmente, vou fazer o download de dois tipos importantes de datasets 
# 1. scRNA-seq (adata): Kuchroo, M., DiStasio, M., Song, E. et al. Single-cell analysis reveals inflammatory interactions driving macular degeneration. Nat Commun 14, 2589 (2023). https://doi.org/10.1038/s41467-023-37025-7
# 2. scRNA-seq espacial (sdata): Choi, J., Li, J., Ferdous, S. et al. Spatial organization of the mouse retina at single cell resolution by MERFISH. Nat Commun 14, 4929 (2023). https://doi.org/10.1038/s41467-023-40674-3

# This file contains the raw MERFISH count matrix for six samples with 500 gene features. The "sampleid" column represents the unique sample ID, while the "region" column corresponds to the tissue section ID. The "majorclass" and "subclass" columns indicate annotated retinal cell types. Finally, the "center_x" and "center_y" columns provide the coordinates of the cell centers.
sdata = sc.read_h5ad("merfish_impute.h5ad") # https://zenodo.org/records/8144355 

##################### Processamento de dados de spatial scRNA-seq de retina saudável #############################
# Criar DataFrame da matriz de expressão
expression_df = pd.DataFrame(
    sdata.X.toarray() if hasattr(sdata.X, "toarray") else sdata.X,
    index=sdata.obs.index,  # Índice das células
    columns=sdata.var.index  # Nomes dos genes
)

# Selecionar os genes e converter nomenclatura de camundongo para humano
gene_list = expression_df.columns.tolist() # Seleciona apenas os nomes das colunas, que contém os nomes dos genes
gene_list_human = [gene.upper() for gene in gene_list]  # Converter genes de camundongo para humano

# In[1.1]: Download de datasets - scRNA-seq convencional
##################### Processamento de dados de scRNA-seq de retina saudável e degenerada #############################
# Primeiro eu preciso da lista de amostras para exportar os dados e saber quem é controle ou AMD
# Decidi exportar separadamente cada grupo da pesquisa, para conseguir fazer as análises exploratórias separadamente

# Caminho para os dados
path = "/Users/barbaradalmaso/Desktop/AMD-RPE-spatial/Dados/GSE221042_RAW/"

def process_files(metadata_path, gene_list, output_path):
    # Ler metadados
    metadatasc = pd.read_csv(metadata_path, sep=';')
    metadatasc['barcodes'] += "_barcodes.tsv.gz"
    metadatasc['features'] += "_features.tsv.gz"
    metadatasc['matrix'] += "_matrix.mtx.gz"
    
    # Inicializar AnnData
    all_data = []
    
    for barcodes_file, features_file, matrix_file in zip(metadatasc['barcodes'], metadatasc['features'], metadatasc['matrix']):
        # Ler os dados
        barcodes = pd.read_csv(f"{path}{barcodes_file}", header=None, sep="\t")[0].values
        features = pd.read_csv(f"{path}{features_file}", header=None, sep="\t")[1].values
        matrix = mmread(f"{path}{matrix_file}").tocsr()
        
        # Filtrar genes de interesse
        valid_genes = np.intersect1d(features, gene_list)
        gene_indices = np.where(np.isin(features, valid_genes))[0]
        filtered_matrix = matrix[gene_indices, :]
        
        # Criar AnnData
        adata = sc.AnnData(X=filtered_matrix.T, var=pd.DataFrame(index=valid_genes), obs=pd.DataFrame(index=barcodes))
        all_data.append(adata)
    
    # Concatenar todos os dados
    combined_data = sc.concat(all_data)
    combined_data.write(output_path)

# Processar os grupos
process_files(f"{path}metadata-sc-control.csv", gene_list_human, f"{path}control.h5ad")
process_files(f"{path}metadata-sc-wet-amd.csv", gene_list_human, f"{path}wet_amd.h5ad")
process_files(f"{path}metadata-sc-dry-amd.csv", gene_list_human, f"{path}dry_amd.h5ad")

# Ao término, vou ter 3 arquivos h5ad com os dados conjuntos de 4 pacientes cada, e os diferentes backgrouds patogenicos.

# In[1.2]: Análise exploratória scRNA-seq com pacientes
# Agora nesse etapa, vou fazer uma análise exploratória dos dados. Inicialmente, irei verificar se os dados tem uma tendencia à normalidade.
wet = sc.read_h5ad("/Users/barbaradalmaso/Desktop/AMD-RPE-spatial/Dados/GSE221042_RAW/wet_amd.h5ad") 
dry = sc.read_h5ad("/Users/barbaradalmaso/Desktop/AMD-RPE-spatial/Dados/GSE221042_RAW/dry_amd.h5ad") 
control = sc.read_h5ad("/Users/barbaradalmaso/Desktop/AMD-RPE-spatial/Dados/GSE221042_RAW/control.h5ad") 

# Primeiro, como observei que a maior parte das celulas possuem baixo count de genes, decidi filtrar 
# as celulas com total count abaixo de 500 (em cada amostra)
control.obs['total_counts'] = control.X.sum(axis=1).A1 if isinstance(control.X, csr_matrix) else control.X.sum(axis=1)
control_filtered = control[(control.obs['total_counts'] > 1000) & (control.obs['total_counts'] < 10000), :]
print(f"Número de células antes da filtragem: {control.n_obs}")
print(f"Número de células após a filtragem: {control_filtered.n_obs}")

dry.obs['total_counts'] = dry.X.sum(axis=1).A1 if isinstance(dry.X, csr_matrix) else dry.X.sum(axis=1)
dry_filtered = dry[(dry.obs['total_counts'] > 1000) & (dry.obs['total_counts'] < 10000), :]
print(f"Número de células antes da filtragem: {dry.n_obs}")
print(f"Número de células após a filtragem: {dry_filtered.n_obs}")

wet.obs['total_counts'] = wet.X.sum(axis=1).A1 if isinstance(wet.X, csr_matrix) else wet.X.sum(axis=1)
wet_filtered = wet[(wet.obs['total_counts'] > 1000) & (wet.obs['total_counts'] < 10000), :]
print(f"Número de células antes da filtragem: {wet.n_obs}")
print(f"Número de células após a filtragem: {wet_filtered.n_obs}")

# Como tirei as celulas que nao possuem nenhuma contagem de expressao genica, o numero de celulas por amostra reduziu drasticamente.
# Agora eh possivel que eu junte todas as amostras de pacientes em um unico dataset, pois nao ficara mais um arquivo tao pesado.
# Mas para isso, necessito fazer metadados de qualidade para nao perder o agrupamento de pacientes.

# Adicionar coluna 'group' aos metadados de cada objeto
control_filtered.obs['group'] = 'control'
dry_filtered.obs['group'] = 'dry_amd'
wet_filtered.obs['group'] = 'wet_amd'

# Combinar os objetos AnnData
combined_data = ad.concat([control_filtered, dry_filtered, wet_filtered], label="group", keys=["control", "dry_amd", "wet_amd"])

# Salvar o objeto combinado em um arquivo .h5ad
output_path = "/Users/barbaradalmaso/Desktop/AMD-RPE-spatial/Dados/combined_patient_sc.h5ad"
combined_data.write(output_path)

# Nessa etapa irei finalmente fazer as analises exploratorias. Irei mostrar os dados de pacientes tanto separados, quanto juntos
####### Teste de normalidade #########
# Criar histogramas para os dados filtrados individuais e o combinado
sns.set_context("talk", font_scale=1.2)  # Aumenta o tamanho das fontes
sns.set_style("whitegrid")  # Adiciona grid para melhor visualização

# Criar histogramas para os dados filtrados individuais e o combinado
fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharey=True)

# Histograma do grupo control
sns.histplot(control_filtered.obs['total_counts'], bins=50, kde=False, ax=axes[0], color='blue', linewidth=2)
axes[0].set_title('Control', fontsize=20, weight='bold')
axes[0].set_xlabel('Total Counts', fontsize=20)
axes[0].set_ylabel('Frequency', fontsize=20)

# Histograma do grupo dry_amd
sns.histplot(dry_filtered.obs['total_counts'], bins=50, kde=False, ax=axes[1], color='orange', linewidth=2)
axes[1].set_title('Dry AMD', fontsize=20, weight='bold')
axes[1].set_xlabel('Total Counts', fontsize=20)
axes[1].set_ylabel('')

# Histograma do grupo wet_amd
sns.histplot(wet_filtered.obs['total_counts'], bins=50, kde=False, ax=axes[2], color='green', linewidth=2)
axes[2].set_title('Wet AMD', fontsize=20, weight='bold')
axes[2].set_xlabel('Total Counts', fontsize=20)
axes[2].set_ylabel('')

# Histograma dos dados combinados
sns.histplot(combined_data.obs['total_counts'], bins=50, kde=False, ax=axes[3], color='purple', linewidth=2)
axes[3].set_title('Combined samples', fontsize=20, weight='bold')
axes[3].set_xlabel('Total Counts', fontsize=20)
axes[3].set_ylabel('')

# Ajustar layout
plt.tight_layout()
plt.show()

####### Teste PCA #########
# Função para realizar PCA
def perform_pca(data, n_components=2):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data.X.toarray() if isinstance(data.X, csr_matrix) else data.X)
    explained_variance = pca.explained_variance_ratio_
    return pca_result, explained_variance

# PCA para os conjuntos de dados filtrados
control_pca, control_var = perform_pca(control_filtered)
dry_pca, dry_var = perform_pca(dry_filtered)
wet_pca, wet_var = perform_pca(wet_filtered)
combined_pca, combined_var = perform_pca(combined_data)

# Criar gráficos de PCA
fig, axes = plt.subplots(1, 4, figsize=(24, 6))

# Control
axes[0].scatter(control_pca[:, 0], control_pca[:, 1], alpha=0.5, color='blue')
axes[0].set_title(f'Control\nPC1: {control_var[0]*100:.2f}%, PC2: {control_var[1]*100:.2f}%', fontsize=16)
axes[0].set_xlabel('PC1', fontsize=20)
axes[0].set_ylabel('PC2', fontsize=20)

# Dry AMD
axes[1].scatter(dry_pca[:, 0], dry_pca[:, 1], alpha=0.5, color='orange')
axes[1].set_title(f'Dry AMD\nPC1: {dry_var[0]*100:.2f}%, PC2: {dry_var[1]*100:.2f}%', fontsize=16)
axes[1].set_xlabel('PC1', fontsize=20)
axes[1].set_ylabel('PC2', fontsize=20)

# Wet AMD
axes[2].scatter(wet_pca[:, 0], wet_pca[:, 1], alpha=0.5, color='green')
axes[2].set_title(f'Wet AMD\nPC1: {wet_var[0]*100:.2f}%, PC2: {wet_var[1]*100:.2f}%', fontsize=16)
axes[2].set_xlabel('PC1', fontsize=20)
axes[2].set_ylabel('PC2', fontsize=20)

# Combined
axes[3].scatter(combined_pca[:, 0], combined_pca[:, 1], alpha=0.5, color='purple')
axes[3].set_title(f'Combined samples\nPC1: {combined_var[0]*100:.2f}%, PC2: {combined_var[1]*100:.2f}%', fontsize=16)
axes[3].set_xlabel('PC1', fontsize=20)
axes[3].set_ylabel('PC2', fontsize=20)

# Ajustar layout
plt.tight_layout()
plt.show()

# In[1.3]: Análise exploratória scRNA-seq espacial
# Carregar os dados espaciais
sdata = sc.read_h5ad("/Users/barbaradalmaso/Desktop/AMD-RPE-spatial/Dados/merfish_impute.h5ad")
bulkdata = sc.read_h5ad("/Users/barbaradalmaso/Desktop/AMD-RPE-spatial/Dados/combined_patient_sc.h5ad")

# Selecionar apenas os genes presentes na lista 'gene_list_expression' (geralmente da amostra dry)
sdata.var.index = sdata.var.index.str.upper()
gene_list_expression = bulkdata.var.index
sdata_filtered_genes = sdata[:, sdata.var.index.isin(gene_list_expression)]

# Filtrar as células com base no total de contagens de forma otimizada
if isinstance(sdata_filtered_genes.X, csr_matrix):
    sdata_filtered_genes.obs['total_counts'] = np.array(sdata_filtered_genes.X.sum(axis=1)).flatten()
else:
    sdata_filtered_genes.obs['total_counts'] = sdata_filtered_genes.X.sum(axis=1)

# Filtrar células com total_count > 500
sdata_filtered = sdata_filtered_genes[(sdata_filtered_genes.obs['total_counts'] > 1000) & (sdata_filtered_genes.obs['total_counts'] < 10000),:]


# Exibir informações sobre a filtragem
print(f"Dimensão original de sdata: {sdata.shape}")
print(f"Dimensão após a filtragem de genes e total_counts > 500: {sdata_filtered.shape}")

# Salvar o objeto combinado em um arquivo .h5ad
output_path = "/Users/barbaradalmaso/Desktop/AMD-RPE-spatial/Dados/filtered_spatialdata.h5ad"
sdata_filtered.write(output_path)

###### Teste de normalidade ######
# Histograma da contagem de genes por célula
sns.set_context("talk", font_scale=1.2)  # Aumenta o tamanho das fontes
sns.set_style("whitegrid")  # Adiciona grid para melhor visualização

# Criar o gráfico com as contagens totais de cada célula após a filtragem
plt.figure(figsize=(6, 6))
sns.histplot(sdata_filtered.obs['total_counts'], bins=50, kde=False, color='red', linewidth=2)
plt.title('Spatial data', fontsize=20, weight='bold')
plt.xlabel('Total Counts', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.show()

# Função para realizar PCA
def perform_pca(data, gene_list, n_components=2):
    # Extraindo a matriz de expressão para os genes de interesse
    gene_expression_data = data[:, data.var.index.isin(gene_list_expression)].X
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(gene_expression_data)  # Realiza o PCA nos dados de expressão
    explained_variance = pca.explained_variance_ratio_
    return pca_result, explained_variance

# Realizar PCA
pca_result, pca_var = perform_pca(sdata_filtered, gene_list_expression)

# Gráfico de PCA
plt.figure(figsize=(8, 6))
sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], alpha=0.7, color='red', edgecolor=None)

# Título e rótulos
plt.title(f'PCA - PC1: {pca_var[0]*100:.2f}% | PC2: {pca_var[1]*100:.2f}%', fontsize=16)
plt.xlabel(f'PC1 ({pca_var[0]*100:.2f}%)', fontsize=14)
plt.ylabel(f'PC2 ({pca_var[1]*100:.2f}%)', fontsize=14)

# Exibição do gráfico
plt.grid(True)
plt.show()

# In[2.1]: Finalizei as análises exploratórias. Verificamos até o momento que existe um padrão 
# de distribuição de 'total counts' da expressão gênica nos diferentes datasets, com distribuição
#  de poisson, conjuntamente com um PCA em forma de V. Agora vamos iniciar os testes de similaridade entre os datasets

# Datasets
sdata = sc.read_h5ad("/Users/barbaradalmaso/Desktop/AMD-RPE-spatial/Dados/filtered_spatialdata.h5ad")  # Arquivo de dados espaciais
bulkdata = sc.read_h5ad("/Users/barbaradalmaso/Desktop/AMD-RPE-spatial/Dados/combined_patient_sc.h5ad") # Arquivo de dados scRNA-seq

# Calcular a média de expressão para cada gene nos datasets 
sdata_mean = sdata.X.mean(axis=0)
bulkdata_mean = bulkdata.X.mean(axis=0)
mean_expression = pd.DataFrame({
    "gene": sdata.var.index,
    "spatial_mean": np.array(sdata_mean).flatten(),
    "bulk_mean": np.array(bulkdata_mean).flatten()
})

# Calcular o coeficiente de correlação de Pearson
corr, p_value = pearsonr(mean_expression["spatial_mean"], mean_expression["bulk_mean"])
print(f"Correlação de Pearson: {corr:.2f} (p-valor: {p_value:.2e})")

# Plotar os dados
plt.figure(figsize=(8, 6))
plt.scatter(mean_expression["spatial_mean"], mean_expression["bulk_mean"], alpha=0.7, edgecolor='k')
plt.title(f"Person's Correlation': {corr:.2f}", fontsize=20)
plt.xlabel("Mean Expression - Spatial Data", fontsize=18)
plt.ylabel("Mean Expression - Single cell Data", fontsize=18)
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Observei uma correlação muito baixa entre a expressão global de genes
# Agora vou fzer t-SNE
# 1. Combinar os dados
sdata_adata = sdata.to_df()  # Converte AnnData para DataFrame
bulkdata_adata = bulkdata.to_df()

# Criar labels para identificação do dataset de origem
sdata_adata['dataset'] = 'sdata'
bulkdata_adata['dataset'] = 'bulkdata'

# Combinar os datasets
combined_data = pd.concat([sdata_adata, bulkdata_adata], axis=0)

# Separar os dados de expressão e labels
expression_matrix = combined_data.drop(columns=['dataset'])
labels = combined_data['dataset']

# 2. Criar AnnData para análise
combined_adata = sc.AnnData(expression_matrix)
combined_adata.obs['dataset'] = labels

# 3. Pré-processamento
sc.pp.normalize_total(combined_adata, target_sum=1e4)
sc.pp.log1p(combined_adata)
sc.pp.scale(combined_adata)

# 4. PCA para redução de dimensionalidade
sc.tl.pca(combined_adata)

# 5. t-SNE
sc.pp.neighbors(combined_adata, n_neighbors=5, n_pcs=10)
sc.tl.tsne(combined_adata)

# 6. Plotar t-SNE com cores baseadas no dataset
sc.pl.tsne(combined_adata, color='dataset', title='t-SNE Geral - Datasets')

# 7. Plotar PCA
sc.pl.pca(combined_adata, color='dataset', title='PCA Geral - Datasets')