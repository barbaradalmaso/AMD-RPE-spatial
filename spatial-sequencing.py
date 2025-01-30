# In[0.1]: Importar pacotes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import os
import seaborn as sns
from scipy.sparse import csr_matrix
import gseapy as gp
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


os.chdir("/Users/barbaradalmaso/Desktop/AMD-RPE-spatial/Dados/")

# In[0.2]: Fazer pré-processamento dos dados
# This file contains the raw MERFISH count matrix for six samples with gene features. The "sampleid" column represents the unique sample ID, while the "region" column corresponds to the tissue section ID. The "majorclass" and "subclass" columns indicate annotated retinal cell types. Finally, the "center_x" and "center_y" columns provide the coordinates of the cell centers.
sdata = sc.read_h5ad("merfish_impute.h5ad") # https://zenodo.org/records/8144355 

# Verificar se regiões com enriquecimento de fatores de crescimento, principalmente FGF, também possuem enriquecimento de vias de liberação de lisossomos
# Mostrar expressão genica média das regiões, e depois fazer um teste de regressão múltipla (pra ver correlação?) ou fazer uma matrix de correlação

# Filtrar as células com base no total de contagens de forma otimizada
if isinstance(sdata.X, csr_matrix):
    sdata.obs['total_counts'] = np.array(sdata.X.sum(axis=1)).flatten()
else:
    sdata.obs['total_counts'] = sdata.X.sum(axis=1)
# Filtrar células com total_count > 1000
sdata_filtered = sdata[(sdata.obs['total_counts'] > 1000) & (sdata.obs['total_counts'] < 10000),:]
# Exibir informações sobre a filtragem
print(f"Dimensão original de sdata: {sdata.shape}")
print(f"Dimensão após a filtragem de genes e total_counts > 500: {sdata_filtered.shape}")

##################### Processamento de dados de spatial scRNA-seq de retina saudável #############################
# Criar DataFrame da matriz de expressão
expression_df = pd.DataFrame(
    sdata_filtered.X.toarray() if hasattr(sdata_filtered.X, "toarray") else sdata_filtered.X,
    index=sdata_filtered.obs.index,  # Índice das células
    columns=sdata_filtered.var.index  # Nomes dos genes
)

expression_df = pd.concat([sdata.obs, expression_df], axis=1) # Integrar df com metadados

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

# Salvar o objeto combinado em um arquivo .h5ad
output_path = "/Users/barbaradalmaso/Desktop/AMD-RPE-spatial/Dados/filtered_spatialdata.h5ad"
sdata_filtered.write(output_path)

# In[1.0]: A partir do dataset ja filtrado, irei fazer análise de regiões do tecido enriquecidas com fatores de crescimento e de genes de lisossomos
sdata = sc.read_h5ad("filtered_spatialdata.h5ad") # https://zenodo.org/records/8144355 

# Transformar 'sample_id' e 'region' como fatores
sdata.obs['sampleid'] = sdata.obs['sampleid'].astype('category')
sdata.obs['region'] = sdata.obs['region'].astype('category')

# Separar os genes de interesse de meu estudo:
# Extração de genes relacionados com liberação de lysossomos
lysosome_geneset = gp.get_library(name='GO_Cellular_Component_2023', organism='Mouse')
lysosome_geneset = lysosome_geneset["Lysosome (GO:0005764)"]
lysosome_geneset = pd.DataFrame({'human_genes': lysosome_geneset})
# Extração de genes relacionados com ativação de fatores de crescimento
fgf_geneset = gp.get_library(name='GO_Biological_Process_2023', organism='Mouse')
fgf_geneset = fgf_geneset["Fibroblast Growth Factor Receptor Signaling Pathway (GO:0008543)"]
fgf_geneset = pd.DataFrame({'human_genes': fgf_geneset})
# Converter nomenclatura de genes de humano para mouse
lysosome_geneset['human_genes'] = lysosome_geneset['human_genes'].str.capitalize()
lysosome_geneset = lysosome_geneset['human_genes'].tolist()
fgf_geneset['human_genes'] = fgf_geneset['human_genes'].str.capitalize()
fgf_geneset = fgf_geneset['human_genes'].tolist()

# Fazer o cálculo de soma de expressão de genes de lisossomos e de genes de FGF para verificar enriquecimento de regiões
# Inicializar um dicionário para armazenar os resultados por combinação de sampleid e region
results = {}

# Iterar sobre todas as combinações únicas de sampleid e region
for sampleid in sdata.obs['sampleid'].unique():
    for region in sdata.obs['region'].unique():
        # Filtrar os dados para a combinação atual de sampleid e region
        filtered_df = sdata.obs[(sdata.obs['sampleid'] == sampleid) & (sdata.obs['region'] == region)]
        
        # Ignorar combinações vazias
        if filtered_df.empty:
            continue
        
        # Obter a matriz de expressão dos genes na região filtrada
        expression_data = sdata[filtered_df.index].to_df()
        
        # Selecionar apenas os genes de lisossomos que estão nas colunas de expression_data
        valid_lysosome_genes = [gene for gene in lysosome_geneset if gene in expression_data.columns]
        lysosome_expression_data = expression_data[valid_lysosome_genes]
        
        # Calcular a soma total de expressão de genes de lisossomos
        lysosome_expression_data['lysosome_expression_sum'] = lysosome_expression_data.sum(axis=1)
        
        # Selecionar apenas os genes de FGFR que estão nas colunas de expression_data
        valid_fgf_genes = [gene for gene in fgf_geneset if gene in expression_data.columns]
        fgf_expression_data = expression_data[valid_fgf_genes]
        
        # Calcular a soma total de expressão dos genes de FGFR
        fgf_expression_data['fgf_expression_sum'] = fgf_expression_data.sum(axis=1)
        
        # Adicionar os resultados de lisossomos e FGFR em filtered_df
        valid_cells = filtered_df.index.intersection(lysosome_expression_data.index)
        filtered_df['lysosome_expression_sum'] = None  # Inicializar com valores nulos
        filtered_df.loc[valid_cells, 'lysosome_expression_sum'] = lysosome_expression_data.loc[valid_cells, 'lysosome_expression_sum']
        
        valid_cells_fgf = filtered_df.index.intersection(fgf_expression_data.index)
        filtered_df['fgf_expression_sum'] = None  # Inicializar com valores nulos
        filtered_df.loc[valid_cells_fgf, 'fgf_expression_sum'] = fgf_expression_data.loc[valid_cells_fgf, 'fgf_expression_sum']
        
        # Salvar o resultado no dicionário
        results[(sampleid, region)] = filtered_df
        
        
        # Iterar sobre os resultados para exibir os valores de 'lysosome_expression_sum' e 'fgf_expression_sum'
        for (sampleid, region), filtered_df in results.items():
            print(f"Resultados para {sampleid} - Região {region}:")
            print(filtered_df[['lysosome_expression_sum', 'fgf_expression_sum']])
            print("\n" + "="*50 + "\n")

# Gerar os gráficos para cada combinação de sampleid e region
for (sampleid, region), filtered_df in results.items():
    # Certificar-se de que as colunas sejam numéricas e substituir NaN/None por zero
    filtered_df['lysosome_expression_sum'] = pd.to_numeric(filtered_df['lysosome_expression_sum'], errors='coerce').fillna(0)
    filtered_df['fgf_expression_sum'] = pd.to_numeric(filtered_df['fgf_expression_sum'], errors='coerce').fillna(0)

    # Criar uma figura com dois subplots lado a lado
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=300)

    # Plot para lysosome_expression_sum
    scatter_lysosome = axes[0].scatter(
        x=filtered_df['center_x'],
        y=filtered_df['center_y'],
        c=filtered_df['lysosome_expression_sum'],
        cmap='coolwarm',
        s=30,
        alpha=0.8
    )
    axes[0].set_title(f'{sampleid} - Região {region} (Lysosome)', fontsize=16)
    axes[0].set_xlabel('Coordenada X', fontsize=12)
    axes[0].set_ylabel('Coordenada Y', fontsize=12)
    axes[0].invert_yaxis()
    cbar = plt.colorbar(scatter_lysosome, ax=axes[0], label='Expressão de Lysossomos', pad=0.02)
    cbar.ax.tick_params(labelsize=10)

    # Plot para fgf_expression_sum
    scatter_fgf = axes[1].scatter(
        x=filtered_df['center_x'],
        y=filtered_df['center_y'],
        c=filtered_df['fgf_expression_sum'],
        cmap='cividis',
        s=30,
        alpha=0.8
    )
    axes[1].set_title(f'{sampleid} - Região {region} (FGF)', fontsize=16)
    axes[1].set_xlabel('Coordenada X', fontsize=12)
    axes[1].set_ylabel('Coordenada Y', fontsize=12)
    axes[1].invert_yaxis()
    cbar = plt.colorbar(scatter_fgf, ax=axes[1], label='Expressão de FGF', pad=0.02)
    cbar.ax.tick_params(labelsize=10)

    # Ajustar layout e exibir
    plt.tight_layout()
    plt.show()
    
# Após a visualização, irei fazer uma análise de correlação
# Inicializar listas para armazenar os valores de lysosome_expression_sum e fgf_expression_sum
lysosome_values = []
fgf_values = []

# Iterar sobre os resultados coletados no dicionário 'results'
for (sampleid, region), filtered_df in results.items():
    # Verificar se há valores para ambas as colunas
    if 'lysosome_expression_sum' in filtered_df.columns and 'fgf_expression_sum' in filtered_df.columns:
        lysosome_values.extend(filtered_df['lysosome_expression_sum'].dropna().tolist())
        fgf_values.extend(filtered_df['fgf_expression_sum'].dropna().tolist())

# Calcular a correlação de Pearson e o valor de p
correlation, p_value = pearsonr(lysosome_values, fgf_values)

sns.set_style("ticks")
plt.figure(figsize=(8,6), dpi=300) 
sns.scatterplot(x=lysosome_values, y=fgf_values, s=100, alpha=0.5, edgecolor=None, color = "#426b65")

# Adicionar linha de tendência (se houver uma relação linear)
sns.regplot(x=lysosome_values, y=fgf_values, scatter=False, color='black')

# Adicionar títulos e rótulos
plt.title(
    ' \n'
    ' ',
    fontsize=18)
plt.xlabel('Lysosome (GO:0005764) genes expression',fontsize=18)
plt.ylabel('Fibroblast Growth Factor Receptor Signaling\n'
           'Pathway (GO:0008543) genes expression', fontsize=18)

# Exibir o gráfico
plt.show()

# In[2.0]: Criar um algoritmo de clusterização para separar regiões com alto e baixo Liso/FGF
# Concatenar results em um único DF
df_combined = pd.concat(results.values(), ignore_index=True)

# Selecionar apenas as colunas de interesse em minha base de dados
df_cluster = df_combined[['lysosome_expression_sum', 'fgf_expression_sum', 'total_counts']].copy()

# Normalizar os dados para evitar diferenças de escala
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_cluster)

# Aplicar K-means com 2 clusters
kmeans = KMeans(n_clusters=3, random_state=123, n_init=10)
df_cluster['Cluster_Kmeans'] = kmeans.fit_predict(df_scaled)

# Visualizar os clusters
# Criar figura 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Definir coordenadas para cada eixo
x = df_cluster['lysosome_expression_sum']
y = df_cluster['fgf_expression_sum']
z = df_cluster['total_counts'] 

# Criar scatterplot 3D
scatter = ax.scatter(x, y, z, 
                     c=df_cluster['Cluster_Kmeans'], 
                     cmap='viridis', 
                     s=30, alpha=0.8)  # s ajusta o tamanho das bolinhas

# Rótulos dos eixos
ax.set_xlabel('', fontsize=12)
ax.set_ylabel('', fontsize=12)
ax.set_zlabel('', fontsize=12)

# Adicionar legenda manualmente
legend1 = ax.legend(*scatter.legend_elements(), title="Cluster", fontsize=12)
ax.add_artist(legend1)

plt.show()
# Adicionar os clusters ao dataframe original
df_combined['Cluster_Kmeans'] = df_cluster['Cluster_Kmeans']

sdata.obs['Cluster_Kmeans'] = df_cluster['Cluster_Kmeans'].values
sdata.write("sdata_clustered.h5ad")
print("Arquivo salvo como 'sdata_clustered.h5ad' com sucesso!")

# In[3.0]: Identificar oa DEGS entre os grupos clusterizados, utilizando o novo anndata
sdata = sc.read_h5ad("sdata_clustered.h5ad") # https://zenodo.org/records/8144355 
sdata.obs = sdata.obs[['sampleid', 'majorclass', 'Cluster_Kmeans']]
sdata = sdata[sdata.obs['Cluster_Kmeans'].isin([0, 2])].copy()
sc.pp.log1p(sdata)

# Verificar o resultado
print(sdata.obs.columns)

# Coletar dados de expressão , a partir de dados de matrix densa de sdata
import scipy.sparse as sp
if not isinstance(sdata.X, sp.spmatrix):
    sdata.X = sp.csr_matrix(sdata.X) 

# Rodar análise de DEGs
# Converte 'Cluster_Kmeans' para o tipo categoria
sdata.obs['Cluster_Kmeans'] = sdata.obs['Cluster_Kmeans'].astype('category')
sc.pp.log1p(sdata)
sc.tl.rank_genes_groups(sdata, groupby='Cluster_Kmeans', method='wilcoxon')
sdata.layers["scaled"] = sc.pp.scale(sdata, copy=True).X

# Ver genes
sc.pl.rank_genes_groups(sdata, n_genes=15, sharey=False)
ranked_genes = sdata.uns['rank_genes_groups']
ranked_genes = ranked_genes['names']['2']

# Consegui a lista de genes com expressão diferencial. 
# Agora eu preciso avaliar a expressão pelas células...
