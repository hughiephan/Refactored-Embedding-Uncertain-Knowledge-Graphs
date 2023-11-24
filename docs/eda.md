# CN15k EDA

## Step: 
```python
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('/kaggle/input/cn15k-dataset/data.tsv', sep='\t', header=None, names=['entity_1','relation','entity_2','weight'])
entity = pd.read_csv('/kaggle/input/cn15k-dataset/entity_id.csv').rename(columns={'entity string': 'entity_string'})
relation = pd.read_csv('/kaggle/input/cn15k-dataset/relation_id.csv').rename(columns={'relation string': 'relation_string'})

df = pd.merge(data, entity, left_on='entity_1', right_on='id', how='inner').rename(columns={'entity_string': 'entity_string_1'})
df = pd.merge(df, entity, left_on='entity_2', right_on='id', how='inner').rename(columns={'entity_string': 'entity_string_2'})
df = pd.merge(df, relation, left_on='relation', right_on='id', how='inner')
df = df[['entity_string_1', 'relation_string', 'entity_string_2', 'weight']]
print(df)
```

## Step:
```python
relation_string_array = df['relation_string'].unique()
for relation_string in relation_string_array:
    new_df = df.loc[df['relation_string'] == relation_string].head(2)
    G = nx.from_pandas_edgelist(new_df, 'entity_string_1', 'entity_string_2', edge_attr='relation_string', create_using=nx.DiGraph())
    plt.figure(figsize=(4, 3))
    pos = nx.spring_layout(G)  # Position nodes using Fruchterman-Reingold force-directed algorithm
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color='skyblue', font_weight='bold', font_size=12, arrows=True)
    edge_labels = {(u, v): d['relation_string'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    plt.title('Relation: ' + relation_string)
    plt.show()
```

## Step:
```python
new_entity = 'dog'
new_df = df.loc[df['entity_string_2'] == new_entity].drop_duplicates(subset=['relation_string'])
G = nx.from_pandas_edgelist(new_df, 'entity_string_1', 'entity_string_2', edge_attr='relation_string', create_using=nx.DiGraph())
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G)  # Position nodes using Fruchterman-Reingold force-directed algorithm
nx.draw(G, pos, with_labels=True, node_size=1000, node_color='skyblue', font_weight='bold', font_size=12, arrows=True)
edge_labels = {(u, v): d['relation_string'] for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
plt.title('Entity: ' + new_entity)
plt.show()
```
