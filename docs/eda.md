# CN15k EDA

## Step 1: Load and merge data

Use Dataset from https://www.kaggle.com/datasets/thala321/cn15k-dataset , which will have the mapping of entities, and relations with their ids

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

![image](https://github.com/hughiephan/UKGE/assets/16631121/35706866-3f04-4c12-9f31-dafa10169e19)

## Step 2: Visualize all types of Relationship
```python
relation_string_array = df['relation_string'].unique()
for relation_string in relation_string_array:
    new_df = df.loc[df['relation_string'] == relation_string].head(2)
    G = nx.from_pandas_edgelist(new_df, 'entity_string_1', 'entity_string_2', edge_attr='relation_string', create_using=nx.DiGraph())
    plt.figure(figsize=(4, 3))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color='skyblue', font_weight='bold', font_size=12, arrows=True)
    edge_labels = {(u, v): d['relation_string'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    plt.title('Relation: ' + relation_string)
    plt.show()
```

![image](https://github.com/hughiephan/UKGE/assets/16631121/e56c3399-e7ea-42f3-8681-530b7153961c)

## Step 3: Visualize all Relationship of a Entity
```python
new_entity = 'dog'
new_df = df.loc[df['entity_string_2'] == new_entity].drop_duplicates(subset=['relation_string'])
G = nx.from_pandas_edgelist(new_df, 'entity_string_1', 'entity_string_2', edge_attr='relation_string', create_using=nx.DiGraph())
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=1000, node_color='skyblue', font_weight='bold', font_size=12, arrows=True)
edge_labels = {(u, v): d['relation_string'] for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
plt.title('Entity: ' + new_entity)
plt.show()
```

![image](https://github.com/hughiephan/UKGE/assets/16631121/afe2bc04-d0da-4d7b-a86d-e20564ca0e11)
