# CN15k EDA

## Step 1: Load and merge data

Use Dataset from https://www.kaggle.com/datasets/thala321/cn15k-dataset , which will have the mapping of entities, and relations with their ids

```python
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

## Step 1

`data.head()` will show only the first few values.

```python
for index1, row1 in data.head().iterrows():
    for index2, row2 in data.head().iterrows():
        for index3, row3 in data.head().iterrows():
            number = np.maximum(1 - row1['weight'] * row2['weight'], row3['weight'])
            print(number)
```

## Step 2: 
EDA, Means 0.7 pick relations beyond certain threashold, 
```
pandas sort the values for all the relations (mean, algorithm)
R14 ~ R12: 0.8
R13 ~ R10. 0.7
.....
R1: 0.2 (remove this one)
```

## Calculate all relations
```python
weights_rs0 = df[df['relation'] == 0]['weight']
weights_rs2 = df[df['relation'] == 2]['weight']

unique_relations = sorted(df['relation'].unique())
for relation in unique_relations:
    # Get all 35 relations and it's mean weight
    relation_df = df[df['relation'] == relation]
    mean_weight = relation_df['weight'].mean()
    print("-------")
    print("Testing with relation", relation)
    print("Relation mean weight: ", mean_weight)
    
    # Calculating: Weight 0 and Weight 1 with all the 35 Relation Mean Weights
    sum = 0
    count = 0
    for weight_0 in weights_rs0: 
        for weight_2 in weights_rs2: 
            number = np.maximum(1 - weight_0 * weight_2, mean_weight)
            count +=1
            sum +=number
        if (count % 100000 == 0): # Should take 30 seconds 
            print("Calculating...")
    print("Sum/count:", sum/count)
```

## Making a dictionary
```python
# Loop through every rows of DF
for index, row in df.head().iterrows():
  # Get R1T1
  r1t1 = str(row["relation"])+row["entity_string_2"]
  print(r1t1)
  x = row["entity_string_1"]
  ourDict[r1t1] = [{"name": x, "weight": row["weight"]}] # Append function in array

print(ourDict)
```

## Example

```
Today is cloudy - key1: is cloudy
Today use umbrella - key2: use umbrella
Today is rainy - key3: is rainy
Today relatedTo tomorrow - key4: relatedTo tomorrow
Today is placeholder X

# Algorithm
1th
Today is cloudy    -   key1: is cloudy
Today is cloudy    -   key1: is cloudy
Today is cloudy    -   key1: is cloudy
each key triplet, the weights

2nd
Today is cloudy    -   key1: is cloudy
Today is cloudy    -   key1: is cloudy
Today use umbrella  -   key2: use umbrella
save triplet

3rd
Today is cloudy    -   key1: is cloudy
Today is cloudy    -   key1: is cloudy
Today is rainy        -     key3: is rainy
save triplet

.....

n-th
Today relatedTo tomorrow       -   key4: relatedTo tomorrow
Today relatedTo tomorrow       -   key4: relatedTo tomorrow
Today relatedTo tomorrow       -   key4: relatedTo tomorrow



Key triplet1: (key1, key2, key3) can appear many times and with have different weights each time

Today is cloudy    -   key1: is cloudy
Today use umbrella  -   key2: use umbrella
Today is rainy        -     key3: is rainy

-------------
Key triplet2: (key1, key2, key3)
Tomorrow is cloudy    -   key1: is cloudy
Tomorrow use umbrella  -   key2: use umbrella
Tomorrow is rainy        -     key3: is rainy

Dictionary
Key1Key2Key3:    Array( (today,weight of today), (tomorrow, weight of tomorrow))



Array (have a check if placeholder is the same)
(key1, key2, key3, max_weight)
(key3, key2, key3, max_weight)

Dictionary
key1key2key3 =  (key1, key2, key3, max_weight)
key3key4key5 =  (key3, key4, key5, max_weight)
```

## Creating Key Triplet

```python
key_triplet_array = []
sum = 0
count = 0
for key_1, values_1 in ourDict.items():
    for key_2, values_2 in ourDict.items():
        for key_3, values_3 in ourDict.items():
            
            weight_1 = values_1[0][1]
            weight_2 = values_2[0][1]
            weight_3 = values_3[0][1]

            head_1 = values_1[0][0]
            head_2 = values_2[0][0]
            head_3 = values_3[0][0]
            
            if head_1 == head_2 == head_3 and key_1 != key_2 and key_2 != key_3:
                max_weight = max(1 - weight_1 * weight_2, weight_3)
                key_triplet = (key_1, key_2, key_3, max_weight)
                key_triplet_array.append(key_triplet)   # Todo: Update this into dictionary format with Keytriplet: Array( (today,weight of today), (tomorrow, weight of tomorrow))
                count = count + 1
                if (count % 100 == 0):
                    print(key_triplet)

# Now loop through the key_triplet_dictionary and check if the key triplets have the same key_2 and key_3. If this happens, we will calculate the average of that.
# score[key_2, key_3] = [weight1, weight2,...., weight1000] 
```
