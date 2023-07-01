# Fish type clustering
:::

::: {.cell .markdown}
![fish](vertopal_4dc51548a10648d59954fb1aee28ed32/c1118358d5ea69d4a3e01abdcafdd5709c0cecd8.jpg)
:::

::: {.cell .markdown id="XIigj48HiaNC"}
Source: <http://jse.amstat.org/jse_data_archive.html>
:::

::: {.cell .markdown id="cyJ4eQ_zpQQL"}
## Import modules
:::

::: {.cell .code execution_count="10" id="CtHQgBnZhhRf"}
``` python
# Perform the necessary imports
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
# Import Normalizer
from sklearn.preprocessing import Normalizer
# Import pandas
import pandas as pd

# Import PCA
from sklearn.decomposition import PCA
```
:::

::: {.cell .markdown id="oH8dFdbXpTNj"}
## Dataset
:::

::: {.cell .code execution_count="11" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="wWZRjN6Ui15Q" outputId="f9a1008c-c281-4370-ebfb-c1d6357e73c1"}
``` python
# Read Dataset as csv file
fish_df = pd.read_csv('fish.csv')

fish_df.columns = ['species', 'weight', 'length1', 'length2', 'length3', 'Height%','Width%']

# print Dataset
print(fish_df.head())
print(fish_df['species'].unique())

X_fish_df = fish_df.drop('species', axis=1)
species = fish_df['species']

samples = X_fish_df.to_numpy()
```

::: {.output .stream .stdout}
      species  weight  length1  length2  length3  Height%  Width%
    0   Bream   290.0     24.0     26.3     31.2     40.0    13.8
    1   Bream   340.0     23.9     26.5     31.1     39.8    15.1
    2   Bream   363.0     26.3     29.0     33.5     38.0    13.3
    3   Bream   430.0     26.5     29.0     34.0     36.6    15.1
    4   Bream   450.0     26.8     29.7     34.7     39.2    14.2
    ['Bream' 'Roach' 'Smelt' 'Pike']
:::
:::

::: {.cell .markdown id="q9DG3ERMiOaQ"}
## Scaling
:::

::: {.cell .code execution_count="12" id="ssENFlkmiIp9"}
``` python
# Create scaler: scaler
scaler = StandardScaler()

# Create KMeans instance: kmeans
kmeans = KMeans(n_clusters=4)

# Create pipeline: pipeline
pipeline = make_pipeline(scaler, kmeans)
```
:::

::: {.cell .markdown id="dnBxNyuJpMjm"}
## Clustering
:::

::: {.cell .code execution_count="13" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="BZp_lZ1YiSFA" outputId="6315c9d1-6981-4790-be2c-d3d353f5c967"}
``` python
# Fit the pipeline to samples
pipeline.fit(samples)

# Calculate the cluster labels: labels
labels = pipeline.predict(samples)

# Create a DataFrame with labels and species as columns: df
df = pd.DataFrame({'labels': labels, 'species': species})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['species'])

# Display ct
print(ct)
```

::: {.output .stream .stdout}
    species  Bream  Pike  Roach  Smelt
    labels                            
    0            0    17      0      0
    1           33     0      1      0
    2            0     0      0     13
    3            0     0     19      1
:::

::: {.output .stream .stderr}
    /usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
:::
:::

::: {.cell .markdown id="u3i9TjkIEx96"}
## Dimension reduction
:::

::: {.cell .code execution_count="14" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="iJpigP21Eo_D" outputId="dae8e72d-9b31-495a-b5d0-517df9899950"}
``` python
scaler = StandardScaler()
scaled_samples = scaler.fit_transform(samples)

# Create a PCA model with 2 components: pca
pca = PCA(n_components=2)

# Fit the PCA instance to the scaled samples
pca.fit(scaled_samples)

# Transform the scaled samples: pca_features
pca_features = pca.transform(scaled_samples)

# Print the shape of pca_features
print(pca_features.shape)
```

::: {.output .stream .stdout}
    (84, 2)
:::
:::

::: {.cell .markdown id="pikKCy1Mt0Cd"}
## Result

The fish data separates really well into 4 clusters!
:::

::: {.cell .code id="1mz-ExoTiti2"}
``` python
```
:::
