# MovieLens 20M: grafo no Neo4j + recomendação KNN item-item (Pearson)

Usa só `ratings.csv` e `movies.csv`. Única dependência externa: o driver `neo4j`.
Pearson, KNN, split e métricas usam apenas a biblioteca padrão do Python.

## 1. Modelo do grafo

| Elemento | Propriedades |
|---|---|
| Nó `(:User)` | `userId` |
| Nó `(:Movie)` | `movieId`, `title`, `genres` (lista), `avgTrainRating`, `nTrainRatings` |
| Aresta `(:User)-[:RATED]->(:Movie)` | `rating`, `timestamp`, `split` (`train` ou `test`) |
| Aresta `(:Movie)-[:SIMILAR_TO]->(:Movie)` | `pearson`, `n_common` (usuários em comum) |

`SIMILAR_TO` guarda os 50 vizinhos de maior correlação de cada filme (calculada só com o treino).

## 2. Como rodar

```bash
pip install -r requirements.txt
# defina NEO4J_PASSWORD (veja .env.example), ex.: export NEO4J_PASSWORD=sua_senha
# Neo4j local: docker run -d -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/sua_senha neo4j:5

python 01_download_dataset.py        # baixa ratings.csv e movies.csv em data/
python 02_preparar_dados.py          # amostra + split 80/20
python 03_carregar_grafo.py --reset  # insere nós e arestas no Neo4j
python 04_calcular_similaridade.py   # Pearson -> arestas SIMILAR_TO
python 05_consultas.py --user 123    # consultas + Top recomendações
python 06_avaliar.py                 # Precisão e Recall
```

Se já baixou o dataset do Kaggle, copie `ratings.csv` e `movies.csv` para `data/` e pule o passo 1.

**Amostra:** o Pearson sobre os 20M é inviável em Python puro (o número de pares cresce com o quadrado
das avaliações por usuário). Por padrão usa os 1000 filmes mais avaliados e 3000 usuários com ≥ 20 avaliações
nesses filmes. Mude por variáveis de ambiente (`MAX_FILMES`, `MAX_USUARIOS`, ...; ver topo de `recsys.py`).
Os resultados valem para essa amostra.

## 3. Fórmulas

**Pearson** entre filmes i e j, com k nos usuários que avaliaram os dois e r̄ a média de treino de cada filme:
`c(i,j) = Σk (r_i^k − r̄_i)(r_j^k − r̄_j) / (√Σk (r_i^k − r̄_i)² · √Σk (r_j^k − r̄_j)²)`.
Só vira aresta se houver ≥ 5 usuários em comum (com 1 ou 2 saem correlações ±1 sem significado).

**KNN:** `r̂(u,i) = Σj c(i,j)·r(u,j) / Σj |c(i,j)|`, com j nos K=20 filmes mais similares a i (c > 0) que o usuário u avaliou.
Obs.: no enunciado o numerador aparece como `r_i^k`; o correto é a nota que o usuário u deu ao vizinho j, `r(u,j)`.
Sem vizinhos, usa a média do filme (depois a média global).

## 4. Consultas (`05_consultas.py`)
- Comédias que o usuário X avaliou com nota ≥ 4: `MATCH (u:User {userId:$uid})-[r:RATED]->(m:Movie) WHERE r.rating >= 4 AND 'Comedy' IN m.genres RETURN count(DISTINCT m)`
- Exemplos, contagem por gênero, **Top recomendações para o usuário X** (KNN em Cypher) e filmes similares.

## 5. Avaliação (`06_avaliar.py`)
- **[A] Top-N:** relevantes = filmes do teste com nota ≥ 4. Precisão@N = acertos/N; Recall@N = acertos/|relevantes|. Compara com baseline de popularidade.
- **[B] Pares do teste:** positivo = nota ≥ 4. Precisão = TP/(TP+FP), Recall = TP/(TP+FN), mais RMSE e MAE.

Em dados esparsos a precisão do Top-N tende a ser baixa, porque um filme bom que o usuário nunca avaliou conta como erro; compare com o baseline.

## Arquivos
`recsys.py` (config, conexão, Pearson, KNN, métricas) · `01`..`06` (um passo cada) · `.gitignore` · `.env.example`
