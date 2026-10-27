"""Insere o grafo no Neo4j.

  (:User {userId})
  (:Movie {movieId, title, genres})                      genres = lista de gêneros
  (:User)-[:RATED {rating, timestamp, split}]->(:Movie)  split = 'train' ou 'test'
  (:Movie)-[:SIMILAR_TO {pearson, n_common}]->(:Movie)   criada em 04_calcular_similaridade.py

Uso: python 03_carregar_grafo.py [--reset]"""
import argparse
import csv
import recsys as R

LOTE = 5000


def lotes(itens):
    for i in range(0, len(itens), LOTE):
        yield itens[i:i + LOTE]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="apaga o grafo antes de inserir")
    args = ap.parse_args()

    filmes = []
    with open(R.AMOSTRA / "movies.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gen = [] if row["genres"] == "(no genres listed)" else row["genres"].split("|")
            filmes.append({"movieId": int(row["movieId"]), "title": row["title"], "genres": gen})

    avaliacoes, usuarios = [], set()
    for split in ("train", "test"):
        with open(R.AMOSTRA / f"{split}.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                usuarios.add(int(row["userId"]))
                avaliacoes.append({"userId": int(row["userId"]), "movieId": int(row["movieId"]),
                                   "rating": float(row["rating"]), "timestamp": int(row["timestamp"]),
                                   "split": split})

    with R.conectar() as driver, R.sessao(driver) as s:
        if args.reset:
            print("Limpando o grafo...")
            while s.run("MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(*) AS c").single()["c"]:
                pass
        s.run("CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.userId IS UNIQUE")
        s.run("CREATE CONSTRAINT movie_id IF NOT EXISTS FOR (m:Movie) REQUIRE m.movieId IS UNIQUE")

        print(f"Inserindo {len(filmes)} filmes...")
        for b in lotes(filmes):
            s.run("UNWIND $rows AS row MERGE (m:Movie {movieId: row.movieId}) "
                  "SET m.title = row.title, m.genres = row.genres", rows=b)

        print(f"Inserindo {len(usuarios)} usuários...")
        for b in lotes([{"userId": u} for u in sorted(usuarios)]):
            s.run("UNWIND $rows AS row MERGE (:User {userId: row.userId})", rows=b)

        print(f"Inserindo {len(avaliacoes)} avaliações (RATED)...")
        for b in lotes(avaliacoes):
            s.run("UNWIND $rows AS row "
                  "MATCH (u:User {userId: row.userId}) MATCH (m:Movie {movieId: row.movieId}) "
                  "CREATE (u)-[:RATED {rating: row.rating, timestamp: row.timestamp, split: row.split}]->(m)",
                  rows=b)
    print("Grafo carregado.")


if __name__ == "__main__":
    main()
