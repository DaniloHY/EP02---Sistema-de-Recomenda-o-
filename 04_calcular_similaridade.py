"""Calcula o Pearson entre filmes (SÓ com o treino) e grava no grafo como
(:Movie)-[:SIMILAR_TO {pearson, n_common}]->(:Movie). Grava também a média de treino em cada Movie."""
import recsys as R

LOTE = 5000


def main():
    with R.conectar() as driver, R.sessao(driver) as s:
        print("Lendo avaliações de treino do Neo4j...")
        treino = R.ler_avaliacoes(s, "train")
        print(f"  {len(treino)} usuários. Calculando Pearson item-item (pode demorar)...")
        vizinhos = R.pearson_vizinhos(treino)
        medias, qtd = R.medias_filmes(treino)

        filmes = [{"movieId": m, "avg": medias[m], "n": qtd[m]} for m in medias]
        for i in range(0, len(filmes), LOTE):
            s.run("UNWIND $rows AS row MATCH (m:Movie {movieId: row.movieId}) "
                  "SET m.avgTrainRating = row.avg, m.nTrainRatings = row.n", rows=filmes[i:i + LOTE])

        while s.run("MATCH ()-[r:SIMILAR_TO]->() WITH r LIMIT 10000 DELETE r RETURN count(*) AS c").single()["c"]:
            pass
        arestas = [{"i": i, "j": j, "c": c, "n": n} for i, lst in vizinhos.items() for j, c, n in lst]
        print(f"Gravando {len(arestas)} arestas SIMILAR_TO...")
        for i in range(0, len(arestas), LOTE):
            s.run("UNWIND $rows AS row "
                  "MATCH (a:Movie {movieId: row.i}) MATCH (b:Movie {movieId: row.j}) "
                  "CREATE (a)-[:SIMILAR_TO {pearson: row.c, n_common: row.n}]->(b)",
                  rows=arestas[i:i + LOTE])
    print("Pronto.")


if __name__ == "__main__":
    main()
