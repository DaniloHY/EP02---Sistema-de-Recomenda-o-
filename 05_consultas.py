"""Consultas no grafo. Uso: python 05_consultas.py [--user 123] [--genre Comedy] [--min-rating 4]"""
import argparse
import recsys as R

Q_CONTAGEM = """
MATCH (u:User {userId: $uid})-[r:RATED]->(m:Movie)
WHERE r.rating >= $minr AND $genre IN m.genres
RETURN count(DISTINCT m) AS quantidade
"""

Q_EXEMPLOS = """
MATCH (u:User {userId: $uid})-[r:RATED]->(m:Movie)
WHERE r.rating >= $minr AND $genre IN m.genres
RETURN m.title AS filme, r.rating AS nota ORDER BY nota DESC, filme LIMIT 10
"""

Q_POR_GENERO = """
MATCH (u:User {userId: $uid})-[r:RATED]->(m:Movie)
WHERE r.rating >= $minr
UNWIND m.genres AS genero
RETURN genero, count(*) AS filmes ORDER BY filmes DESC
"""

# Top recomendações: KNN item-item em Cypher. Para cada filme i não visto, pega os filmes j que
# o usuário avaliou (treino) ligados a i por SIMILAR_TO, fica com os K de maior pearson e
# calcula Σ c·r / Σ |c|.
Q_RECOMENDAR = """
MATCH (u:User {userId: $uid})-[r:RATED {split: 'train'}]->(j:Movie)
MATCH (i:Movie)-[s:SIMILAR_TO]->(j)
WHERE s.pearson > 0 AND NOT (u)-[:RATED {split: 'train'}]->(i)
WITH i, s.pearson AS c, r.rating AS nota
ORDER BY c DESC
WITH i, collect({c: c, r: nota})[0..$k] AS nb
WITH i, nb,
     reduce(num = 0.0, x IN nb | num + x.c * x.r) AS num,
     reduce(den = 0.0, x IN nb | den + abs(x.c)) AS den
WHERE den > 0
RETURN i.title AS filme, round(num / den * 100) / 100 AS nota_prevista, size(nb) AS vizinhos
ORDER BY nota_prevista DESC, vizinhos DESC LIMIT $top
"""

Q_SIMILARES = """
MATCH (m:Movie {movieId: $mid})-[s:SIMILAR_TO]->(o:Movie)
RETURN o.title AS filme, round(s.pearson * 1000) / 1000 AS pearson, s.n_common AS em_comum
ORDER BY s.pearson DESC LIMIT 10
"""


def mostrar(titulo, registros):
    print(f"\n=== {titulo} ===")
    registros = list(registros)
    if not registros:
        print("(sem resultados)")
    for r in registros:
        print("  " + " | ".join(f"{k}: {v}" for k, v in r.data().items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", type=int, help="padrão: usuário com mais avaliações")
    ap.add_argument("--genre", default="Comedy")
    ap.add_argument("--min-rating", type=float, default=4.0)
    args = ap.parse_args()

    with R.conectar() as driver, R.sessao(driver) as s:
        uid = args.user
        if uid is None:
            uid = s.run("MATCH (u:User)-[r:RATED]->() RETURN u.userId AS id, count(r) AS n "
                        "ORDER BY n DESC LIMIT 1").single()["id"]
        p = dict(uid=uid, genre=args.genre, minr=args.min_rating)
        print(f"Usuário analisado: {uid}")

        mostrar(f"Filmes de {args.genre} com nota >= {args.min_rating} avaliados pelo usuário {uid}",
                s.run(Q_CONTAGEM, **p))
        mostrar("Exemplos", s.run(Q_EXEMPLOS, **p))
        mostrar(f"Filmes com nota >= {args.min_rating} por gênero", s.run(Q_POR_GENERO, uid=uid, minr=args.min_rating))
        mostrar(f"TOP {R.TOP_N} recomendações para o usuário {uid}",
                s.run(Q_RECOMENDAR, uid=uid, k=R.K, top=R.TOP_N))

        mais_visto = s.run("MATCH (m:Movie)<-[r:RATED]-() RETURN m.movieId AS id, m.title AS t, count(r) AS n "
                           "ORDER BY n DESC LIMIT 1").single()
        mostrar(f"Filmes mais similares a '{mais_visto['t']}'", s.run(Q_SIMILARES, mid=mais_visto["id"]))


if __name__ == "__main__":
    main()
