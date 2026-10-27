"""Seleciona uma amostra (os 20M de avaliações são inviáveis para Pearson em Python puro)
e divide em 80% treino / 20% teste por usuário. Saída: data/amostra/{train,test,movies}.csv"""
import csv
import random
import time
from calendar import timegm
from collections import Counter, defaultdict
import recsys as R


def timestamp_unix(t):
    """Aceita o formato do GroupLens (número Unix) e o do Kaggle ('2000-08-25 21:57:02', UTC)."""
    try:
        return int(t)
    except ValueError:
        return timegm(time.strptime(t.strip(), "%Y-%m-%d %H:%M:%S"))


def linhas(caminho):
    with open(caminho, newline="", encoding="utf-8") as f:
        leitor = csv.reader(f)
        next(leitor)  # cabeçalho
        yield from leitor


def main():
    R.AMOSTRA.mkdir(parents=True, exist_ok=True)
    ratings = R.DATA / "ratings.csv"

    print("1/3 Escolhendo os filmes mais avaliados...")
    cont_filme = Counter(l[1] for l in linhas(ratings))
    candidatos = sorted(((m, c) for m, c in cont_filme.items() if c >= R.MIN_AVAL_FILME),
                        key=lambda t: (-t[1], int(t[0])))
    filmes = {m for m, _ in candidatos[:R.MAX_FILMES]}

    print("2/3 Escolhendo os usuários...")
    cont_user = Counter(l[0] for l in linhas(ratings) if l[1] in filmes)
    elegiveis = sorted((u for u, c in cont_user.items() if c >= R.MIN_AVAL_USUARIO), key=int)
    rng = random.Random(R.SEED)
    if len(elegiveis) > R.MAX_USUARIOS:
        elegiveis = rng.sample(elegiveis, R.MAX_USUARIOS)
    usuarios = set(elegiveis)

    print("3/3 Dividindo treino/teste...")
    por_user = defaultdict(list)
    for u, m, r, t in linhas(ratings):
        if u in usuarios and m in filmes:
            por_user[u].append((int(u), int(m), float(r), timestamp_unix(t)))
    treino, teste = [], []
    for u in sorted(por_user, key=int):
        rows = sorted(por_user[u], key=lambda x: x[1])
        rng.shuffle(rows)
        n_treino = max(1, round(R.FRACAO_TREINO * len(rows)))
        treino += rows[:n_treino]
        teste += rows[n_treino:]

    for nome, rows in (("train.csv", treino), ("test.csv", teste)):
        with open(R.AMOSTRA / nome, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["userId", "movieId", "rating", "timestamp"])
            w.writerows(rows)

    with open(R.DATA / "movies.csv", newline="", encoding="utf-8") as fin, \
         open(R.AMOSTRA / "movies.csv", "w", newline="", encoding="utf-8") as fout:
        leitor, w = csv.reader(fin), csv.writer(fout)
        w.writerow(next(leitor))
        w.writerows(l for l in leitor if l[0] in filmes)

    total = len(treino) + len(teste)
    print(f"Usuários: {len(por_user)} | Filmes: {len(filmes)} | Avaliações: {total}")
    print(f"Treino: {len(treino)} ({len(treino)/total:.0%}) | Teste: {len(teste)} ({len(teste)/total:.0%})")


if __name__ == "__main__":
    main()
