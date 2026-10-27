"""Módulo único do projeto: configuração, conexão com o Neo4j, Pearson item-item,
regressão KNN e métricas. Só biblioteca padrão (o driver neo4j é importado em conectar())."""
import math
import os
from collections import defaultdict
from pathlib import Path

# ----------------------------------------------------------------- configuração
ROOT = Path(__file__).parent
DATA = ROOT / "data"          # ratings.csv e movies.csv originais
AMOSTRA = DATA / "amostra"    # train.csv, test.csv, movies.csv da amostra


def _env(nome, padrao, tipo=int):
    return tipo(os.getenv(nome, padrao))


SEED = _env("SEED", 42)
MAX_FILMES = _env("MAX_FILMES", 1000)            # filmes mais avaliados
MIN_AVAL_FILME = _env("MIN_AVAL_FILME", 50)
MAX_USUARIOS = _env("MAX_USUARIOS", 3000)
MIN_AVAL_USUARIO = _env("MIN_AVAL_USUARIO", 20)
FRACAO_TREINO = _env("FRACAO_TREINO", 0.8, float)

MIN_COMUM = _env("MIN_COMUM", 5)                 # usuários em comum mínimos p/ criar SIMILAR_TO
VIZINHOS_GUARDADOS = _env("VIZINHOS_GUARDADOS", 50)
K = _env("K", 20)                                # K do KNN
LIMIAR = _env("LIMIAR", 4.0, float)              # nota >= 4 => "gostou"
TOP_N = _env("TOP_N", 10)
USUARIOS_AVALIACAO = _env("USUARIOS_AVALIACAO", 500)


# ----------------------------------------------------------------- Neo4j
def conectar():
    from neo4j import GraphDatabase
    senha = os.getenv("NEO4J_PASSWORD")
    if not senha:
        raise RuntimeError("Defina a variável de ambiente NEO4J_PASSWORD (veja .env.example).")
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), senha))
    driver.verify_connectivity()
    return driver


def sessao(driver):
    return driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j"))


def ler_avaliacoes(session, split):
    """{userId: {movieId: nota}} para split 'train' ou 'test'."""
    q = ("MATCH (u:User)-[r:RATED {split: $split}]->(m:Movie) "
         "RETURN u.userId AS u, m.movieId AS m, r.rating AS r")
    out = defaultdict(dict)
    for rec in session.run(q, split=split):
        out[rec["u"]][rec["m"]] = rec["r"]
    return out


def ler_vizinhos(session):
    """{movieId: [(vizinho, pearson, n_comum), ...]} ordenado por pearson desc."""
    q = ("MATCH (i:Movie)-[s:SIMILAR_TO]->(j:Movie) "
         "RETURN i.movieId AS i, j.movieId AS j, s.pearson AS c, s.n_common AS n")
    out = defaultdict(list)
    for rec in session.run(q):
        out[rec["i"]].append((rec["j"], rec["c"], rec["n"]))
    for i in out:
        out[i].sort(key=lambda t: (-t[1], t[0]))
    return dict(out)


# ----------------------------------------------------------------- modelo
def medias_filmes(avaliacoes):
    """Nota média (r_barra_i) e quantidade de avaliações de cada filme."""
    soma, qtd = defaultdict(float), defaultdict(int)
    for itens in avaliacoes.values():
        for m, r in itens.items():
            soma[m] += r
            qtd[m] += 1
    return {m: soma[m] / qtd[m] for m in soma}, dict(qtd)


def media_global(avaliacoes):
    total = n = 0
    for itens in avaliacoes.values():
        total += sum(itens.values())
        n += len(itens)
    return total / n if n else 0.0


def pearson_vizinhos(avaliacoes, min_comum=MIN_COMUM, top_n=VIZINHOS_GUARDADOS):
    """c(i,j) = Σk (r_i^k - r̄_i)(r_j^k - r̄_j) / ( √Σk (r_i^k - r̄_i)² · √Σk (r_j^k - r̄_j)² ),
    com k nos usuários que avaliaram i E j. Retorna {i: [(j, c, n_comum), ...]} (c desc, top_n)."""
    medias, _ = medias_filmes(avaliacoes)
    acc = {}  # (i, j), i<j -> [sxy, sxx, syy, n]
    for itens in avaliacoes.values():
        dev = sorted((m, r - medias[m]) for m, r in itens.items())
        for a in range(len(dev)):
            i, di = dev[a]
            for b in range(a + 1, len(dev)):
                j, dj = dev[b]
                v = acc.get((i, j))
                if v is None:
                    acc[(i, j)] = [di * dj, di * di, dj * dj, 1]
                else:
                    v[0] += di * dj
                    v[1] += di * di
                    v[2] += dj * dj
                    v[3] += 1
    viz = defaultdict(list)
    for (i, j), (sxy, sxx, syy, n) in acc.items():
        den = math.sqrt(sxx) * math.sqrt(syy)
        if n < min_comum or den == 0:
            continue
        c = sxy / den
        viz[i].append((j, c, n))
        viz[j].append((i, c, n))
    for i in viz:
        viz[i].sort(key=lambda t: (-t[1], t[0]))
        del viz[i][top_n:]
    return dict(viz)


def prever(filme, notas_usuario, vizinhos, k=K, min_vizinhos=1):
    """KNN item-item: r̂(u,i) = Σj c(i,j)·r(u,j) / Σj |c(i,j)|, j = os k filmes mais
    similares a i (c > 0) que o usuário u avaliou. Retorna (previsão ou None, vizinhos usados)."""
    num = den = 0.0
    usados = 0
    for j, c, _ in vizinhos.get(filme, ()):  # já ordenado por c desc
        if c <= 0:
            break
        r = notas_usuario.get(j)
        if r is None:
            continue
        num += c * r
        den += abs(c)
        usados += 1
        if usados >= k:
            break
    if usados >= min_vizinhos and den > 0:
        return num / den, usados
    return None, usados


def prever_com_reserva(filme, notas_usuario, vizinhos, medias, gmedia, k=K):
    """Sem vizinhos: média do filme; sem média: média global. Retorna (nota, usou_knn)."""
    p, _ = prever(filme, notas_usuario, vizinhos, k)
    if p is not None:
        return p, True
    return medias.get(filme, gmedia), False


def recomendar(notas_usuario, vizinhos, k=K, n=TOP_N):
    """Top-n filmes ainda não vistos (no treino) com maior nota prevista."""
    lista = []
    for filme in vizinhos:
        if filme in notas_usuario:
            continue
        p, usados = prever(filme, notas_usuario, vizinhos, k)
        if p is not None:
            lista.append((p, usados, filme))
    lista.sort(key=lambda t: (-t[0], -t[1], t[2]))
    return [(f, p) for p, _, f in lista[:n]]


def precisao_recall(recomendados, relevantes):
    acertos = len(set(recomendados) & set(relevantes))
    precisao = acertos / len(recomendados) if recomendados else 0.0
    recall = acertos / len(relevantes) if relevantes else 0.0
    return precisao, recall
