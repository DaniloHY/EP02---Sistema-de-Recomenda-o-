"""Avaliação (treino 80% já está no grafo como split='train'; teste = split='test').

[A] Top-N: relevantes = filmes do TESTE com nota >= LIMIAR.
    Precisão@N = acertos/N ; Recall@N = acertos/|relevantes| (média por usuário).
    Compara com um baseline de popularidade.
[B] Pares do teste: positivo = nota >= LIMIAR. Precisão = TP/(TP+FP), Recall = TP/(TP+FN).
    Também RMSE, MAE e cobertura do KNN."""
import argparse
import math
import random
import recsys as R


def media(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=R.USUARIOS_AVALIACAO)
    ap.add_argument("--top-n", type=int, default=R.TOP_N)
    ap.add_argument("--k", type=int, default=R.K)
    args = ap.parse_args()

    with R.conectar() as driver, R.sessao(driver) as s:
        treino = R.ler_avaliacoes(s, "train")
        teste = R.ler_avaliacoes(s, "test")
        vizinhos = R.ler_vizinhos(s)

    medias, qtd = R.medias_filmes(treino)
    gmedia = R.media_global(treino)
    populares = [m for m, _ in sorted(qtd.items(), key=lambda t: (-t[1], t[0]))]

    usuarios = [u for u in sorted(teste) if u in treino and any(r >= R.LIMIAR for r in teste[u].values())]
    rng = random.Random(R.SEED)
    if len(usuarios) > args.users:
        usuarios = rng.sample(usuarios, args.users)
    print(f"Usuários avaliados: {len(usuarios)} | K={args.k} | N={args.top_n} | limiar={R.LIMIAR}")

    # [A] Top-N
    P, Rc, Pb, Rb = [], [], [], []
    for u in usuarios:
        relevantes = {m for m, r in teste[u].items() if r >= R.LIMIAR}
        recs = [m for m, _ in R.recomendar(treino[u], vizinhos, args.k, args.top_n)]
        p, r = R.precisao_recall(recs, relevantes)
        P.append(p); Rc.append(r)
        base = [m for m in populares if m not in treino[u]][:args.top_n]
        p, r = R.precisao_recall(base, relevantes)
        Pb.append(p); Rb.append(r)
    print(f"\n[A] Top-{args.top_n}")
    print(f"  KNN item-item  : Precisão@{args.top_n} = {media(P):.4f} | Recall@{args.top_n} = {media(Rc):.4f}")
    print(f"  Baseline popul.: Precisão@{args.top_n} = {media(Pb):.4f} | Recall@{args.top_n} = {media(Rb):.4f}")

    # [B] Pares do teste
    tp = fp = fn = tn = n = knn = 0
    se = ae = 0.0
    for u in usuarios:
        for filme, real in teste[u].items():
            pred, usou = R.prever_com_reserva(filme, treino[u], vizinhos, medias, gmedia, args.k)
            n += 1
            knn += usou
            se += (pred - real) ** 2
            ae += abs(pred - real)
            pp, pr = pred >= R.LIMIAR, real >= R.LIMIAR
            tp += pp and pr
            fp += pp and not pr
            fn += (not pp) and pr
            tn += (not pp) and not pr
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    print(f"\n[B] Pares do teste (n={n}, cobertura KNN={knn / n:.1%})")
    print(f"  RMSE = {math.sqrt(se / n):.4f} | MAE = {ae / n:.4f}")
    print(f"  Precisão = {prec:.4f} | Recall = {rec:.4f}  (TP={tp} FP={fp} FN={fn} TN={tn})")


if __name__ == "__main__":
    main()
