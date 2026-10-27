"""Baixa o MovieLens 20M oficial (GroupLens, o mesmo dataset do Kaggle) e extrai SOMENTE
ratings.csv e movies.csv em data/. Se já tiver os arquivos do Kaggle, copie-os para data/ e pule este passo."""
import urllib.request
import zipfile
from recsys import DATA

URL = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
ZIP = DATA / "ml-20m.zip"


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    if not ZIP.exists():
        print("Baixando MovieLens 20M (aprox. 190 MB)...")
        parcial = ZIP.with_suffix(".part")
        urllib.request.urlretrieve(URL, parcial)
        parcial.rename(ZIP)
    with zipfile.ZipFile(ZIP) as z:
        for nome in ("ratings.csv", "movies.csv"):
            destino = DATA / nome
            if not destino.exists():
                print("Extraindo", nome)
                with z.open("ml-20m/" + nome) as src, open(destino, "wb") as dst:
                    while True:
                        bloco = src.read(1024 * 1024)
                        if not bloco:
                            break
                        dst.write(bloco)
    print("Pronto:", DATA)


if __name__ == "__main__":
    main()
