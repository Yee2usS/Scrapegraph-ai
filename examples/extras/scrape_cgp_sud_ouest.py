#!/usr/bin/env python3
"""Collecte les cabinets de conseil en gestion de patrimoine (CGP) des
Pyrenees-Atlantiques (64), des Landes (40) et des Hautes-Pyrenees (65).

Deux etapes:
  1. DECOUVERTE  -- requetes sur un moteur de recherche, ville par ville,
                    pour constituer une liste d'URL candidates (pas d'appel LLM,
                    donc gratuit: c'est l'etape a lancer en premier pour tester).
  2. EXTRACTION  -- SmartScraperGraph passe sur chaque site et remplit le
                    schema `Cabinet`. C'est ici que le LLM est appele.

Exemples:
    # 1. decouverte seule, rien n'est facture, verifie que le reseau repond
    python scrape_cgp_sud_ouest.py --discover-only

    # 2. petit test d'extraction sur 5 sites (necessite ANTHROPIC_API_KEY et
    #    le connecteur: uv add langchain-anthropic)
    python scrape_cgp_sud_ouest.py --limit 5

    # 3. run complet
    python scrape_cgp_sud_ouest.py --out cgp_sud_ouest.csv

Le script est reprenable: relance-le sur le meme CSV, il saute les URL deja
traitees.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from scrapegraphai.graphs import SmartScraperGraph
from scrapegraphai.utils.research_web import search_on_web

# --------------------------------------------------------------------------
# Perimetre geographique
# --------------------------------------------------------------------------

DEPARTEMENTS = {
    "64": (
        "Pyrenees-Atlantiques",
        [
            "Pau", "Bayonne", "Biarritz", "Anglet", "Saint-Jean-de-Luz",
            "Oloron-Sainte-Marie", "Orthez", "Hendaye", "Bidart", "Ciboure",
            "Cambo-les-Bains", "Mourenx", "Billere", "Lescar", "Ustaritz",
            "Saint-Palais", "Mauleon-Licharre", "Nay",
        ],
    ),
    "40": (
        "Landes",
        [
            "Mont-de-Marsan", "Dax", "Biscarrosse", "Saint-Paul-les-Dax",
            "Capbreton", "Hagetmau", "Soustons", "Tarnos", "Parentis-en-Born",
            "Morcenx", "Aire-sur-l'Adour", "Labenne",
            "Saint-Vincent-de-Tyrosse", "Peyrehorade", "Mimizan",
        ],
    ),
    "65": (
        "Hautes-Pyrenees",
        [
            "Tarbes", "Lourdes", "Bagneres-de-Bigorre", "Aureilhan", "Ibos",
            "Vic-en-Bigorre", "Semeac", "Argeles-Gazost", "Maubourguet",
            "Lannemezan", "Trie-sur-Baise",
        ],
    ),
}

REQUETES = [
    "conseiller en gestion de patrimoine {ville}",
    "cabinet gestion de patrimoine {ville}",
    "CGP conseiller patrimonial {ville} {departement}",
    "conseiller financier independant {ville}",
]

# Annuaires, reseaux sociaux et agregateurs: on veut les sites des cabinets
# eux-memes, pas les fiches d'annuaire (qui bloquent les bots de toute facon).
DOMAINES_EXCLUS = {
    "pagesjaunes.fr", "linkedin.com", "facebook.com", "instagram.com",
    "twitter.com", "x.com", "youtube.com", "societe.com", "verif.com",
    "infogreffe.fr", "pappers.fr", "annuaire-entreprises.data.gouv.fr",
    "indeed.com", "hellowork.com", "leboncoin.fr", "mappy.com",
    "google.com", "yelp.fr", "trouve-ton-cgp.fr", "wikipedia.org",
}


# --------------------------------------------------------------------------
# Schema de sortie
# --------------------------------------------------------------------------


class Cabinet(BaseModel):
    """Fiche d'un cabinet de conseil en gestion de patrimoine."""

    nom: Optional[str] = Field(None, description="Nom commercial du cabinet")
    raison_sociale: Optional[str] = Field(None, description="Denomination juridique")
    adresse: Optional[str] = Field(None, description="Numero et rue")
    code_postal: Optional[str] = Field(None, description="Code postal a 5 chiffres")
    ville: Optional[str] = None
    telephone: Optional[str] = None
    email: Optional[str] = None
    site_web: Optional[str] = None
    linkedin: Optional[str] = None
    dirigeants: List[str] = Field(
        default_factory=list, description="Noms des dirigeants ou associes"
    )
    numero_orias: Optional[str] = Field(
        None, description="Numero d'immatriculation ORIAS s'il est affiche"
    )
    statuts: List[str] = Field(
        default_factory=list,
        description="Statuts reglementaires affiches: CIF, COA, IOBSP, IAS, CJA...",
    )
    services: List[str] = Field(
        default_factory=list,
        description="Prestations proposees: bilan patrimonial, immobilier, "
        "assurance-vie, defiscalisation, retraite, transmission...",
    )


PROMPT = (
    "Extrais les informations de ce cabinet de conseil en gestion de patrimoine "
    "(CGP): nom commercial, raison sociale, adresse postale complete, code postal, "
    "ville, telephone, email, lien LinkedIn, noms des dirigeants ou associes, "
    "numero ORIAS, statuts reglementaires (CIF, COA, IOBSP, IAS) et services "
    "proposes. N'invente aucune donnee: laisse le champ vide si l'information "
    "n'apparait pas sur la page."
)

CHAMPS_CSV = [
    "departement", "nom", "raison_sociale", "adresse", "code_postal", "ville",
    "telephone", "email", "site_web", "linkedin", "dirigeants", "numero_orias",
    "statuts", "services", "url_source",
]


# --------------------------------------------------------------------------
# Etape 1 : decouverte
# --------------------------------------------------------------------------


def domaine(url: str) -> str:
    """Renvoie le domaine enregistrable approximatif d'une URL."""
    hote = urlparse(url).netloc.lower().removeprefix("www.")
    return hote


def est_exclu(url: str) -> bool:
    d = domaine(url)
    return any(d == bloque or d.endswith("." + bloque) for bloque in DOMAINES_EXCLUS)


def decouvrir(
    departements: List[str],
    moteur: str,
    par_requete: int,
    delai: float,
    verbose: bool = True,
) -> dict[str, str]:
    """Renvoie {url_racine: departement} pour les cabinets candidats.

    On deduplique par domaine: un cabinet = un site, peu importe le nombre de
    pages remontees par le moteur.
    """
    trouves: dict[str, str] = {}
    vus: set[str] = set()

    for code in departements:
        nom_dept, villes = DEPARTEMENTS[code]
        for ville in villes:
            for modele in REQUETES:
                requete = modele.format(ville=ville, departement=nom_dept)
                try:
                    resultats = search_on_web(
                        requete,
                        search_engine=moteur,
                        max_results=par_requete,
                        language="fr",
                        region="fr",
                    )
                except Exception as exc:  # reseau, quota moteur, captcha...
                    print(f"  ! echec recherche '{requete}': {exc}", file=sys.stderr)
                    resultats = []

                nouveaux = 0
                for url in resultats:
                    d = domaine(url)
                    if not d or d in vus or est_exclu(url):
                        continue
                    vus.add(d)
                    trouves[f"{urlparse(url).scheme}://{urlparse(url).netloc}"] = code
                    nouveaux += 1

                if verbose:
                    print(f"  [{code}] {requete!r} -> {nouveaux} nouveau(x)")

                # Les moteurs coupent vite si on tape trop regulierement.
                time.sleep(delai + random.uniform(0, delai))

    return trouves


# --------------------------------------------------------------------------
# Etape 2 : extraction
# --------------------------------------------------------------------------


# Fenetres de contexte des modeles Claude actuels. ScrapeGraphAI embarque sa
# propre table (`models_tokens`) mais elle s'arrete aux modeles Claude 4, donc
# un modele recent y est introuvable: `abstract_graph` retombe alors en silence
# sur 8192 jetons, ce qui tronque les pages longues sans lever d'erreur. On
# passe donc `model_tokens` explicitement.
FENETRES_CLAUDE = {
    "claude-opus-5": 1_000_000,
    "claude-sonnet-5": 1_000_000,
    "claude-haiku-4-5": 200_000,
}


def config_graphe(modele: str, headless: bool, verbose: bool) -> dict:
    llm: dict = {"model": modele}

    if modele.startswith("anthropic/"):
        cle = os.getenv("ANTHROPIC_API_KEY")
        if not cle:
            sys.exit(
                "ANTHROPIC_API_KEY absente. Exporte la cle, ou passe un modele local:\n"
                "  --model ollama/llama3.2"
            )
        llm["api_key"] = cle
        nom = modele.split("/", 1)[1]
        if nom in FENETRES_CLAUDE:
            llm["model_tokens"] = FENETRES_CLAUDE[nom]
        else:
            print(
                f"! modele Claude inconnu de ce script ({nom}): sans model_tokens, "
                "ScrapeGraphAI retombera sur 8192 jetons et tronquera les pages.",
                file=sys.stderr,
            )
    elif modele.startswith("openai/"):
        cle = os.getenv("OPENAI_API_KEY")
        if not cle:
            sys.exit(
                "OPENAI_API_KEY absente. Exporte la cle, ou passe un modele local:\n"
                "  --model ollama/llama3.2"
            )
        llm["api_key"] = cle
    elif modele.startswith("ollama/"):
        llm["model_tokens"] = 8192
        llm["format"] = "json"

    return {"llm": llm, "headless": headless, "verbose": verbose}


def deja_traitees(chemin: Path) -> set[str]:
    if not chemin.exists():
        return set()
    with chemin.open(newline="", encoding="utf-8") as f:
        return {ligne["url_source"] for ligne in csv.DictReader(f) if ligne.get("url_source")}


def aplatir(donnees: dict, url: str, dept: str) -> dict:
    """Transforme la sortie du graphe en une ligne CSV."""
    ligne = {champ: "" for champ in CHAMPS_CSV}
    ligne["departement"] = dept
    ligne["url_source"] = url
    for champ in CHAMPS_CSV:
        valeur = donnees.get(champ)
        if isinstance(valeur, list):
            ligne[champ] = " | ".join(str(v) for v in valeur if v)
        elif valeur not in (None, "NA", "NaN"):
            ligne[champ] = str(valeur)
    if not ligne["site_web"]:
        ligne["site_web"] = url
    return ligne


def extraire(
    cibles: dict[str, str],
    sortie: Path,
    config: dict,
    delai: float,
    filtrer_cp: bool,
) -> int:
    faites = deja_traitees(sortie)
    nouveau_fichier = not sortie.exists()
    ecrites = 0

    with sortie.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CHAMPS_CSV)
        if nouveau_fichier:
            writer.writeheader()

        for i, (url, dept) in enumerate(cibles.items(), 1):
            if url in faites:
                print(f"[{i}/{len(cibles)}] deja fait, on saute: {url}")
                continue

            print(f"[{i}/{len(cibles)}] {url}")
            try:
                graphe = SmartScraperGraph(
                    prompt=PROMPT, source=url, config=config, schema=Cabinet
                )
                resultat = graphe.run()
            except Exception as exc:
                print(f"  ! echec: {type(exc).__name__}: {exc}", file=sys.stderr)
                time.sleep(delai)
                continue

            if not isinstance(resultat, dict):
                print("  ! sortie inattendue, ignoree", file=sys.stderr)
                continue

            ligne = aplatir(resultat, url, dept)

            # Un cabinet remonte par la recherche peut etre hors zone.
            cp = ligne["code_postal"].strip()
            if filtrer_cp and cp[:2] and cp[:2] not in DEPARTEMENTS:
                print(f"  - hors perimetre (CP {cp}), ignore")
                continue

            writer.writerow(ligne)
            f.flush()  # on ne perd rien si le run est interrompu
            ecrites += 1
            print(f"  + {ligne['nom'] or '(nom absent)'} - {ligne['ville'] or '?'}")
            time.sleep(delai)

    return ecrites


# --------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--departements", nargs="+", default=["64", "40", "65"], choices=list(DEPARTEMENTS))
    p.add_argument("--out", type=Path, default=Path("cgp_sud_ouest.csv"))
    p.add_argument("--urls", type=Path, help="JSON d'URL a scraper, au lieu de la decouverte")
    p.add_argument("--discover-only", action="store_true", help="s'arrete apres la decouverte (aucun appel LLM)")
    p.add_argument("--model", default="openai/gpt-4o-mini")
    p.add_argument("--search-engine", default="duckduckgo", choices=["duckduckgo", "bing", "searxng", "serper"])
    p.add_argument("--results-per-query", type=int, default=8)
    p.add_argument("--limit", type=int, help="ne traite que les N premieres cibles")
    p.add_argument("--delay", type=float, default=2.0, help="pause en secondes entre deux appels")
    p.add_argument("--no-headless", action="store_true")
    p.add_argument("--no-filter-cp", action="store_true", help="garde les fiches hors 64/40/65")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    fichier_urls = args.out.with_suffix(".urls.json")

    if args.urls:
        cibles = json.loads(args.urls.read_text(encoding="utf-8"))
    else:
        print(f"== Decouverte sur les departements {', '.join(args.departements)} ==")
        cibles = decouvrir(
            args.departements, args.search_engine, args.results_per_query, args.delay
        )
        fichier_urls.write_text(json.dumps(cibles, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n{len(cibles)} sites candidats -> {fichier_urls}")

    if args.discover_only:
        print("--discover-only: on s'arrete la. Relis la liste avant de payer du LLM.")
        return

    if args.limit:
        cibles = dict(list(cibles.items())[: args.limit])

    config = config_graphe(args.model, not args.no_headless, args.verbose)
    print(f"\n== Extraction de {len(cibles)} sites avec {args.model} ==")
    n = extraire(cibles, args.out, config, args.delay, not args.no_filter_cp)
    print(f"\n{n} fiche(s) ecrite(s) dans {args.out}")


if __name__ == "__main__":
    main()
