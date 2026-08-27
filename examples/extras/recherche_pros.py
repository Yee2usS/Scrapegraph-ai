#!/usr/bin/env python3
"""Recherche des professionnels d'un metier donne sur une zone donnee.

Une seule passe par zone: SearchGraph interroge un moteur de recherche, scrape
les premiers resultats, et le LLM fusionne le tout en une liste structuree.
Comptez quelques minutes, pas quelques dizaines.

C'est l'usage direct de ScrapeGraphAI: un prompt, un resultat. Pour un ratissage
exhaustif ville par ville, voir `scrape_cgp_sud_ouest.py`, beaucoup plus long
mais plus couvrant.

Exemples:
    # cabinets de gestion de patrimoine dans le 64, 40 et 65
    python recherche_pros.py

    # un autre metier, une autre zone
    python recherche_pros.py --metier "expert-comptable" --departements 64
    python recherche_pros.py --metier "medecin generaliste" --departements 40 \
        --max-results 15

Necessite OPENAI_API_KEY (ou ANTHROPIC_API_KEY avec --model anthropic/...).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from scrapegraphai.graphs import SearchGraph

DEPARTEMENTS = {
    "64": "Pyrenees-Atlantiques",
    "40": "Landes",
    "65": "Hautes-Pyrenees",
    "33": "Gironde",
    "32": "Gers",
    "31": "Haute-Garonne",
    "47": "Lot-et-Garonne",
}


class Etablissement(BaseModel):
    """Un professionnel ou cabinet identifie."""

    nom: Optional[str] = Field(None, description="Nom du cabinet ou du professionnel")
    adresse: Optional[str] = Field(None, description="Adresse postale complete")
    code_postal: Optional[str] = Field(None, description="Code postal a 5 chiffres")
    ville: Optional[str] = None
    telephone: Optional[str] = None
    email: Optional[str] = None
    site_web: Optional[str] = None
    dirigeants: List[str] = Field(
        default_factory=list, description="Dirigeants, associes ou praticiens"
    )
    specialites: List[str] = Field(
        default_factory=list, description="Specialites ou prestations proposees"
    )


class Etablissements(BaseModel):
    etablissements: List[Etablissement]


CHAMPS = [
    "departement", "nom", "adresse", "code_postal", "ville", "telephone",
    "email", "site_web", "dirigeants", "specialites",
]


# Fenetres de contexte des modeles Claude actuels. La table interne de
# ScrapeGraphAI (`models_tokens`) s'arrete aux modeles Claude 4: un modele plus
# recent y est introuvable et `abstract_graph` retombe alors en silence sur 8192
# jetons, ce qui tronque les pages sans lever d'erreur.
FENETRES_CLAUDE = {
    "claude-opus-5": 1_000_000,
    "claude-sonnet-5": 1_000_000,
    "claude-haiku-4-5": 200_000,
}


def config_llm(modele: str) -> dict:
    """Construit la section `llm` en lisant la cle du bon fournisseur."""
    llm: dict = {"model": modele}

    if modele.startswith("anthropic/"):
        cle = os.getenv("ANTHROPIC_API_KEY")
        if not cle:
            sys.exit("ANTHROPIC_API_KEY absente. Exporte la cle avant de lancer.")
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
            sys.exit("OPENAI_API_KEY absente. Exporte la cle avant de lancer.")
        llm["api_key"] = cle
    elif modele.startswith("ollama/"):
        llm["model_tokens"] = 8192
        llm["format"] = "json"

    return llm


def construire_prompt(metier: str, zone: str, code: str) -> str:
    return (
        f"Liste tous les cabinets et professionnels exercant comme "
        f"{metier} dans le departement {zone} ({code}), en France. "
        "Pour chacun, donne le nom, l'adresse postale complete, le code postal, "
        "la ville, le telephone, l'email, le site web, les dirigeants ou "
        "praticiens, et les specialites. N'invente aucune donnee: laisse le "
        "champ vide si l'information n'est pas disponible."
    )


def ligne_csv(etab: dict, code: str) -> dict:
    ligne = {champ: "" for champ in CHAMPS}
    ligne["departement"] = code
    for champ in CHAMPS:
        valeur = etab.get(champ)
        if isinstance(valeur, list):
            ligne[champ] = " | ".join(str(v) for v in valeur if v)
        elif valeur not in (None, "NA", "NaN"):
            ligne[champ] = str(valeur)
    return ligne


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--metier", default="conseiller en gestion de patrimoine (CGP)")
    p.add_argument("--departements", nargs="+", default=["64", "40", "65"])
    p.add_argument("--out", type=Path, default=Path("resultats.csv"))
    p.add_argument("--model", default="openai/gpt-4o-mini")
    p.add_argument(
        "--max-results",
        type=int,
        default=10,
        help="sites scrapes par departement (plus = plus complet, plus lent, plus cher)",
    )
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    config = {
        "llm": config_llm(args.model),
        "max_results": args.max_results,
        "verbose": args.verbose,
    }

    lignes: List[dict] = []
    for code in args.departements:
        zone = DEPARTEMENTS.get(code, code)
        print(f"\n== {args.metier} - {zone} ({code}) ==")

        graphe = SearchGraph(
            prompt=construire_prompt(args.metier, zone, code),
            config=config,
            schema=Etablissements,
        )
        try:
            resultat = graphe.run()
        except Exception as exc:
            print(f"  ! echec: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

        # Quels sites ont reellement servi de source: a lire, c'est ce qui dit
        # si le resultat vient d'annuaires officiels ou de pages quelconques.
        for url in graphe.get_considered_urls():
            print(f"  source: {url}")

        trouves = (resultat or {}).get("etablissements") or []
        for etab in trouves:
            if isinstance(etab, dict):
                lignes.append(ligne_csv(etab, code))
        print(f"  -> {len(trouves)} etablissement(s)")

    if not lignes:
        print("\nAucun resultat. Augmente --max-results ou reformule --metier.")
        return

    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CHAMPS)
        writer.writeheader()
        writer.writerows(lignes)

    print(f"\n{len(lignes)} ligne(s) ecrite(s) dans {args.out}")


if __name__ == "__main__":
    main()
