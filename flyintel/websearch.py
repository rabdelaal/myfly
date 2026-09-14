"""
WebSearch — outil de recherche web pour le modèle (apprentissage de données).

Le modèle (la mouche / le readout) peut appeler `search()` pour interroger le
web et `learn()` pour ingérer et persister de nouvelles données à chaud.

Backend :
  1. Exa Search (exa.ai) si EXA_API_KEY est définie — résultats riches + extrait.
  2. Sinon repli DuckDuckGo HTML (sans clé) : résultats gratuits, extrait court.

Usage (depuis le code du modèle) :
    from flyintel import websearch
    for r in websearch.search("alpha-synapse conductance male CNS"):
        print(r["title"], r["url"])
    saved = websearch.learn("latest connectome papers", dir="knowledge")
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

import requests

_USER_AGENT = "myfly-websearch/0.1 (+autonomous model learning)"
_HEADERS = {"User-Agent": _USER_AGENT}


# --------------------------------------------------------------------------- #
# Backend Exa (exige EXA_API_KEY)
# --------------------------------------------------------------------------- #
def _exa_search(query: str, num: int) -> list[dict]:
    key = os.environ.get("EXA_API_KEY")
    if not key:
        return []
    r = requests.post(
        "https://api.exa.ai/search",
        headers={"Authorization": f"Bearer {key}", **_HEADERS},
        json={"query": query, "numResults": num, "contents": {"text": True}},
        timeout=30,
    )
    r.raise_for_status()
    out = []
    for item in r.json().get("results", []):
        text = (item.get("text") or "").strip()
        out.append({
            "title": item.get("title") or item.get("url", ""),
            "url": item.get("url", ""),
            "snippet": text[:500],
            "text": text,
            "source": "exa",
        })
    return out


# --------------------------------------------------------------------------- #
# Repli DuckDuckGo HTML (sans clé)
# --------------------------------------------------------------------------- #
def _duckduckgo_search(query: str, num: int) -> list[dict]:
    r = requests.get(
        "https://html.duckduckgo.com/html/",
        params={"q": query},
        headers=_HEADERS,
        timeout=30,
    )
    r.raise_for_status()
    # Extrait les blocs de résultat (lien + titre + extrait) du HTML.
    # Layout : <a class="result__a" href="...">titre</a> puis
    #          <a class="result__snippet" href="...">extrait</a>
    results = []
    for a, sn in zip(
        re.findall(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', r.text),
        re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', r.text),
    ):
        url, title = a
        url = re.sub(r"^//", "https://", url)
        # Décode le wrapper de redirection DuckDuckGo (?uddg=<url>&rut=...)
        m = re.search(r"[?&]uddg=([^&]+)", url)
        if m:
            from urllib.parse import unquote

            url = unquote(m.group(1))
        title = re.sub(r"<[^>]+>", "", title)
        snippet = re.sub(r"<[^>]+>", "", sn).strip()
        results.append({
            "title": title.strip(),
            "url": url,
            "snippet": snippet[:500],
            "text": "",
            "source": "duckduckgo",
        })
        if len(results) >= num:
            break
    return results


# --------------------------------------------------------------------------- #
# API publique
# --------------------------------------------------------------------------- #
def search(query: str, num: int = 5) -> list[dict]:
    """Recherche web, retourne une liste de {title, url, snippet, text, source}.
    Utilise Exa si EXA_API_KEY est définie, sinon DuckDuckGo (gratuit)."""
    exa = _exa_search(query, num)
    if exa:
        return exa
    return _duckduckgo_search(query, num)


def fetch(url: str, max_chars: int = 20000) -> str:
    """Récupère et extrait le texte lisible d'une page web."""
    r = requests.get(url, headers=_HEADERS, timeout=30)
    r.raise_for_status()
    text = re.sub(r"<script.*?</script>|<style.*?</style>", "", r.text, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def learn(query: str, dir: str = "knowledge", num: int = 3,
          sleep_s: float = 1.0) -> list[str]:
    """Recherche, récupère et PERSISTE les N premiers résultats en markdown dans
    `dir/`. Retourne la liste des fichiers écrits — à ré-ingérer par le modèle.

    Chaque fichier : <dir>/<slug>-<i>.md  (métadonnée + texte plein)."""
    d = Path(dir)
    d.mkdir(parents=True, exist_ok=True)
    saved = []
    slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")[:40] or "query"
    for i, res in enumerate(search(query, num)):
        text = res.get("text") or res.get("snippet") or ""
        if not text:
            try:
                text = fetch(res["url"])
            except requests.RequestException:
                text = res.get("snippet") or ""
        path = d / f"{slug}-{i}.md"
        path.write_text(
            f"# {res['title']}\n\n"
            f"- source: {res['url']}\n"
            f"- engine: {res.get('source')}\n\n"
            f"{text}\n",
            encoding="utf-8",
        )
        saved.append(str(path))
        if sleep_s and i < num - 1:
            time.sleep(sleep_s)  # politesse rate-limit
    return saved


# --------------------------------------------------------------------------- #
# Self-check (ponytail : un check exécutable minimal)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    r = search("Male CNS connectome Drosophila")
    print(f"search -> {len(r)} resultat(s); premier: {r[0]['title'] if r else '(vide)'}")
    assert r, "la recherche web doit retourner des resultats (Exa ou DuckDuckGo)"
    saved = learn("Drosophila connectome", dir="_websearch_demo", num=1, sleep_s=0)
    print("learn ->", saved)
    assert saved and Path(saved[0]).exists()
    print("OK")