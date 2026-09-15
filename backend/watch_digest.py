"""
Digest de veille one-shot : apprend un sujet (websearch.learn) puis résume
par extraction — les N phrases qui recouvrent le plus les mots du sujet.
Zéro modèle, zéro entraînement : du comptage, en secondes.

Usage :
    from watch_digest import digest
    print(digest("drosophila connectome", num=2))
    # + cron système pour la veille périodique (le module reste one-shot).
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flyintel import websearch


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if len(s.strip()) > 40]


def _score(sent: str, keywords: set[str]) -> int:
    words = set(re.findall(r"[a-z]{3,}", sent.lower()))
    return len(words & keywords)


def digest(topic: str, num: int = 2, per_doc: int = 3, dir: str = "knowledge") -> str:
    """Apprend `topic` puis retourne un digest texte avec sources."""
    keywords = set(re.findall(r"[a-z]{3,}", topic.lower()))
    files = websearch.learn(topic, dir=dir, num=num)
    lines = [f"# Digest : {topic}", ""]
    for f in files:
        parsed = websearch._parse_learned(Path(f))
        ranked = sorted(_sentences(parsed["text"]),
                        key=lambda s: _score(s, keywords), reverse=True)
        lines.append(f"## {parsed['title']}")
        lines.append(f"source: {parsed['source']}")
        for s in ranked[:per_doc]:
            lines.append(f"- {s}")
        lines.append("")
    return "\n".join(lines).strip()


if __name__ == "__main__":
    d = digest("drosophila connectome", num=1, per_doc=2, dir="_digest_demo")
    print(d[:600])
    assert d.startswith("# Digest") and "source: http" in d
    import shutil
    shutil.rmtree("_digest_demo", ignore_errors=True)
    print("OK")