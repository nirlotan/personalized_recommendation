"""
A short page with only the tables and figures of results_v4.html that the latest experiments changed:
the exclusion of organizations and likely bots, and the 6, 12 and 25 names drawn from the full follow list
and the 6 closed-set names.

    python make_results_page_v4.py
    python make_new_experiments_page.py

Writes results/new_experiments_v4.html (the elements are copied from results_v4.html, so the two agree).
"""
import copy
import re

from bs4 import BeautifulSoup

from reproduction.config import ROOT

R = ROOT / "results"


def heading(page, tag, prefix):
    return next(h for h in page.find_all(tag) if h.get_text(" ", strip=True).startswith(prefix))


def card_with(page, prefix):
    """The card (div.card) holding the h4 or h3-titled element that starts with `prefix`."""
    h = heading(page, ["h4", "summary"], prefix)
    return h.find_parent("div", class_="card")


def following_until(h, stop_tags=("h3", "h2")):
    """The siblings after heading `h` up to the next heading."""
    out = []
    for el in h.find_next_siblings():
        if el.name in stop_tags:
            break
        out.append(el)
    return out


def main():
    page = BeautifulSoup((R / "results_v4.html").read_text(encoding="utf-8"), "html.parser")
    parts = []

    def add(*els):
        parts.extend(copy.copy(e) for e in els)

    def h(level, text):
        return BeautifulSoup(f"<{level}>{text}</{level}>", "html.parser").contents[0]

    # Tables 9-11: the exclusion paragraph, the note on the new columns and the cards
    t9 = heading(page, "h3", "Table 9 ")
    t9_parts = following_until(t9)
    excl = next(p for p in t9_parts if p.name == "p" and p.get_text().startswith("Organizations and likely bots"))
    note = next(p for p in t9_parts if p.name == "p" and "of 50" in p.get_text())
    add(h("h2", "Users: organizations and likely bots left out"), excl)
    add(h("h2", "LLM ranking per category with the new conditions"), note, t9,
        *[e for e in t9_parts if e.name == "div" and "card" in (e.get("class") or [])])
    t10 = heading(page, "h3", "Table 10 ")
    add(t10, *[e for e in following_until(t10) if e.name == "div" and "card" in (e.get("class") or [])])

    # Section 5: Figure 4, Table 16 and the source-vs-number analysis (Table 23)
    add(h("h2", "Demographics vs. account names"), card_with(page, "Figure 4 "), card_with(page, "Table 16 "))
    src = heading(page, "h3", "Where the account names come from")
    add(src, *following_until(src))

    # supplement: Claude Haiku 5.5 per category and completeness
    add(h("h2", "Supplementary · Claude Haiku 5.5 per category"))
    for prefix in ("Table 20 ", "Table 21 ", "Table 22 "):
        t = heading(page, "h3", prefix)
        add(t, *[e for e in following_until(t) if e.name == "div" and "card" in (e.get("class") or [])])

    body = "\n".join(str(p) for p in parts)
    style = str(page.head.find("style"))
    script = str(page.find("script")) if page.find("script") else ""
    intro = ("<p class='intro'>Only the tables and figures of <a href='results_v4.html'>results_v4.html</a> that the experiments of the "
             "last two days changed: 56 organizations and likely bots left out of the users of Tables 9–11 and Section 5; account names "
             "drawn from the full follow list (25 names from the user's k = 50 list, and the first 12 and 6 of those 25), for "
             "GPT-6 Luna and Claude Haiku 5.5; and 6 closed-set names (the first 6 of the k = 12 list). Table and figure numbers are "
             "those of results_v4.html.</p>")
    html = ("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>New Experiments Summary</title>{style}</head><body><div class='wrap'>"
            f"<h1>New experiments · results_v4</h1>{intro}{body}</div>{script}</body></html>")
    html = re.sub(r"\n{3,}", "\n\n", html)
    out = R / "new_experiments_v4.html"
    out.write_text(html, encoding="utf-8")
    print("wrote", out, f"{len(html) / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
