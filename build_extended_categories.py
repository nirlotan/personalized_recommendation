"""
Extend the category lists of popular_accounts_manually_validated.xlsx beyond the top 100,
without manual validation or LLMs.

Candidates are the accounts of data/UsersWithWikipedia.csv: SocialVec entities with a
Wikipedia page, in SocialVec popularity order (the order the original lists follow).
Each account is located in Wikidata by its Twitter id (P6552) or username (P2002), and
categorized by:

  1. DBpedia (the source of the original lists): the DBpedia ontology types of the account's
     Wikidata item (via owl:sameAs). A specific DBpedia type decides the category, or that the
     account belongs to none of them.
  2. Wikidata, when DBpedia has only generic types (Person, Company, Website, ...) or no entry:
     "instance of" (P31), "occupation" (P106) and "industry" (P452), each matched against a
     category's root classes and all their subclasses.

    python build_extended_categories.py              # lookups are cached in data/wikidata/
    python build_extended_categories.py --top 1000

Writes data/popular_accounts_extended.xlsx (one sheet per category, up to --top accounts) and
summary tables in results/extended_categories/.
"""

import argparse
import ast
import json
import re
import time
import urllib.parse
import urllib.request

import pandas as pd

from reproduction.config import CATEGORY_LABELS, DATA_DIR, PAPER_CATEGORY_ORDER, RESULTS_DIR
from reproduction.popularity_bands import load_category_sheets

ACCOUNTS = DATA_DIR / "UsersWithWikipedia.csv"
CACHE = DATA_DIR / "wikidata"
OUT_XLSX = DATA_DIR / "popular_accounts_extended.xlsx"
OUT_DIR = RESULTS_DIR / "extended_categories"
USER_AGENT = "personalized-recommendation-research/0.1 (category lists from Wikidata and DBpedia)"

# DBpedia ontology class -> category (the classes the original lists were built from, plus dbo:Band)
DBPEDIA_CLASSES = {
    "MusicalArtist": "MusicalArtist", "Band": "MusicalArtist",
    "Actor": "Actor", "Comedian": "Comedian", "Politician": "Politician", "Journalist": "Journalist",
    "TelevisionHost": "TelevisionHost", "TelevisionShow": "TelevisionShow",
    "TelevisionStation": "TelevisionStation", "SportsTeam": "SportsTeam", "Film": "Film",
    "Newspaper": "news",
}
# DBpedia types too general to decide a category (superclasses of the classes above, or
# "Company"/"Website", which the original news and fashion lists carried); Wikidata decides instead
DBPEDIA_GENERIC = {
    "Agent", "Person", "Organisation", "Company", "Work", "Website", "Thing", "Species", "Animal",
    "Eukaryote", "Artist", "Broadcaster", "Presenter", "Group", "WrittenWork", "PeriodicalLiterature",
}

# Wikidata: category -> [(property, root classes)]; every subclass (P279*) of a root counts
WIKIDATA_RULES = {
    "MusicalArtist": [("P106", ["Q639669"]), ("P31", ["Q215380"])],          # musician; musical group
    "Actor": [("P106", ["Q33999"])],                                          # actor
    "Comedian": [("P106", ["Q245068"])],                                      # comedian
    "Politician": [("P106", ["Q82955"])],                                     # politician
    "Journalist": [("P106", ["Q1930187"])],                                   # journalist
    "TelevisionHost": [("P106", ["Q947873", "Q18668527"])],                   # TV presenter; talk show host
    "SportsTeam": [("P31", ["Q12973014", "Q61974339"])],                      # sports team; esports team
    "TelevisionStation": [("P31", ["Q1616075", "Q2001305", "Q1254874"])],     # TV station; TV channel; TV network
    "TelevisionShow": [("P31", ["Q5398426", "Q15416"])],                      # TV series; TV program
    "news": [("P31", ["Q11032", "Q192283", "Q17232649", "Q1153191", "Q1193236"])],  # newspaper; news agency;
                                                                              # news website; online newspaper; news media
    "Film": [("P31", ["Q11424"])],                                            # film
    "fashion": [("P31", ["Q1941779", "Q1618899"]), ("P452", ["Q11828862", "Q12684"])],  # fashion house; fashion
                                                                              # label; industry: clothing, fashion
    "Resturants": [("P31", ["Q18534542", "Q18509232"])],                      # restaurant chain; fast food chain
    "cars": [("P31", ["Q786820", "Q10429667"])],                              # automobile manufacturer; car brand
}
# Wikidata files "television presenter" and "talk show host" under both actor and journalist, and
# some comedian subclasses under actor; the more specific occupation wins.
MORE_SPECIFIC = {"Actor": ["TelevisionHost", "Comedian"], "Journalist": ["TelevisionHost"]}
# Person categories, in the priority used to pick one category per person when DBpedia cannot
# (more specific occupations first; acting is common as a side occupation, so it comes late)
OCCUPATION_PRIORITY = ["TelevisionHost", "Comedian", "MusicalArtist", "Politician", "Actor", "Journalist"]
# Categories without a proper DBpedia class: always taken from Wikidata, in addition to whatever
# DBpedia decides (e.g. Fox News is a TV station in DBpedia and a news outlet in Wikidata, and was in
# both original lists)
FACET_CATEGORIES = {"news", "fashion", "Resturants", "cars"}


# ---------------------------------------------------------------- SPARQL access
def sparql(endpoint, query, retries=6):
    req = urllib.request.Request(endpoint, data=urllib.parse.urlencode({"query": query, "format": "json"}).encode(),
                                 headers={"User-Agent": USER_AGENT, "Content-Type": "application/x-www-form-urlencoded",
                                          "Accept": "application/sparql-results+json"})
    for attempt in range(retries):
        try:
            return json.load(urllib.request.urlopen(req, timeout=180))["results"]["bindings"]
        except Exception as exc:  # rate limits and timeouts: back off and retry
            print(f"  retry {attempt + 1}: {exc}", flush=True)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"query to {endpoint} failed")


def wikidata_lookup(accounts, chunk=300):
    """Wikidata items of each account (by Twitter id or username) with their P31, P106 and P452 values."""
    path = CACHE / "account_types.csv"
    done = pd.read_csv(path, dtype=str) if path.exists() else pd.DataFrame(columns=["chunk"])
    rows = done.to_dict("records")
    n_chunks = (len(accounts) + chunk - 1) // chunk
    start = 0 if done.empty else int(done["chunk"].astype(int).max()) + 1
    for c in range(start, n_chunks):
        part = accounts.iloc[c * chunk:(c + 1) * chunk]
        ids = " ".join(f'"{i}"' for i in part["user_id"])
        names = " ".join(json.dumps(s) for s in part["screen_name"].dropna())
        query = f"""
        SELECT ?tw ?item ?p31 ?p106 ?p452 WHERE {{
          {{ VALUES ?tw {{ {ids} }} ?item wdt:P6552 ?tw . }} UNION {{ VALUES ?tw {{ {names} }} ?item wdt:P2002 ?tw . }}
          OPTIONAL {{ ?item wdt:P31 ?p31 . }} OPTIONAL {{ ?item wdt:P106 ?p106 . }} OPTIONAL {{ ?item wdt:P452 ?p452 . }}
        }}"""
        seen = set()
        for r in sparql("https://query.wikidata.org/sparql", query):
            tw, item = r["tw"]["value"], r["item"]["value"].rsplit("/", 1)[1]
            for prop in ("p31", "p106", "p452"):
                if prop in r and (key := (tw, item, prop, r[prop]["value"])) not in seen:
                    seen.add(key)
                    rows.append({"chunk": c, "tw": tw, "item": item, "prop": prop.upper(),
                                 "value": r[prop]["value"].rsplit("/", 1)[1]})
        if c % 10 == 0 or c == n_chunks - 1:
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"  Wikidata chunk {c + 1}/{n_chunks}", flush=True)
        time.sleep(0.5)
    types = pd.read_csv(path, dtype=str)
    by_name = dict(zip(accounts["screen_name"].astype(str), accounts["user_id"].astype(str)))
    known_ids = set(accounts["user_id"].astype(str))
    types["user_id"] = types["tw"].where(types["tw"].isin(known_ids), types["tw"].map(by_name))
    return types.dropna(subset=["user_id"])


def subclass_closures():
    """All subclasses (P279*) of every Wikidata root class used in WIKIDATA_RULES."""
    path = CACHE / "subclass_closure.json"
    closures = json.loads(path.read_text()) if path.exists() else {}
    for rules in WIKIDATA_RULES.values():
        for _, roots in rules:
            for root in roots:
                if root not in closures:
                    res = sparql("https://query.wikidata.org/sparql", f"SELECT ?c WHERE {{ ?c wdt:P279* wd:{root} }}")
                    closures[root] = sorted({r["c"]["value"].rsplit("/", 1)[1] for r in res})
                    path.write_text(json.dumps(closures))
                    time.sleep(1)
    return closures


def dbpedia_lookup(items, chunk=200):
    """DBpedia ontology types of each Wikidata item (via owl:sameAs)."""
    path = CACHE / "dbpedia_types.csv"
    done = pd.read_csv(path, dtype=str) if path.exists() else pd.DataFrame(columns=["item", "dbpedia_type"])
    looked_up_path = CACHE / "dbpedia_looked_up.txt"
    looked_up = set(looked_up_path.read_text().split()) if looked_up_path.exists() else set()
    todo = [q for q in items if q not in looked_up]
    rows = done.to_dict("records")
    for start in range(0, len(todo), chunk):
        part = todo[start:start + chunk]
        values = " ".join(f"<http://www.wikidata.org/entity/{q}>" for q in part)
        query = f"""SELECT ?wd ?type WHERE {{ VALUES ?wd {{ {values} }} ?s owl:sameAs ?wd . ?s rdf:type ?type .
                    FILTER(STRSTARTS(STR(?type), "http://dbpedia.org/ontology/")) }}"""
        for r in sparql("https://dbpedia.org/sparql", query):
            rows.append({"item": r["wd"]["value"].rsplit("/", 1)[1], "dbpedia_type": r["type"]["value"].rsplit("/", 1)[1]})
        looked_up.update(part)
        if (start // chunk) % 10 == 0 or start + chunk >= len(todo):
            pd.DataFrame(rows).drop_duplicates().to_csv(path, index=False)
            looked_up_path.write_text("\n".join(sorted(looked_up)))
            print(f"  DBpedia {min(start + chunk, len(todo))}/{len(todo)} items", flush=True)
        time.sleep(0.5)
    return pd.read_csv(path, dtype=str).drop_duplicates()


# ---------------------------------------------------------------- categorization
def wikidata_categories(types, closures):
    """Per account: categories from Wikidata (any matching occupation or type)."""
    def classes(cat, prop):
        cls = set().union(*(closures[r] for p, roots in WIKIDATA_RULES[cat] if p == prop for r in roots))
        for other in MORE_SPECIFIC.get(cat, []):
            cls -= set().union(*(closures[r] for p, roots in WIKIDATA_RULES[other] if p == prop for r in roots))
        return cls

    out = {}
    for cat, rules in WIKIDATA_RULES.items():
        for prop in {p for p, _ in rules}:
            hit = types[(types["prop"] == prop) & types["value"].isin(classes(cat, prop))]["user_id"].unique()
            for u in hit:
                out.setdefault(u, set()).add(cat)
    return out


def first_listed_occupation_categories(accounts, closures):
    """Per person with occupations in the file's (ordered) labels: the category of the first matching occupation."""
    occupation_category = {}
    for cat in OCCUPATION_PRIORITY:
        for _, roots in WIKIDATA_RULES[cat]:
            for root in roots:
                for q in closures[root]:
                    occupation_category.setdefault(q, cat)
    out = {}
    for user_id, labels in zip(accounts["user_id"].astype(str), accounts["labels"]):
        if pd.isna(labels):
            continue
        occupations = ast.literal_eval(labels).get("occupation (P106)") or []
        occupations = occupations if isinstance(occupations, list) else [occupations]
        for occ in occupations:
            q = re.search(r"\((Q\d+)\)$", occ) if isinstance(occ, str) else None
            if q and q.group(1) in occupation_category:
                out[user_id] = {occupation_category[q.group(1)]}
                break
    return out


def categorize(accounts, types, dbpedia, closures, person_rule="first"):
    """
    One row per (account, category) with the source of the decision.

    Categories with a DBpedia class are decided by DBpedia when it gives a specific type (source
    'dbpedia'), and otherwise by Wikidata. For people, Wikidata may match several occupations;
    `person_rule` keeps the first-listed one ('first'; people without ordered occupations in the file
    fall back to 'priority'), the highest one in OCCUPATION_PRIORITY ('priority'), or all of them
    ('all') (source 'wikidata-occupation'). Facet categories (news, fashion,
    food chains, car makers) always come from Wikidata (source 'wikidata'), plus DBpedia's Newspaper.
    """
    item_of = types.groupby("user_id")["item"].apply(set).to_dict()
    db_types = dbpedia.groupby("item")["dbpedia_type"].apply(set).to_dict()
    wd_any = wikidata_categories(types, closures)
    wd_first = first_listed_occupation_categories(accounts, closures) if person_rule == "first" else {}
    person_categories = set(OCCUPATION_PRIORITY)

    rows, info = [], {}
    for user_id in accounts["user_id"].astype(str):
        items = item_of.get(user_id, set())
        dtypes = set().union(*(db_types.get(i, set()) for i in items)) if items else set()
        specific = dtypes - DBPEDIA_GENERIC
        info[user_id] = {"wikidata_items": " ".join(sorted(items)), "dbpedia_types": " ".join(sorted(dtypes - {"Thing"}))}
        wikidata = wd_any.get(user_id, set())
        decided = set()
        if specific:
            decided = {(DBPEDIA_CLASSES[t], "dbpedia") for t in specific if t in DBPEDIA_CLASSES}
        else:
            people = wikidata & person_categories
            if people:
                if person_rule == "first" and user_id in wd_first:
                    people = wd_first[user_id]
                elif person_rule in ("priority", "first"):
                    people = {next(c for c in OCCUPATION_PRIORITY if c in people)}
            decided = {(c, "wikidata-occupation") for c in people}
            decided |= {(c, "wikidata") for c in wikidata - person_categories - FACET_CATEGORIES}
        decided |= {(c, "wikidata") for c in wikidata & FACET_CATEGORIES if c not in {d for d, _ in decided}}
        rows += [(user_id, c, s) for c, s in decided]
    return pd.DataFrame(rows, columns=["user_id", "category", "source"]), info


# ---------------------------------------------------------------- outputs
def compare_person_rules(accounts, types, dbpedia, closures, original):
    """Recall of the original lists and top-100 overlap for each person rule (person categories)."""
    rank = dict(zip(accounts["user_id"].astype(str), accounts["popularity_rank"]))
    found_in_wikidata = set(types["user_id"])
    rows = []
    for rule in ("first", "priority", "all"):
        cats, _ = categorize(accounts, types, dbpedia, closures, person_rule=rule)
        for cat in OCCUPATION_PRIORITY:
            ids = set(cats.loc[cats["category"] == cat, "user_id"])
            orig = original[cat] & found_in_wikidata
            top100 = set(sorted(ids, key=rank.get)[:100])
            rows.append({"person_rule": rule, "category": CATEGORY_LABELS[cat], "accounts": len(ids),
                         "original_found_in_wikidata": len(orig), "original_recovered": len(orig & ids),
                         "top100_overlap_with_original": len(top100 & original[cat])})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--top", type=int, default=1000, help="accounts per category to keep")
    p.add_argument("--person-rule", default="first", choices=["first", "priority", "all"],
                   help="how to pick a person's category when DBpedia cannot (see categorize)")
    args = p.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    accounts = pd.read_csv(ACCOUNTS, low_memory=False)
    accounts["popularity_rank"] = range(1, len(accounts) + 1)  # the file is in SocialVec popularity order
    print("Wikidata lookup ...")
    types = wikidata_lookup(accounts)
    closures = subclass_closures()
    print("DBpedia lookup ...")
    dbpedia = dbpedia_lookup(sorted(types["item"].unique()))
    cats, info = categorize(accounts, types, dbpedia, closures, person_rule=args.person_rule)

    sheets = load_category_sheets()
    original = {c: set(s["twitter_user_id"].astype(str)) for c, s in sheets.items()}
    compare_person_rules(accounts, types, dbpedia, closures, original).to_csv(OUT_DIR / "person_rule_comparison.csv", index=False)
    validated = {c: set(s.loc[s["use"] == 1, "twitter_user_id"].astype(str)) for c, s in sheets.items()}
    acc = accounts.assign(user_id=accounts["user_id"].astype(str)).set_index("user_id")
    wd_values = types.groupby(["user_id", "prop"])["value"].apply(lambda v: " ".join(sorted(set(v)))).unstack()

    summary = []
    with pd.ExcelWriter(OUT_XLSX) as xl:
        for cat in [c for c in PAPER_CATEGORY_ORDER if c in WIKIDATA_RULES]:
            c = cats[cats["category"] == cat].drop_duplicates("user_id")
            c = c.join(acc[["screen_name", "name", "followers_count", "popularity_rank"]], on="user_id").sort_values("popularity_rank")
            full = len(c)
            c = c.head(args.top).reset_index(drop=True)
            sheet = pd.DataFrame({
                "rank_in_category": range(1, len(c) + 1),
                "twitter_screen_name": c["screen_name"].values,
                "twitter_user_id": c["user_id"].astype("int64").values,
                "twitter_name": c["name"].values,
                "followers_count": c["followers_count"].values,
                "socialvec_popularity_rank": c["popularity_rank"].values,
                "category_source": c["source"].values,
                "dbpedia_types": [info[u]["dbpedia_types"] for u in c["user_id"]],
                "wikidata_items": [info[u]["wikidata_items"] for u in c["user_id"]],
                "wikidata_instance_of": [wd_values.at[u, "P31"] if u in wd_values.index and "P31" in wd_values else "" for u in c["user_id"]],
                "wikidata_occupation": [wd_values.at[u, "P106"] if u in wd_values.index and "P106" in wd_values else "" for u in c["user_id"]],
                "in_original_list": [u in original.get(cat, set()) for u in c["user_id"]],
                "use": [1 if u in validated.get(cat, set()) else None for u in c["user_id"]],
            }).fillna({"wikidata_instance_of": "", "wikidata_occupation": ""})
            sheet.to_excel(xl, sheet_name=cat, index=False)

            orig = original.get(cat, set())
            orig_found = {u for u in orig if u in info and info[u]["wikidata_items"]}
            ids = set(c["user_id"])
            summary.append({
                "category": CATEGORY_LABELS[cat], "sheet": cat, "accounts_found": full, "accounts_kept": len(c),
                "rank_of_100th": c["popularity_rank"].iloc[99] if len(c) >= 100 else None,
                "rank_of_200th": c["popularity_rank"].iloc[199] if len(c) >= 200 else None,
                "rank_of_1000th": c["popularity_rank"].iloc[999] if len(c) >= 1000 else None,
                "from_dbpedia": int((c["source"] == "dbpedia").sum()),
                "from_wikidata": int((c["source"] != "dbpedia").sum()),
                "original_list": len(orig),
                "original_found_in_wikidata": len(orig_found),
                "original_recovered": len(orig & ids),
                "validated_top20_recovered": len(validated.get(cat, set()) & ids),
                "validated_top20": len(validated.get(cat, set())),
                "top100_overlap_with_original": len(set(c["user_id"].head(100)) & orig),
            })
        summary = pd.DataFrame(summary)
        summary.to_excel(xl, sheet_name="summary", index=False)

    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    coverage = {
        "accounts": len(accounts),
        "found_in_wikidata": int(accounts["user_id"].astype(str).isin(set(types["user_id"])).sum()),
        "wikidata_items": int(types["item"].nunique()),
        "items_with_dbpedia_types": int(dbpedia["item"].nunique()),
        "accounts_with_a_category": int(cats["user_id"].nunique()),
        "accounts_in_several_categories": int((cats.groupby("user_id")["category"].nunique() > 1).sum()),
        "source_counts": cats.drop_duplicates(["user_id", "category"])["source"].value_counts().to_dict(),
        "person_rule": args.person_rule,
    }
    (OUT_DIR / "coverage.json").write_text(json.dumps(coverage, indent=2))
    print(summary.to_string(index=False))
    print(json.dumps(coverage, indent=2))
    print(f"Wrote {OUT_XLSX}")


if __name__ == "__main__":
    main()
