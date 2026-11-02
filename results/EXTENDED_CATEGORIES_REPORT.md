# Extending the category lists beyond the top 100

*Built by `python build_extended_categories.py` (no LLM, no manual validation). Output:
`data/popular_accounts_extended.xlsx`. Summary tables: `results/extended_categories/`.*

## Summary

- **Goal:** extend the 14 category lists of `popular_accounts_manually_validated.xlsx` beyond the top 100 popular
  entities, using only `data/UsersWithWikipedia.csv` and free knowledge bases.
- **Output:** a new spreadsheet with one sheet per category (same sheet names as the original), each holding up to the
  1,000 most popular accounts of that category, ordered by SocialVec popularity. Each account records the source and
  evidence for its category.
- **Method:** every account is located in **Wikidata** by its Twitter id or username. Its category comes from its
  **DBpedia** ontology type (the source of the original lists) when DBpedia gives a specific type, and otherwise from
  Wikidata's "instance of", "occupation" and "industry" statements.
- **Result, 200 accounts:** ten categories reach 200 accounts (musical artists, actors, journalists, politicians, news
  outlets, comedians, TV shows, TV stations, sports teams, TV hosts).
- **Result, 1,000 accounts:** three reach 1,000 (musical artists, actors, journalists).
- **Result, small categories:** films (70), fashion (90), food chains (107) and car makers (40) cannot reach 200 from this
  file.
- **Agreement with the original lists:** ten categories recover 81–100% of the existing lists' entities (where those
  entities are in Wikidata), and most recover 15–19 of the 20 manually validated entities. TV hosts, films, fashion and
  news outlets recover less (56–67%), for the reasons in Section 4.

## 1. Data sources

| Source | What it provides |
|---|---|
| `data/UsersWithWikipedia.csv` | 31,502 SocialVec entities that have a Wikipedia page, listed in SocialVec popularity order (how often they are followed in SocialVec's user sample). This is the order the original lists follow; 98% of these accounts are followed by at least one of the experiment's users. The file's `labels` column carries Wikidata statements for 11,359 people (occupation, gender, …), in Wikidata's statement order. |
| Wikidata (query.wikidata.org) | The Wikidata item of each account, found through the Twitter user id (P6552) or Twitter username (P2002), with its "instance of" (P31), "occupation" (P106) and "industry" (P452) values, and the subclass hierarchy (P279) of the classes used. |
| DBpedia (dbpedia.org/sparql) | DBpedia ontology types of each Wikidata item, linked through `owl:sameAs`. The original lists were built from these types (their `dbpedia_types` column), so DBpedia is the primary source. |

Both knowledge bases are free public SPARQL endpoints. Queries were batched (300 accounts or 200 items per query),
paced, and cached in `data/wikidata/`, so rerunning the build makes no new queries.

## 2. Method

### 2.1 Candidates and popularity

All 31,502 accounts of `UsersWithWikipedia.csv` are candidates. Popularity is the account's position in
the file (SocialVec popularity rank, 1 = most followed). Within each category, accounts are ordered by this rank, and the
first 1,000 are kept (`--top`).

### 2.2 Linking accounts to Wikidata

Each account is looked up in Wikidata by its numeric Twitter id (P6552) and by its username (P2002).
- **Coverage:** 19,343 accounts (61%) were found, mapping to
  19,450 items. Coverage is high for the most popular accounts (92% of a sample of the top accounts) and
  lower further down.
- **Several items per account:** a few accounts link to more than one item (e.g. `@nytimes` to both *The New York Times*
  and *nytimes.com*). Their types are combined.
- **Unlinked accounts:** accounts that cannot be linked get no category.

### 2.3 DBpedia types (primary source)

For each linked Wikidata item, the DBpedia resource with `owl:sameAs` to it supplies its DBpedia ontology types;
17,638 of the 19,450 items have DBpedia types. A type decides the category when it is specific:

| DBpedia class | Category |
|---|---|
| `dbo:MusicalArtist` | Musical artists |
| `dbo:Band` | Musical artists |
| `dbo:Actor` | Actors |
| `dbo:Comedian` | Comedians |
| `dbo:Politician` | Politicians |
| `dbo:Journalist` | Journalists |
| `dbo:TelevisionHost` | TV hosts |
| `dbo:TelevisionShow` | TV shows |
| `dbo:TelevisionStation` | TV stations |
| `dbo:SportsTeam` | Sports teams |
| `dbo:Film` | Films |
| `dbo:Newspaper` | News outlets |

**How the types are read:**
- **Generic types:** `Person`, `Agent`, `Organisation`, `Company`, `Work`, `Website`, `Artist`, `Broadcaster`, `Presenter`,
  `Group`, `PeriodicalLiterature` and the like are too general to decide a category, so Wikidata decides instead (2.4).
- **Specific types outside the categories:** a specific type that is none of the classes above (e.g. `BasketballPlayer`,
  `Writer`, `RadioProgram`) means the account belongs to none of these categories. For example, LeBron James is
  `BasketballPlayer` in DBpedia, so he is not counted as an actor even though Wikidata lists acting among his
  occupations.
- **Bands:** `dbo:Band` is added to musical artists. The original lists contain individual artists only.

### 2.4 Wikidata rules (fallback and facets)

Each category is defined by root classes. An item matches a category if one of its values for the given property is a
root class or any subclass of it (the transitive P279 closure, computed once per root).

| Category | Wikidata roots | Role |
|---|---|---|
| Musical artists | occupation (P106): musician (Q639669); instance of (P31): musical group (Q215380) | person |
| Actors | occupation (P106): actor (Q33999) | person |
| Comedians | occupation (P106): comedian (Q245068) | person |
| Politicians | occupation (P106): politician (Q82955) | person |
| Journalists | occupation (P106): journalist (Q1930187) | person |
| TV hosts | occupation (P106): television presenter (Q947873), talk show host (Q18668527) | person |
| Sports teams | instance of (P31): sports team (Q12973014), esports team (Q61974339) | DBpedia first |
| TV stations | instance of (P31): television station (Q1616075), television channel (Q2001305), television network (Q1254874) | DBpedia first |
| TV shows | instance of (P31): television series (Q5398426), television program (Q15416) | DBpedia first |
| News outlets | instance of (P31): newspaper (Q11032), news agency (Q192283), news website (Q17232649), online newspaper (Q1153191), news media (Q1193236) | facet |
| Films | instance of (P31): film (Q11424) | DBpedia first |
| Fashion | instance of (P31): fashion house (Q1941779), fashion label (Q1618899); industry (P452): clothing industry (Q11828862), fashion (Q12684) | facet |
| Food chains | instance of (P31): restaurant chain (Q18534542), fast food restaurant chain (Q18509232) | facet |
| Car makers | instance of (P31): automobile manufacturer (Q786820), car brand (Q10429667) | facet |

**Roles:**
- **DBpedia first:** Wikidata is used only when DBpedia has no specific type for the account.
- **Person:** as DBpedia first, with one category chosen per person (2.5).
- **Facet:** these categories have no proper DBpedia class, so they are always taken from Wikidata, in addition to
  whatever DBpedia decided. For example, Fox News is a TV station in DBpedia and a news outlet in Wikidata, and it was in
  both original lists. DBpedia's `Newspaper` also counts as news.

**A correction to the Wikidata hierarchy.** Wikidata files *television presenter* and *talk show host* as subclasses of
both *actor* and *journalist*, and a few comedian subclasses under *actor*. Taken literally, every TV host would also be
an actor and a journalist. The more specific occupation therefore wins: those classes are removed from the actor and
journalist closures.

### 2.5 One category per person

Wikidata occupations include side careers: anyone with an acting credit is an actor. When DBpedia gives only `Person`,
we keep one person category, taken from the person's **first-listed** matching occupation in Wikidata's statement order
(from the file's `labels`). People without ordered occupations fall back to a fixed priority: TV host, comedian,
musician, politician, actor, journalist.

We chose this rule by comparing three options on the six person categories. Each cell shows the original-list entities
recovered (of those found in Wikidata), then the overlap of the new top 100 with the original list:

| Category | First-listed (chosen) | Fixed priority | All occupations |
|---|---|---|---|
| TV hosts | 14/22 · 4 | 15/22 · 1 | 15/22 · 1 |
| Comedians | 89/93 · 36 | 90/93 · 28 | 92/93 · 26 |
| Musical artists | 78/94 · 52 | 89/94 · 53 | 92/94 · 52 |
| Politicians | 88/89 · 56 | 88/89 · 55 | 88/89 · 55 |
| Actors | 85/98 · 32 | 56/98 · 32 | 92/98 · 21 |
| Journalists | 83/95 · 20 | 71/95 · 24 | 88/95 · 16 |
| **Total** | **437 · 200** | **409 · 193** | **467 · 171** |

**Trade-off:**
- *All occupations:* recovers the most original entities, but fills the top of the lists with side careers.
- *Fixed priority:* moves most actor-musicians out of the actor list.
- *First-listed:* gives the best balance.

The other rules are available with `--person-rule`.

### 2.6 Output format

`data/popular_accounts_extended.xlsx` has one sheet per category (same sheet names as the original file) and a
`summary` sheet. Columns per account:

| Column | Meaning |
|---|---|
| `rank_in_category` | position within the category by popularity |
| `twitter_screen_name`, `twitter_user_id`, `twitter_name`, `followers_count` | from `UsersWithWikipedia.csv` |
| `socialvec_popularity_rank` | position in the full popularity list |
| `category_source` | `dbpedia`, `wikidata-occupation` (a person's occupation decided) or `wikidata` (instance-of or industry decided) |
| `dbpedia_types`, `wikidata_items`, `wikidata_instance_of`, `wikidata_occupation` | the evidence behind the decision |
| `in_original_list` | the account is in the category's sheet of the original file |
| `use` | 1 if the account is one of the original manually validated entities of that category |

## 3. Results

Out of 31,502 accounts, 9,917 received at least one category, and 94 received more than one.
Decisions by source: 4,668 DBpedia, 4,692 Wikidata occupation, 654 Wikidata instance-of or industry.

| Category | Accounts found | Kept | Popularity rank of the 100th | of the 200th | of the 1,000th | Source: DBpedia / Wikidata | Original list recovered | Validated top 20 recovered | Top-100 overlap with original |
|---|---|---|---|---|---|---|---|---|---|
| Musical artists | 3,073 | 1,000 | 491 | 1,087 | 7,779 | 637 / 363 | 76 / 94 | 15 / 20 | 52 |
| News outlets | 478 | 478 | 3,209 | 8,702 | – | 242 / 236 | 59 / 88 | 16 / 20 | 43 |
| Comedians | 411 | 411 | 2,192 | 7,177 | – | 222 / 189 | 89 / 93 | 19 / 20 | 36 |
| Politicians | 751 | 751 | 2,241 | 5,623 | – | 686 / 65 | 88 / 89 | 18 / 19 | 56 |
| TV stations | 408 | 408 | 5,519 | 12,958 | – | 380 / 28 | 84 / 91 | 16 / 20 | 42 |
| Actors | 2,178 | 1,000 | 506 | 1,253 | 9,646 | 6 / 994 | 83 / 98 | 16 / 20 | 32 |
| TV shows | 488 | 488 | 2,927 | 7,345 | – | 474 / 14 | 75 / 93 | 17 / 20 | 38 |
| Sports teams | 352 | 352 | 5,402 | 16,315 | – | 214 / 138 | 89 / 95 | 19 / 20 | 64 |
| Fashion | 90 | 90 | – | – | – | 0 / 90 | 9 / 16 | 9 / 20 | 9 |
| Journalists | 1,264 | 1,000 | 2,318 | 4,403 | 23,805 | 35 / 965 | 81 / 95 | 17 / 20 | 20 |
| TV hosts | 304 | 304 | 5,434 | 15,860 | – | 6 / 298 | 14 / 22 | 11 / 20 | 4 |
| Films | 70 | 70 | – | – | – | 69 / 1 | 10 / 17 | 9 / 18 | 10 |
| Food chains | 107 | 107 | 27,196 | – | – | 0 / 107 | 23 / 23 | 19 / 20 | 23 |
| Car makers | 40 | 40 | – | – | – | 0 / 40 | 15 / 17 | 15 / 19 | 15 |

**How to read the table:**
- *Popularity rank of the n-th:* how far down SocialVec's popularity list the category's n-th entity sits. For example,
  the 200th sports team is the 16,315th most popular account.
- *Original list recovered:* entities of the original 100-entity list that appear in the new list, out of those found in
  Wikidata.
- *Validated top 20 recovered:* the same, for the 20 manually validated entities.
- *Source:* counted over the kept accounts.

**What the table shows:**
- **People:** person categories rely mostly on Wikidata, because today's DBpedia gives many people only the type
  `Person`. Exceptions are politicians and comedians, where DBpedia types are common.
- **Organizations:** TV stations, TV shows, sports teams and films come mostly from DBpedia.
- **Facets:** news, fashion, food chains and car makers come mostly or entirely from Wikidata.

## 4. Quality

No manual validation was done, so precision is not measured. Spot checks of each list show mostly correct entities,
including in the newly added ranks.

**Top of the lists and new entities (ranks 101–200):**

| Category | Top 8 | Examples from ranks 101–200 |
|---|---|---|
| Musical artists | Taylor Swift, ye, Rihanna, Justin Bieber, Miley Ray Cyrus, KATY PERRY, Wiz Khalifa, Lady Gaga | hayley from Paramore 🌺, Lil B THE BASEDGOD, Love Regenerator, Dolly Parton, caliginous boat, kesha, Kodak Black, DJ KHALED |
| News outlets | CNN Breaking News, The New York Times, CNN, The Associated Press, The Onion, The Wall Street Journal, HuffPost, The Washington Post | Digg, Fox News Alert, NBC News Health, New York Times Opinion, NOISEY, Roll Call, NYT Politics, The Young Turks |
| TV stations | CNN Breaking News, ESPN, CNN, Fox News, BBC News (World), ABC News, Breaking News, MTV | CNN en Español, Nat Geo WILD, TCM, NBC10 Philadelphia, Bloomberg TV, NBCWashington, ABC13 Houston, Showtime |
| TV shows | SportsCenter, The Daily Show, E! News, Fallon Tonight, Game of Thrones, College GameDay, Saturday Night Live - SNL, TODAY | VeryBritishProblems, Gossip Girl, The Last Word, How To Get Away ABC, Modern Family, Shameless on Showtime, CBS This Morning, Shark Week |
| Sports teams | Los Angeles Lakers, Dallas Cowboys, New England Patriots, New York Yankees, Golden State Warriors, Chicago Cubs, Red Sox, Pittsburgh Steelers | OpTic Gaming™, Notre Dame Football, Phoenix Suns, #DUUUVAL, Sacramento Kings, Ohio State Football, Stewart-Haas Racing, Buffalo Sabres |
| Food chains | Wendy's, Taco Bell, Chipotle⁷, Subway®, Chick-fil-A, Inc., Whataburger®, Dunkin', Buffalo Wild Wings | Graeter's Ice Cream, Buca di Beppo, Carrabba's, Which Wich, Logan's Roadhouse, Marco's Pizza, Famous Dave's |

**Known errors and their sources:**
- **People with side careers.** Kim Kardashian, Khloé Kardashian, Oprah Winfrey and Ariana Grande appear among the top
  actors. DBpedia types them only as `Person`, and an acting occupation comes first among their Wikidata occupations.
  This is the main precision problem for the actor list (top-100 overlap with the original list: 32).
- **Errors in DBpedia itself.** Elon Musk is typed `Politician` in DBpedia, so he appears among the top politicians.
- **Secondary accounts.** Wikidata attaches some secondary accounts to the main entity's item, so they inherit its type.
  Examples are a team's PR account (*Warriors PR*), presidential accounts (*President Obama* next to *Barack Obama*),
  and breaking-news accounts. The original lists contained such pairs too (CNN and CNN Breaking News).
- **Errors in the original lists.** The original TV show list includes video games (*NBA 2K*, *Animal Crossing*) and
  radio programs, which the new rules correctly leave out. They count as "not recovered" in the table.
- **Coverage.** 12,159 accounts (39%) could not be linked to Wikidata and get no
  category. This mostly affects less popular accounts. Username matching is exact, so accounts whose Wikidata username
  differs in case or has changed are missed.
- **Small categories.**
  - *Films:* accounts are often promotional handles that are not linked to the film's item.
  - *Fashion:* brands are often typed only as companies.
  - *Food chains and car makers:* few such accounts have a Wikipedia page among SocialVec's popular accounts.

## 5. Reproducing

```bash
.venv/Scripts/python build_extended_categories.py --top 1000                   # uses the cache in data/wikidata/
.venv/Scripts/python build_extended_categories.py --top 1000 --person-rule all # alternative person rule
```

Deleting `data/wikidata/` repeats all lookups: about 105 Wikidata queries and 100 DBpedia queries, roughly 15 minutes.
Both endpoints are free; please keep the pacing in the script.
