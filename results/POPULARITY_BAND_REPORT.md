# Personalization beyond the most popular entities

*Extension of Section 4 of "Social Knowledge for Cross-Domain User Preference Modeling". All numbers below are produced by `python run_popularity_band.py --start 81 --end 100`; raw outputs are in `results/popularity_band_81_100/`.*

## Motivation

The main experiments (Table 2) rank the 20 most popular entities of each category. For such entities, popularity is a strong non-personalized baseline: most users follow several of them, and those they follow tend to be the best known. Recommendation is most useful, however, for entities that are not universally known. We therefore ask whether social user modeling still predicts user preferences when the candidates are drawn from further down the popularity distribution, where popularity provides less guidance.

## Experimental setup

**Candidate entities.** For each category we start from the list of typed entities from which the original candidates were selected, comprising up to 100 entities per category that were aligned with the category's DBpedia type. We sort these entities by their number of Twitter followers and take those ranked 81–100 as candidates. Only 9 of the 14 categories list 100 entities, so the remaining five (TV hosts, Films, Fashion, Food chains and Car makers) are excluded. No category lists more than 100 entities, which also rules out a band further down, such as ranks 100–120. Three of the 180 band entities have no SocialVec embedding and are dropped: one from Journalists and two from TV shows. That leaves 19 and 18 candidates in those categories and 20 in the others. Follower counts in the selected bands range from about 20K to 1.5M, compared with roughly 20K to 116M for the original candidates. The selected entities are listed in the appendix.

**Users.** We use the same users as in the main experiments: the 11,153 users sampled for the 14 categories. For each category we evaluate the users sampled for that category who follow at least one band entity, since average precision is undefined for users with no relevant candidate. This yields 198 to 749 users per category and 3,798 in total.

**User representation.** As in the main experiments, a user is represented by the average SocialVec embedding of the entities they follow. Here we exclude *all* entities listed for the target category: the 100 typed entities, including the band candidates, and the category's original 20 candidates. The representation therefore carries no direct information about the user's choices in the target domain.

**Evaluation.** As in Section 4, the candidates are ranked by cosine similarity to the user representation, and the band entities the user follows are the relevant items. We report MAP per category and its unweighted mean over categories. The popularity baseline ranks the candidates by follower count. As a reference point, we also report the expected AP of a uniformly random ranking. For N candidates and R relevant items, it equals (H_N + (R−1)/(N−1)·(N−H_N))/N, where H_N is the N-th harmonic number. We compare the two methods with a two-sided Wilcoxon signed-rank test on per-user AP; the overall test pools all users.

**Matched control.** Users who follow less popular entities may differ from the general population. We therefore also rank the original 20 candidates of each category (ranks 1–20 in Table 1) *for the same users and with the same user representations*. Differences between the two conditions are then due to the candidates alone.

## Results

**Table A.** MAP for ranking the original top-20 entities (1–20) and the entities ranked 81–100 by follower count. Both conditions use the same users and the same user representations, which exclude all entities of the target category. Rel./user is the average number of candidates each user follows. Δ is the relative gain of social similarity over popularity. p is from a Wilcoxon signed-rank test of social similarity against popularity, computed separately for each band.

| Category | Users | Rel./user (1–20) | Popularity (1–20) | Social (1–20) | Δ (1–20) | p (1–20) | Rel./user (81–100) | Random (81–100) | Popularity (81–100) | Social (81–100) | Δ (81–100) | p (81–100) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Musical artists | 472 | 8.1 | 0.518 | 0.655 | +26% | < 0.001 | 2.4 | 0.240 | 0.349 | 0.520 | +49% | < 0.001 |
| News outlets | 426 | 9.4 | 0.604 | 0.731 | +21% | < 0.001 | 2.5 | 0.244 | 0.233 | 0.379 | +62% | < 0.001 |
| Comedians | 277 | 9.0 | 0.617 | 0.678 | +10% | < 0.001 | 1.8 | 0.214 | 0.363 | 0.267 | −26% | < 0.001 |
| Politicians | 641 | 12.3 | 0.734 | 0.803 | +9% | < 0.001 | 5.6 | 0.377 | 0.359 | 0.438 | +22% | < 0.001 |
| TV stations | 198 | 6.4 | 0.527 | 0.672 | +27% | < 0.001 | 1.4 | 0.198 | 0.186 | 0.413 | +122% | < 0.001 |
| Actors | 579 | 7.4 | 0.560 | 0.696 | +24% | < 0.001 | 3.2 | 0.274 | 0.305 | 0.483 | +58% | < 0.001 |
| TV shows | 216 | 5.4 | 0.429 | 0.673 | +57% | < 0.001 | 1.4 | 0.215 | 0.325 | 0.398 | +23% | 0.011 |
| Sports teams | 240 | 6.9 | 0.441 | 0.576 | +31% | < 0.001 | 2.5 | 0.245 | 0.260 | 0.436 | +68% | < 0.001 |
| Journalists | 749 | 11.4 | 0.666 | 0.720 | +8% | < 0.001 | 4.1 | 0.325 | 0.299 | 0.385 | +29% | < 0.001 |
| **Overall** | 3,798 | | 0.566 | 0.689 | +22% | < 0.001 | | 0.259 | 0.298 | 0.413 | +39% | < 0.001 |

For the original top-20 candidates, the gains on this subset of users (+22% overall) are in line with Table 2. Social similarity significantly outperforms popularity in every category (all p < 10⁻⁹; overall p < 10⁻¹⁶⁰).

For the entities ranked 81–100, MAP is lower for both methods. Users follow fewer of these candidates (1.4–5.6 per user, compared with 5.4–12.3), so the task is harder. Even a random ranking is expected to score only 0.259 on average. Popularity loses most of its value in this range: its overall MAP of 0.298 is only 0.039 above chance, and in four categories (News outlets, Politicians, TV stations and Journalists) it scores below a random ranking. Among entities with similar, moderate follower counts, worldwide popularity says little about which of them a particular user follows.

Social similarity, in contrast, remains well above chance in every category: 0.413 overall, 0.154 above the random ranking. It outperforms popularity in 8 of the 9 categories, and its relative gain grows from +22% for the top-20 entities to +39% for ranks 81–100. The largest gains are for TV stations (+122%), sports teams (+68%), news outlets (+62%) and actors (+58%). In absolute terms the advantage over popularity is similar in both conditions: +0.115 MAP in the tail and +0.123 at the top. The larger relative gain in the tail therefore comes mainly from the weaker popularity baseline. The difference is significant overall (p < 10⁻¹⁰⁰) and in every category except TV shows (p = 0.011), which is not significant after Bonferroni correction for nine comparisons.

The exception is the Comedians category, where popularity outperforms social similarity (0.363 vs. 0.267; p < 0.001). In this band, follows are concentrated on the two entities with the most followers, Dean Obeidallah and Hal Sparks, who are followed by 29% and 13% of the evaluated users respectively. Both are known mainly as political commentators rather than stand-up comedians, so their audience is broader than the category suggests. Popularity places these two entities first and second, whereas social similarity ranks them only 10th and 12th on average. It favors stand-up comedians that are less frequently followed.

Overall, these results indicate that the benefit of social user modeling is not limited to the most widely known entities. When candidates are less popular, the one-fits-all popularity ranking approaches chance level, while the social representation of the user still provides a substantial and consistent signal for personalization.

## Robustness

*Excluding only the candidates.* If the user representation excludes only the 20 band candidates, keeping the category's other entities as in the main experiments, the results are nearly unchanged: overall social MAP is 0.415 vs. 0.298 for popularity (+39%). Keeping the other entities of the target domain thus adds little information about preferences among its less popular entities.

*All users.* The users sampled for a category were selected as followers of its top-20 entities, so they may be atypical of that category's audience. We therefore repeat the matched comparison of Table A on *all* experiment users rather than only those sampled for the category. Each of the 11,153 users is evaluated in every category in which they follow at least one band entity *and* at least one of the original 20 candidates. As in Table A, the user representation excludes every entity listed for the target category, and both bands are ranked for exactly the same users. This yields 36,739 user–category pairs from 10,039 distinct users, about 3.7 categories per user. Only 10% of these pairs come from users sampled for the category in question, so this test is largely independent of the sample used in Table A. The results are shown in Table B.

**Table B.** As Table A, evaluated on all experiment users who follow at least one entity from each band in the category. Users counts user–category pairs. The per-category p-values are from a Wilcoxon signed-rank test in which each user appears once. The overall test pools all pairs, so a user who appears in several categories contributes several pairs.

| Category | Users | Rel./user (1–20) | Popularity (1–20) | Social (1–20) | Δ (1–20) | p (1–20) | Rel./user (81–100) | Random (81–100) | Popularity (81–100) | Social (81–100) | Δ (81–100) | p (81–100) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Musical artists | 5,922 | 6.1 | 0.398 | 0.561 | +41% | < 0.001 | 2.3 | 0.238 | 0.330 | 0.493 | +49% | < 0.001 |
| News outlets | 4,426 | 8.5 | 0.560 | 0.698 | +25% | < 0.001 | 2.6 | 0.248 | 0.235 | 0.385 | +64% | < 0.001 |
| Comedians | 2,528 | 7.5 | 0.567 | 0.625 | +10% | < 0.001 | 1.7 | 0.209 | 0.373 | 0.288 | −23% | < 0.001 |
| Politicians | 6,348 | 10.9 | 0.693 | 0.779 | +12% | < 0.001 | 5.3 | 0.366 | 0.359 | 0.448 | +25% | < 0.001 |
| TV stations | 2,585 | 5.4 | 0.527 | 0.629 | +19% | < 0.001 | 1.4 | 0.197 | 0.186 | 0.362 | +95% | < 0.001 |
| Actors | 6,064 | 6.0 | 0.515 | 0.629 | +22% | < 0.001 | 3.1 | 0.270 | 0.294 | 0.464 | +58% | < 0.001 |
| TV shows | 2,769 | 4.3 | 0.433 | 0.611 | +41% | < 0.001 | 1.4 | 0.213 | 0.325 | 0.377 | +16% | < 0.001 |
| Sports teams | 1,585 | 4.6 | 0.350 | 0.499 | +42% | < 0.001 | 2.1 | 0.229 | 0.231 | 0.451 | +95% | < 0.001 |
| Journalists | 4,512 | 9.2 | 0.570 | 0.677 | +19% | < 0.001 | 3.6 | 0.303 | 0.288 | 0.361 | +25% | < 0.001 |
| **Overall** | 36,739 | | 0.513 | 0.634 | +24% | < 0.001 | | 0.252 | 0.291 | 0.403 | +38% | < 0.001 |

Table B confirms the findings of Table A on a population roughly ten times larger.
- **Top 20:** social similarity outperforms popularity in all nine categories, by +24% overall.
- **Ranks 81–100:** the overall gain rises to +38% (0.403 vs. 0.291).
- **Popularity near chance:** at ranks 81–100, popularity again scores below a random ranking for news outlets, politicians, TV stations and journalists, and is at chance level for sports teams (0.231 vs. 0.229).
- **Significance:** with the larger samples, every per-category difference is significant at p < 0.001, including TV shows (+16%), which did not survive Bonferroni correction in Table A.
- **Comedians:** this is again the only category in which popularity outperforms social similarity (−23%).

Popularity MAP on the top-20 candidates is lower here than in Table A (0.513 vs. 0.566), because users who were not sampled for a category follow fewer of its top entities (4.3–10.9 per user, compared with 5.4–12.3). Social similarity retains its advantage regardless.

## LLM ranking of less popular entities

We also applied the LLM protocol of the paper's Table 3 to the 81–100 candidates, using GPT-6 Luna with reasoning effort `low` and the authors' prompts. To limit cost, we evaluated a stratified random sample of 30 users per category (270 users) from the Table A population.
- **Personalized prompts:** they name 12 or 25 entities sampled from the user's closed-world profile, or 50 accounts sampled from everything the user follows. As in the main experiment, *every* listed entity of the target category is removed before sampling, so the prompts carry no direct information about the target domain. The candidates are listed by follower count.
- **Baseline:** the non-personalized prompt lists the candidates in spreadsheet order.
- **Comparison:** popularity, random and social-similarity scores are those of Table A for the same users. Note that the social representation uses all of the user's entities, while an LLM prompt lists at most 50 names: on average 11.7, 21.8 and 29.0 for k = 12, 25 and 50, since only accounts with known names are listed.

**Table C.** MAP for ranks 81–100 on 270 sampled users (30 per category).

| Category | Users | Random | Popularity | Social (all entities) | Social k = 12 | Social k = 25 | LLM baseline | LLM k = 12 | LLM k = 25 | LLM k = 50 |
|---|---|---|---|---|---|---|---|---|---|---|
| Musical artists | 30 | 0.252 | 0.453 | 0.611 | 0.489 | 0.498 | 0.316 | 0.395 | 0.369 | 0.489 |
| News outlets | 30 | 0.253 | 0.267 | 0.396 | 0.331 | 0.353 | 0.268 | 0.532 | 0.546 | 0.533 |
| Comedians | 30 | 0.217 | 0.290 | 0.281 | 0.222 | 0.243 | 0.154 | 0.290 | 0.272 | 0.351 |
| Politicians | 30 | 0.386 | 0.328 | 0.463 | 0.413 | 0.412 | 0.540 | 0.484 | 0.546 | 0.519 |
| TV stations | 30 | 0.191 | 0.149 | 0.444 | 0.338 | 0.334 | 0.236 | 0.353 | 0.463 | 0.666 |
| Actors | 30 | 0.268 | 0.320 | 0.518 | 0.464 | 0.441 | 0.271 | 0.353 | 0.403 | 0.487 |
| TV shows | 30 | 0.213 | 0.276 | 0.407 | 0.371 | 0.374 | 0.334 | 0.485 | 0.456 | 0.570 |
| Sports teams | 30 | 0.252 | 0.269 | 0.516 | 0.292 | 0.284 | 0.222 | 0.302 | 0.315 | 0.426 |
| Journalists | 30 | 0.324 | 0.303 | 0.423 | 0.343 | 0.343 | 0.392 | 0.424 | 0.423 | 0.467 |
| **Overall** | 270 | 0.262 | 0.295 | 0.451 | 0.362 | 0.365 | 0.304 | 0.402 | 0.421 | 0.501 |

*Social k = 12* and *Social k = 25* build the user embedding from exactly the accounts listed in the LLM's k = 12 and k = 25 prompts: 11.7 and 21.8 accounts on average, since some users have fewer eligible accounts.

Paired Wilcoxon tests over the 270 users:

| Comparison | LLM baseline | LLM k = 12 | LLM k = 25 | LLM k = 50 |
|---|---|---|---|---|
| vs. popularity | p = 0.57 | p < 0.001 | p < 0.001 | p < 0.001 |
| vs. social similarity (all entities) | p < 0.001 (lower) | p = 0.051 | p = 0.30 | p = 0.016 (higher) |
| vs. social similarity, same accounts | | p = 0.006 (higher) | p < 0.001 (higher) | |

**Findings.**
- **Without user information, the LLM is no better than popularity.** In this range of the popularity distribution, the non-personalized LLM ranking (0.304) does not differ from popularity (0.295, p = 0.57). Both are barely above a random ranking (0.262), consistent with Table A. The model's general knowledge of these entities does not identify which ones users follow.
- **With user information, the LLM personalizes well.** Naming 12 entities from the user's profile raises MAP to 0.402, and 50 accounts to 0.501. Every personalized condition clearly outperforms popularity (p < 0.001).
- **Given the same evidence, the LLM ranks better.** When both methods use exactly the accounts in the LLM's prompt, the LLM leads at both profile sizes:
  - *12 accounts:* 0.402 vs. 0.362 (p = 0.006).
  - *25 accounts:* 0.421 vs. 0.365 (p < 0.001).

  In both cases the LLM is ahead in 7 of 9 categories, and the embedding is ahead for musical artists and actors. The embedding barely improves from 12 to 25 of these accounts. Its overall advantage therefore depends on using the user's full profile of followed entities.
- **The LLM is competitive with social embeddings built from the full profile.**
  - *12 and 25 entities:* with a short profile, the LLM is statistically indistinguishable from the social embedding built from all of the user's entities (0.402 and 0.421 vs. 0.451; p = 0.051 and 0.30).
  - *50 accounts:* with 50 sampled accounts it scores higher (0.501 vs. 0.451, p = 0.016).
  - *Where each method does better:* the methods are complementary.
    - *LLM better:* news outlets (0.53–0.55 vs. 0.40 at every k) and TV shows. For TV stations, the LLM is better only with 50 accounts (0.666 vs. 0.444).
    - *Social similarity better:* musical artists, sports teams and actors, at every k.
  - *A possible explanation:* the candidates in the LLM's strongest categories have informative names, such as regional TV stations, partisan outlets and party accounts. A language model can relate these to the user's other interests directly from their names.
- **Caveat.** With 30 users per category, per-category MAP has a standard error of about 0.05, so only large per-category differences should be read as meaningful.

Full details of the LLM setup, including how sensitive the LLM is to the order of the candidates, are in `results/LLM_REPORT.md`.

## Limitations

- **Popularity range.** The entity lists contain at most 100 entities per category, so we could not evaluate the planned band of ranks 100–120, or any less popular band. Ranks 81–100 are the least popular entities available. They span 20K–1.5M followers, depending on the category.
- **Category labels.** The 20 original candidates per category were manually validated, but the rest of each list was not. The 81–100 bands therefore contain some entities whose type does not match the category. Examples include esports organizations and *Star Trek* among sports teams, state party accounts among TV stations, and a physician among musical artists. We kept the lists as they are, so as not to select entities based on the outcome.
- **User sampling.** Users were originally sampled as followers of the top-20 entities, not of the band entities. The evaluated users are therefore those who follow both popular and less popular entities in a category, and they may not represent typical followers of less popular entities.
- **Fewer relevant items.** Users follow 1.4–5.6 band candidates on average. With so few relevant items, per-user AP is more variable, so per-category estimates with 200–300 users (TV stations, TV shows, sports teams, comedians) are less precise.

## Appendix: candidate entities (ranks 81–100 by follower count)

Entities marked * have no SocialVec embedding and were excluded.

| Category | Followers | Entities |
|---|---|---|
| Musical artists | 23K–1.48M | FINALLEVEL, TEYANATAYLOR, JanelleMonae, springsteen, DojaCat, IAMSUMMERWALKER, MoneyBaggYo, KidRock, BarbraStreisand, tmorello, Polo_Capalot, AriLennox, TedNugent, KayaJones, ScottGottliebMD, richardmarx, NancySinatra, johnrich, Travistritt, _DionneWarwick |
| News outlets | 74K–325K | HARRISFAULKNER, npratc, ErinBurnett, TwitchyTeam, BNODesk, BlazeTV, dcexaminer, TexasTribune, HeatherChilders, RealClearNews, Mediaite, townhallcom, dagenmcdowell, KarenAttiah, BNONews, goldietaylor, biannagolodryga, JohnBerman, newsmax, AnaCabrera |
| Comedians | 22K–106K | DeanObeidallah, HalSparks, natebargatze, melissavcomedy, saraschaefer1, kibblesmith, GaryGulman, JAdomian, dopequeenpheebs, SteveHofstetter, LilyTomlin, blainecapatch, anylaurie16, tomshillue, aamer_rahman, sammorril, JenaFriedman, karencheee, jackiekashian, yannispappas |
| Politicians | 20K–328K | CaptMarkKelly, brianschatz, harrisonjaime, RepJayapal, JohnWDean, RepJeffries, K_JeanPierre, Sifill_LDF, HawleyMO, MarshaBlackburn, RonaldKlain, RepThomasMassie, ossoff, RepRaskin, CongressmanRaja, JohnFetterman, jrpsaki, CoriBush, JoshShapiroPA, nikkifried |
| TV stations | 18K–214K | 12News, WESH, LogoTV, STARZ, tcm, KXAN_News, WeatherNation, HawaiiNewsNow, ABC7News, HBODocs, CNNPR, SNYtv, SportsNetLA, NBCSCeltics, FlaDems, MSGNetworks, NBCSWashington, SYFYWIRE, NCDemParty, NewYorkGOP |
| Actors | 101K–464K | MichaelRapaport, PedroPascal1, YNB, PattyArquette, RealDeanCain, tarastrong, SachaBaronCohen, carlreiner, Chris_Meloni, AdamBaldwin, BradleyWhitford, jonvoight, halleberry, TomArnold, KristySwansonXO, EllenBarkin, kenolin1, thekjohnston, RoArquette, morgfair |
| TV shows | 46K–254K | Jeopardy, CBSSunday, dragonage, trailerparkboys, BravoTopChef, SquawkCNBC, AncientAliens, SeinfeldTV, drunkhistory, grownish, QueenSugarOWN, TheOrville, suda_51, MST3K, cspanwj, CobraKaiSeries, KellyClarksonTV\*, OfficialRWBY\*, DemConvention, Svengoolie |
| Sports teams | 221K–588K | TeamLiquid, RaysBaseball, Athletics, Mariners, EvilGeniuses, NYIslanders, BlueJacketsNHL, GoldenKnights, StarTrek, clgaming, NYCFC, Luminosity, FlaPanthers, Marlins, usahockey, FloridaGators, GoHeels, YankeesPR, Complexity, trentylocks |
| Journalists | 51K–147K | carlbernstein, pdoocy, mtracey, sbg1, AriBerman, TaylorLorenz, JonLemire, CBS_Herridge\*, ElieNYC, JenGriffinFNC, JasonLeopold, CarolLeonnig, adamhousley, kyledcheney, willsommer, JDiamond1, clarissaward, PaulaReidCBS, AndrewFeinberg, ejeancarroll |

## Reproducing

```bash
.venv/Scripts/python run_popularity_band.py --start 81 --end 100
.venv/Scripts/python run_llm_experiments.py --parts band      # Table C (needs an OpenAI key)
```

Table C's data is in `results/llm/band_per_user.csv.gz` and `results/llm/band_by_category.csv`.

Outputs in `results/popularity_band_81_100/`:

| File | Contents |
|---|---|
| `comparison_tables.md` | Tables A and B: matched top-20 vs. 81–100 comparisons |
| `matched_*.csv` | The data behind Tables A and B |
| `tables.md` | All five conditions, unmatched, with exact p-values |
| `summary_*.csv` | One summary table per condition |
| `per_user_ap.csv.gz` | Per-user AP for each condition, with random-ranking AP |
| `band_entities.csv` | The candidates, their follower counts and SocialVec coverage |
