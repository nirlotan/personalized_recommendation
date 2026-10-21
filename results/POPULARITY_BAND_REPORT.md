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

**Table A.** MAP for ranking the original top-20 entities (1–20) and the entities ranked 81–100 by follower count. Both conditions use the same users and the same user representations, which exclude all entities of the target category. Rel./user is the average number of candidates each user follows. Δ is the relative gain of social similarity over popularity. p is from a Wilcoxon signed-rank test (social vs. popularity, 81–100).

| Category | Users | Rel./user (1–20) | Popularity (1–20) | Social (1–20) | Δ (1–20) | Rel./user (81–100) | Random (81–100) | Popularity (81–100) | Social (81–100) | Δ (81–100) | p (81–100) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Musical artists | 472 | 8.1 | 0.518 | 0.655 | +26% | 2.4 | 0.240 | 0.349 | 0.520 | +49% | < 0.001 |
| News outlets | 426 | 9.4 | 0.604 | 0.731 | +21% | 2.5 | 0.244 | 0.233 | 0.379 | +62% | < 0.001 |
| Comedians | 277 | 9.0 | 0.617 | 0.678 | +10% | 1.8 | 0.214 | 0.363 | 0.267 | −26% | < 0.001 |
| Politicians | 641 | 12.3 | 0.734 | 0.803 | +9% | 5.6 | 0.377 | 0.359 | 0.438 | +22% | < 0.001 |
| TV stations | 198 | 6.4 | 0.527 | 0.672 | +27% | 1.4 | 0.198 | 0.186 | 0.413 | +122% | < 0.001 |
| Actors | 579 | 7.4 | 0.560 | 0.696 | +24% | 3.2 | 0.274 | 0.305 | 0.483 | +58% | < 0.001 |
| TV shows | 216 | 5.4 | 0.429 | 0.673 | +57% | 1.4 | 0.215 | 0.325 | 0.398 | +23% | 0.011 |
| Sports teams | 240 | 6.9 | 0.441 | 0.576 | +31% | 2.5 | 0.245 | 0.260 | 0.436 | +68% | < 0.001 |
| Journalists | 749 | 11.4 | 0.666 | 0.720 | +8% | 4.1 | 0.325 | 0.299 | 0.385 | +29% | < 0.001 |
| **Overall** | 3,798 | | 0.566 | 0.689 | +22% | | 0.259 | 0.298 | 0.413 | +39% | < 0.001 |

For the original top-20 candidates, the gains on this subset of users (+22% overall) are in line with Table 2. All differences between social similarity and popularity in the 1–20 condition are significant (p < 0.001).

For the entities ranked 81–100, MAP is lower for both methods. Users follow fewer of these candidates (1.4–5.6 per user, compared with 5.4–12.3), so the task is harder. Even a random ranking is expected to score only 0.259 on average. Popularity loses most of its value in this range: its overall MAP of 0.298 is only 0.039 above chance, and in four categories (News outlets, Politicians, TV stations and Journalists) it scores below a random ranking. Among entities with similar, moderate follower counts, worldwide popularity says little about which of them a particular user follows.

Social similarity, in contrast, remains well above chance in every category: 0.413 overall, 0.154 above the random ranking. It outperforms popularity in 8 of the 9 categories, and its relative gain grows from +22% for the top-20 entities to +39% for ranks 81–100. The largest gains are for TV stations (+122%), sports teams (+68%), news outlets (+62%) and actors (+58%). In absolute terms the advantage over popularity is similar in both conditions: +0.115 MAP in the tail and +0.123 at the top. The larger relative gain in the tail therefore comes mainly from the weaker popularity baseline. The difference is significant overall (p < 10⁻¹⁰⁰) and in every category except TV shows (p = 0.011), which is not significant after Bonferroni correction for nine comparisons.

The exception is the Comedians category, where popularity outperforms social similarity (0.363 vs. 0.267; p < 0.001). In this band, follows are concentrated on the two entities with the most followers, Dean Obeidallah and Hal Sparks, who are followed by 29% and 13% of the evaluated users respectively. Both are known mainly as political commentators rather than stand-up comedians, so their audience is broader than the category suggests. Popularity places these two entities first and second, whereas social similarity ranks them only 10th and 12th on average. It favors stand-up comedians that are less frequently followed.

Overall, these results indicate that the benefit of social user modeling is not limited to the most widely known entities. When candidates are less popular, the one-fits-all popularity ranking approaches chance level, while the social representation of the user still provides a substantial and consistent signal for personalization.

## Robustness

*Excluding only the candidates.* If the user representation excludes only the 20 band candidates, keeping the category's other entities as in the main experiments, the results are nearly unchanged: overall social MAP is 0.415 vs. 0.298 for popularity (+39%). Keeping the other entities of the target domain thus adds little information about preferences among its less popular entities.

*All users.* If we evaluate all 11,153 experiment users who follow at least one band entity in a category, rather than only the users sampled for that category, we obtain 39,437 user–category pairs. The pattern is the same: social similarity scores 0.403 vs. 0.287 for popularity (+40%), compared with +24% on the top-20 candidates for the same users (0.634 vs. 0.513). Comedians again remain the only category where popularity is better (−23%).

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
```

Outputs in `results/popularity_band_81_100/`:

| File | Contents |
|---|---|
| `tables.md` | All five conditions, with exact p-values |
| `summary_*.csv` | One summary table per condition |
| `per_user_ap.csv.gz` | Per-user AP for each condition, with random-ranking AP |
| `band_entities.csv` | The candidates, their follower counts and SocialVec coverage |
