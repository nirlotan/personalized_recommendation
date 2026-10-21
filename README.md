# Social Knowledge for Cross-Domain User Preference Modeling

This repository contains the implementation and experimental code for research on cross-domain personalized recommendation using social network embeddings. The work demonstrates that user preferences can be effectively modeled and predicted across multiple topical domains using large-scale social data from Twitter/X.

## 🎯 Overview

Traditional recommender systems face significant challenges:
- **Cold-start problem**: Require substantial user feedback to model preferences accurately
- **Domain-specific**: User feedback typically doesn't generalize across domains (e.g., movies → music)

This research addresses these limitations by leveraging **social embeddings** learned from Twitter network data to enable **cross-domain personalization** even without domain-specific user feedback.

## 🔑 Key Contributions

### 1. **Social User Modeling via Entity Embeddings**
- Projects users into a 100-dimensional social embedding space based on popular accounts they follow
- Uses pre-trained embeddings of ~200K popular Twitter entities
- Enables inductive user modeling: represent any user based on entities they favor

### 2. **Cross-Domain Preference Prediction**
- Demonstrates that user preferences generalize across 14 different topical domains:
  - Musical artists, News outlets, Comedians, Politicians
  - TV stations, Actors, TV shows, Sports teams
  - Fashion brands, Journalists, TV hosts, Films
  - Food chains, Car manufacturers
- Achieves **22% average improvement** in Mean Average Precision (MAP) over popularity-based ranking

### 3. **Efficient Personalization with Limited Data**
- Shows effective personalization with as few as **10-20 entities** per user
- Performance converges at ~50 entities, achieving 98%+ of optimal results
- Practical for cold-start scenarios via simple user questionnaires

### 4. **Socio-Demographic Analysis**
- Reveals correlations between social embeddings and user traits:
  - Gender, age, ethnicity, education level, political affiliation
- Provides insights into how social factors drive cross-domain preferences
- Visualizes demographic profiles of entity followers using radar plots

### 5. **LLM-Based Social Modeling**
- Demonstrates that GPT-4o can replicate personalization results
- Users described as lists of favorite entities (instead of embeddings)
- Achieves **13% MAP improvement** with just 12 entities, **23% with 50 entities**

## 📊 Experimental Dataset

The experiments use a carefully constructed dataset:
- **~12,000 Twitter users** from a 40K random sample (US-based)
- **14 topical categories** with 20 popular entities each (280 entities total)
- **~1,000 users per category** who follow at least one entity in that category
- Users follow **~2K accounts** on average
- Entities range from 1.4 to 11.2 follows per user (average: 5.5)

### Example Entities by Category
- **Musical Artists**: Taylor Swift, Rihanna, Kanye West, Ariana Grande
- **Politicians**: Barack Obama, Bernie Sanders, Elizabeth Warren, Ron DeSantis
- **News Outlets**: CNN, NYT, Fox News, The Economist
- **Sports Teams**: Golden State Warriors, New York Yankees, Green Bay Packers

## 🛠️ Technical Approach

### Social Embedding Generation
1. Start with 200K pre-trained entity embeddings from SocialVec (learned using Word2Vec skip-gram on Twitter network)
2. For each user, collect embeddings of entities they follow
3. Average entity embeddings to create user profile vector
4. Entities co-followed by users reside close in embedding space

### Preference Prediction
1. Given user embedding, rank candidate entities by **cosine similarity**
2. Remove target domain entities from user profile (cross-domain evaluation)
3. Evaluate via **link prediction**: entities user actually follows = ground truth

### Evaluation Metrics
- **Mean Average Precision (MAP)**: Primary metric
- **Baseline**: Popularity-based ranking (by follower count)
- **Two evaluation modes**:
  - **Full**: User profile from all ~2K followed accounts
  - **Closed-world**: Only using entities from the 14 experimental categories

## 📈 Key Results

| Domain | Popularity MAP | Social Embeddings MAP | Improvement |
|--------|----------------|----------------------|-------------|
| Musical Artists | 0.520 | 0.633 | +22% |
| TV Shows | 0.365 | 0.635 | +74% |
| Films | 0.221 | 0.399 | +81% |
| Actors | 0.521 | 0.647 | +24% |
| News Outlets | 0.581 | 0.692 | +19% |
| **Overall** | **0.468** | **0.571** | **+22%** |

### Performance vs. Sample Size
- **10 entities**: 93% of optimal MAP
- **20 entities**: 96% of optimal MAP  
- **50 entities**: 98%+ of optimal MAP
- Performance plateaus beyond 100 entities

## 💡 Practical Applications

1. **Cold-Start Onboarding**: Collect 10-20 favorite entities during user signup
2. **Personalized Search**: Adapt results based on user's social profile
3. **Conversational AI**: Guide conversations toward user-relevant topics
4. **Cross-Platform Recommendation**: Transfer preferences from social media to other domains
5. **Privacy-Preserving**: Social embeddings anonymize user identity while preserving preference signals

## 📁 Repository Structure

```
personalized_recommendation/
├── run_reproduction.py          # Reproduces the paper's tables and figures
├── reproduction/                # Reproduction code
│   ├── data.py                  #   categories, users, closed-world splits
│   ├── embeddings.py            #   SocialVec loading and user projection
│   ├── ranking.py               #   cosine / popularity ranking, average precision
│   ├── experiments.py           #   Table 2, Figure 1, Table 4
│   ├── demographics.py          #   Figure 2 (socio-demographic classifiers)
│   ├── llm.py                   #   Table 3 (GPT-4o, optional)
│   ├── paper_results.py         #   published numbers, for comparison
│   ├── plots.py, report.py
├── results/                     # Generated outputs; REPORT.md compares them with the paper
├── data/                        # Experimental dataset
├── legacy_code/                 # Original notebooks
├── paper/                       # The paper (arXiv:2603.10148)
├── requirements.txt
└── requirements-llm.txt         # Extra dependencies for the GPT-4o experiment
```

## 🔁 Reproducing the paper

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # on Linux/macOS: .venv/bin/pip
.venv/Scripts/python run_reproduction.py
```

The first run downloads the SocialVec embeddings (~300 MB) into `models/`. A full run takes
about two minutes and writes the tables, figures and per-user scores to `results/`.
[results/REPORT.md](results/REPORT.md) puts every reproduced number next to the paper's.

The GPT-4o experiment (Table 3) is optional because it makes about 33.5K API calls. It uses the
authors' dspy prompts and caches responses, so interrupted runs resume where they stopped:

```bash
.venv/Scripts/pip install -r requirements-llm.txt
# Azure: set AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION  (or use --llm-model openai/gpt-4o with OPENAI_API_KEY)
.venv/Scripts/python run_reproduction.py --llm --llm-users-per-category 5   # cheap trial run
.venv/Scripts/python run_reproduction.py --llm
```


## 🔒 Ethics & Privacy

- Research approved by IRB (031/23)
- User embeddings are anonymized via aggregation in low-dimensional space
- Identity cannot be recovered from social embeddings
- Highlights both utility and risks of social user modeling
- Demonstrates that LLMs may already encode similar social biases

## 📄 License

See [LICENSE](LICENSE) file for details.
