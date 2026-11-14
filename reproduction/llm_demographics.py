"""
Demographics-only LLM personalization: the LLM first describes a user's likely
gender, age, ethnicity and education from the names of accounts they follow,
then ranks the candidates given only that description.
"""

import re
from typing import Dict, List

PROMPT_VERSION = "v3"  # part of the cache keys; bump when a prompt changes
# v1: description prompt asked for the "likely" attributes; the model could decline to guess.
# v2: the model must always guess, from stereotypes about the accounts' audiences.
# v3: as v2, with dspy's end-of-output marker stripped from the description before ranking.
# v4: as v3, with SINGLE_VALUE added: exactly one value per attribute (no ranges or alternatives).
SINGLE_VALUE = ("State exactly one value for each attribute: one gender, one age group (for example \"in their 30s\"), "
                "one ethnicity and one level of education. Never give a range spanning two groups or alternatives "
                "such as \"30s or 40s\" or \"Caucasian or mixed\".")


# Topics or interests named in a description, which the prompt forbids (it leaks more than demographics)
TOPICS = (r"\b(?:politic\w*|sport\w*|music\w*|entertainment|news|pop culture|conservative|liberal|progressive|republican|"
          r"democrat\w*|business|military|gaming|comedy|celebrit\w*|interest\w*|fan|fans|enthusiast\w*|hip[- ]hop|media)\b")

# Descriptive statistics of the stated attributes: each description gets the first value it states for an attribute;
# hedges such as "30s or 40s" also count as giving more than one.
_RANGE = r"\s*(?:-|to|–)\s*\d{2}"
ATTRIBUTE_VALUES = {
    "Gender": [("Female", r"\b(?:female|woman|women)\b"), ("Male", r"\b(?:male|man|men)\b")],
    "Age": [("Under 30", rf"\b(?:teen\w*|young adults?|20s|twenties|[12]\d{_RANGE})\b"),
            ("30s", rf"\b(?:30s|thirties|3\d{_RANGE})\b"),
            ("40s", rf"\b(?:40s|forties|4\d{_RANGE})\b"),
            ("Middle-aged", r"\bmiddle[- ]aged\b"),
            ("50 or older", rf"\b(?:[5-9]0s|older|seniors?|[5-9]\d{_RANGE})\b")],
    "Ethnicity": [("Caucasian / White", r"\b(?:caucasian|white)\b"), ("African American / Black", r"\b(?:african[- ]american|black)\b"),
                  ("Hispanic / Latino", r"\b(?:hispanic|latin[oax]\w*)\b"), ("Asian", r"\basian\b"),
                  ("Mixed / diverse", r"\b(?:mixed|diverse|multi[- ]?ethnic|multiracial)\b")],
    "Education": [("High school", r"\bhigh[- ]school\b"),
                  ("Some higher education", r"\b(?:some college|some higher education|at least some (?:higher education|college))\b"),
                  ("College / university degree", r"\b(?:bachelor\w*|college degree|university degree|undergraduate)\b"),
                  ("Graduate degree", r"\b(?:graduate degree|postgraduate|master\w*|advanced degree|doctora\w*|phd)\b"),
                  ("Higher education, level not stated", r"\b(?:higher education|college|university|degree)\b")],
}
UNSPECIFIC = "Higher education, level not stated"  # used only when no specific level is stated


def stated_values(text, attr):
    """The values a description states for an attribute, in order of first mention."""
    hits = sorted((mt.start(), value) for value, pat in ATTRIBUTE_VALUES[attr] if (mt := re.search(pat, text, re.I)))
    values = [v for _, v in hits]
    specific = [v for v in values if v != UNSPECIFIC]
    return specific or values


def clean_description(text: str) -> str:
    """Remove dspy's end-of-output marker, which some models leave inside the answer field."""
    return re.sub(r"\[\[\s*##\s*completed\s*##\s*\]\]", "", str(text)).strip()


def make_demographic_programs(lm, describe_lm=None, single_value=False):
    """
    The two dspy programs: describe(followed_names) -> description and
    rank(description, candidates_json) -> ranked ids. Ranking runs with `lm`;
    the description with `describe_lm` (default: `lm`). single_value adds SINGLE_VALUE
    to the description instructions.
    """
    describe_lm = describe_lm or lm
    import dspy

    class DemographicDescription(dspy.Signature):
        """
        Given the names of accounts that a Twitter user follows, write a short, concise description of the user that states their most likely gender, age, ethnicity, and level of education (primary school, high school, higher education, etc.).
        Deduce these attributes only from the names of the accounts they follow.
        You must always guess the most likely value of all four attributes, based on stereotypes about the typical audience of these accounts, even when the evidence is weak. Never answer that the attributes cannot be determined or that the evidence is insufficient.
        Describe only these four attributes: do not mention the user's interests or hobbies, and do not name any of the accounts.
        """
        followed_accounts = dspy.InputField(desc="Names of accounts the user follows")
        description = dspy.OutputField(desc="One or two sentences stating the user's most likely gender, age, ethnicity, and level of education")

    class TwitterDemographicRecommendation(dspy.Signature):
        """
        given the below Twitter user, for whom we provide a short demographic description, and a list of possible accounts to recommend for the user to follow, create a prioritized recommendation list.
        You must rank the entire list provided to you.
        Try to deduce what you think the user might like based on their demographic description.
        Return ONLY a valid list object with the ranked account twitter_user_id sorted by relevance (highest to lowest). Do not include name or reasoning.
        Example format: [71026122, 141341662, 567562934, ...]
        """
        user_description = dspy.InputField(desc="Short description of the user's gender, age, ethnicity, and level of education")
        potential_accounts = dspy.InputField(desc="JSON list of potential accounts to recommend with twitter_user_id and names")
        ranked_ids = dspy.OutputField(desc="list with the account twitter_user_id sorted by priority as a ranked key")

    if single_value:
        DemographicDescription = DemographicDescription.with_instructions(
            DemographicDescription.instructions.rstrip() + "\n" + SINGLE_VALUE)
    describe_program = dspy.Predict(DemographicDescription)
    rank_program = dspy.Predict(TwitterDemographicRecommendation)

    def describe(followed: str) -> str:
        with dspy.context(lm=describe_lm):
            return clean_description(describe_program(followed_accounts=followed).description)

    def rank(description: str, candidates: str) -> str:
        with dspy.context(lm=lm):
            return rank_program(user_description=description, potential_accounts=candidates).ranked_ids

    def render(kind: str, **inputs) -> List[Dict[str, str]]:
        """The chat messages dspy sends for a call (for inspection; no model call)."""
        signature = (describe_program if kind == "describe" else rank_program).signature
        return dspy.ChatAdapter().format(signature, demos=[], inputs=inputs)

    return describe, rank, render
