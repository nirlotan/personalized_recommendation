"""
Data utility functions for user and topic account processing.

This module contains functions for filtering user accounts, counting categories,
separating topic accounts, and preparing dataframes for topic-based experiments.
"""

import pandas as pd
from typing import List, Tuple


def get_relevant_subset_users(x: List, account_list: List) -> List:
    """
    Filter user accounts to only include those in the account list.
    
    Args:
        x: List of user account IDs
        account_list: List of valid account IDs to filter against
        
    Returns:
        List of account IDs that are in both x and account_list
    """
    return list(set(x).intersection(set(account_list)))


def find_number_of_categories_per_user(list_of_accounts: List, all_topics: dict) -> int:
    """
    Count how many categories a user follows based on their account list.
    
    Args:
        list_of_accounts: List of account IDs that the user follows
        all_topics: Dictionary mapping topic names to DataFrames with 'twitter_user_id' column
        
    Returns:
        Number of categories the user follows (0 to len(all_topics))
    """
    num_of_categories = 0
    for t in all_topics:
        all_topics[t]['twitter_user_id'] = all_topics[t]['twitter_user_id'].astype(int)
        if len(set(all_topics[t]['twitter_user_id'].values).intersection(set(list_of_accounts))) > 0:
            num_of_categories += 1
    return num_of_categories


def seperate_topic_accounts(user_account_list: List, topic_account_list: List) -> Tuple[List, List]:
    """
    Split user accounts into non-topic accounts and topic accounts.
    
    This function separates a user's followed accounts into two groups:
    - Accounts that are NOT in the specified topic (used as context for embeddings)
    - Accounts that ARE in the specified topic (used as ground truth)
    
    Args:
        user_account_list: List of all account IDs that the user follows
        topic_account_list: List of account IDs for a specific topic category
        
    Returns:
        Tuple of (non_topic_accounts, topic_accounts):
            - non_topic_accounts: Accounts user follows that are NOT in the topic
            - topic_accounts: Accounts user follows that ARE in the topic
    """
    user_accounts = set(user_account_list)
    topic_accounts = set(topic_account_list)
    return list(user_accounts - topic_accounts), list(user_accounts.intersection(topic_accounts))


def prepare_df_for_topic(topic: str, df_users: pd.DataFrame, all_topics: dict) -> pd.DataFrame:
    """
    Prepare a dataframe for a specific topic experiment by separating accounts.
    
    This function filters users for a given topic and splits their followed accounts
    into topic and non-topic accounts for cross-domain evaluation.
    
    Args:
        topic: Name of the topic category (e.g., 'MusicalArtist', 'Politician')
        df_users: DataFrame containing user data with 'topic' and 'experiment_accounts' columns
        all_topics: Dictionary mapping topic names to DataFrames with 'twitter_user_id' column
        
    Returns:
        DataFrame with additional columns:
            - non_topic_accounts: List of account IDs not in the target topic
            - topic_accounts: List of account IDs in the target topic
    """
    df_for_topic = df_users[df_users['topic'] == topic].copy()
    df_for_topic[["non_topic_accounts", "topic_accounts"]] = df_for_topic['experiment_accounts'].apply(
        lambda x: pd.Series(seperate_topic_accounts(x, all_topics[topic]['twitter_user_id'].values))
    )
    return df_for_topic
