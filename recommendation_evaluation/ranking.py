"""
Ranking functions for social vector and popularity-based recommendation.

This module contains functions for ranking accounts based on cosine similarity
to user embeddings (social vector approach) and popularity (follower count).
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from typing import List


def get_sv_rank(user_vec: np.ndarray, relevant_vecs: pd.DataFrame) -> pd.DataFrame:
    """
    Rank accounts by cosine similarity between user vector and account vectors.
    
    Builds a DataFrame of cosine similarities between one user vector and
    all relevant account vectors, then ranks them by similarity (ascending).
    
    Args:
        user_vec: User's social vector embedding (numpy array)
        relevant_vecs: DataFrame containing account information with columns:
            - twitter_screen_name: Account screen name
            - twitter_user_id: Account user ID
            - account_sv: Account's social vector embedding
            
    Returns:
        DataFrame with columns:
            - rank: Ranking position (1-indexed, lower is better)
            - account: Account screen name
            - twitter_user_id: Account user ID
            - similarity: Cosine distance (lower is more similar)
    """
    records = []
    for i, row in relevant_vecs.iterrows():
        records.append({
            'account': row['twitter_screen_name'],
            'twitter_user_id': row['twitter_user_id'],
            'similarity': cosine(row['account_sv'], user_vec)
        })
    sv_ranked_list = pd.DataFrame(records).sort_values(
        by='similarity', ascending=True
    ).reset_index(drop=True).reset_index()
    sv_ranked_list.rename(columns={'index': 'rank'}, inplace=True)
    sv_ranked_list['rank'] = sv_ranked_list['rank'] + 1
    return sv_ranked_list


def get_popular_rank(relevant_vecs: pd.DataFrame) -> pd.DataFrame:
    """
    Rank accounts by popularity (follower count).
    
    Creates a popularity-based ranking of accounts, sorted by their follower
    count in descending order (most popular first).
    
    Args:
        relevant_vecs: DataFrame containing account information with columns:
            - twitter_screen_name: Account screen name
            - twitter_user_id: Account user ID
            - followers_count: Number of followers
            
    Returns:
        DataFrame with columns:
            - rank: Ranking position (1-indexed)
            - account: Account screen name
            - twitter_user_id: Account user ID
    """
    popularity_rank = relevant_vecs.sort_values(
        by=['followers_count'], ascending=False
    ).reset_index(drop=True).reset_index()
    popularity_rank.rename(columns={
        'index': 'rank',
        'twitter_screen_name': 'account'
    }, inplace=True)
    popularity_rank['rank'] = popularity_rank['rank'] + 1
    return popularity_rank[['rank', 'account', 'twitter_user_id']]


def reset_and_rank_topic_accounts(df: pd.DataFrame, topic_accounts: List) -> pd.DataFrame:
    """
    Filter a ranked list to specific topic accounts and reset rank indices.
    
    This function filters a full ranked list down to only the accounts that belong
    to a specific topic, then resets the rank indices to be sequential (1, 2, 3...).
    This is useful for evaluating rankings on only the relevant accounts.
    
    Args:
        df: DataFrame with 'twitter_user_id' and ranking information
        topic_accounts: List of account IDs to filter to
        
    Returns:
        DataFrame filtered to topic_accounts with:
            - index: Sequential ranking position starting at 1
            - All original columns from df
    """
    df = df[df['twitter_user_id'].isin(topic_accounts)]
    df = df.reset_index(drop=True)
    df = df.reset_index()
    df['index'] = df['index'] + 1
    return df
