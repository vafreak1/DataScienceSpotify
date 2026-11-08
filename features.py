"""
Shared feature engineering for the Music Hits project.

Every model notebook imports `build_features` from here, so all models
start from exactly the same features. Model-specific steps (imputation,
scaling, one-hot encoding) are NOT done here - they happen in each model
notebook, because trees and neural networks need different preprocessing.

There are two kinds of features:
  1. Label-free features: computed only from the input columns (year, log of
     followers, title keywords, genre flags, ...). No risk of leakage.
  2. Target encodings: replace a category (artist, label, album, genre) by
     the hit rate of that category. These use the label, so they are computed
     "out-of-fold" to avoid leakage (see `target_encode`).
"""

import numpy as np
import pandas as pd

# keywords in the song title that showed a clearly different hit rate in the EDA
TITLE_KEYWORDS = ["feat", "remaster", "remix", "live", "karaoke", "instrumental", "version", "cover"]

# categories that get a target encoding (one hit rate per artist / label / album)
TE_COLUMNS = ["artist_name", "record_label", "album_name"]

# smoothing strength: how many 'average' songs we pretend each category already has.
# Tested 1, 5, 20 in notebook 03 -> almost no difference in model AUC; 5 is a middle ground.
SMOOTHING = 5


# --------------------------------------------------------------------------
# 1. Label-free features
# --------------------------------------------------------------------------

def split_genres(genre_strings):
    """'pop|dance pop' -> ['pop', 'dance pop'];  missing -> []"""
    return genre_strings.fillna("").str.split("|").apply(lambda genres: [g for g in genres if g])


def basic_features(df):
    """Features that only use the input columns of one row (no labels involved)."""
    out = pd.DataFrame(index=df.index)

    # --- audio features: copied as they are ---
    audio_cols = ["audio_dance", "audio_energy", "audio_key", "audio_loudness", "audio_mode",
                  "audio_speech", "audio_acoustic", "audio_instrumental", "audio_live",
                  "audio_valence", "audio_tempo", "time_signature", "duration_sec"]
    out[audio_cols] = df[audio_cols]

    # --- hidden missing values -> real NaN (a tempo of 0 or 0/1 beats per bar is not real music) ---
    out.loc[out["audio_tempo"] == 0, "audio_tempo"] = np.nan
    out.loc[out["time_signature"].isin([0, 1]), "time_signature"] = np.nan

    # --- artist popularity: log1p squeezes 0 ... 116 million into 0 ... ~18.6 ---
    out["log_followers"] = np.log1p(df["artist_followers"])

    # --- release date ---
    date = pd.to_datetime(df["release_date"])
    out["year"] = date.dt.year
    out["month"] = date.dt.month
    # many dates are '1 January' -> usually only the year was known
    out["date_is_jan_1"] = ((date.dt.month == 1) & (date.dt.day == 1)).astype(int)

    # --- simple categorical / boolean columns ---
    out["explicit"] = df["explicit_lyrics"].astype(int)
    for album_type in ["album", "single", "compilation"]:
        out[f"album_type_{album_type}"] = (df["album_type"] == album_type).astype(int)

    # --- song title ---
    title = df["track_name"].fillna("").str.lower()
    for word in TITLE_KEYWORDS:
        out[f"title_has_{word}"] = title.str.contains(word, regex=False).astype(int)
    out["title_length"] = title.str.len()
    out["title_is_album_name"] = (df["track_name"] == df["album_name"]).astype(int)

    # --- genres: is there any genre, and how many ---
    genres = split_genres(df["artist_genres"])
    out["genres_missing"] = (genres.str.len() == 0).astype(int)
    out["n_genres"] = genres.str.len()

    return out


def count_features(df, all_rows):
    """How many songs share this artist / label / album, counted in train + test together.
    This needs no labels, so using the test rows here is allowed."""
    out = pd.DataFrame(index=df.index)
    for col in TE_COLUMNS:
        counts = all_rows[col].value_counts()
        out[f"{col}_n_songs"] = df[col].map(counts).fillna(0)
    return out


def genre_multi_hot(df, top_genres):
    """One 0/1 column per frequent genre: 1 if the artist has that genre."""
    # one row per (song, genre) pair, keeping only the frequent genres
    long = split_genres(df["artist_genres"]).explode()
    long = long[long.isin(top_genres)]
    # cross-table: songs x genres, 1 where the song has the genre
    table = pd.crosstab(long.index, long.values)
    # songs without any frequent genre become rows of zeros
    table = table.reindex(index=df.index, columns=top_genres, fill_value=0)
    return table.add_prefix("genre_")


# --------------------------------------------------------------------------
# 2. Target encoding (uses the label -> must be done out-of-fold)
# --------------------------------------------------------------------------

def smoothed_hit_rate(keys, y, prior):
    """Hit rate per category, pulled towards the overall hit rate (prior) for small categories.
    Example with SMOOTHING=5 and prior=0.10:
      artist with 2 songs, both hits     -> (2 + 5*0.10) / (2 + 5)     = 0.36   (not 1.0!)
      artist with 200 songs, 150 hits    -> (150 + 5*0.10) / (200 + 5) = 0.73
    """
    stats = pd.DataFrame({"key": keys.values, "y": y.values}).groupby("key")["y"].agg(["sum", "count"])
    return (stats["sum"] + SMOOTHING * prior) / (stats["count"] + SMOOTHING)


def target_encode(fit_keys, fit_y, fit_folds, apply_keys_list):
    """
    Replace each category by its smoothed hit rate.

    - Rows of the fitting data get an OUT-OF-FOLD value: a row in fold k is encoded
      with hit rates computed on the other 4 folds only. So a song never 'sees'
      its own label - otherwise the model would learn to trust this feature far too much.
    - Rows of other data (validation, test) are encoded with hit rates from ALL fitting rows.
    - Unknown categories get the overall hit rate (prior).
    """
    prior = fit_y.mean()

    # out-of-fold encoding for the fitting rows
    fit_encoded = pd.Series(np.nan, index=fit_keys.index)
    for k in np.unique(fit_folds):
        in_fold = fit_folds == k
        rates = smoothed_hit_rate(fit_keys[~in_fold], fit_y[~in_fold], prior)
        fit_encoded[in_fold] = fit_keys[in_fold].map(rates).values
    fit_encoded = fit_encoded.fillna(prior)

    # full-data encoding for validation / test rows
    rates_all = smoothed_hit_rate(fit_keys, fit_y, prior)
    applied = [keys.map(rates_all).fillna(prior) for keys in apply_keys_list]

    return fit_encoded, applied


def genre_target_encode(fit_df, fit_y, fit_folds, apply_dfs):
    """
    Target encoding for genres. A song can have several genres, so we:
      1. 'explode' the data: one row per (song, genre) pair
      2. encode each genre with its hit rate (out-of-fold, like above)
      3. collapse back to one row per song: mean and max of its genre hit rates
    Songs without any genre get NaN here (the 'genres_missing' flag already covers them).
    """
    def explode(df):
        long = split_genres(df["artist_genres"]).explode().dropna()
        return long.index.to_numpy(), long.reset_index(drop=True)  # (song row index, genre)

    def collapse(row_index, values, index):
        per_song = pd.DataFrame({"row": row_index, "v": values}).groupby("row")["v"].agg(["mean", "max"])
        return per_song.reindex(index).rename(columns=lambda c: f"genre_te_{c}")

    fit_rows, fit_genres = explode(fit_df)
    fit_genre_y = fit_y.loc[fit_rows].reset_index(drop=True)
    fit_genre_folds = pd.Series(fit_folds, index=fit_df.index).loc[fit_rows].to_numpy()

    apply_exploded = [explode(df) for df in apply_dfs]
    fit_values, apply_values = target_encode(
        fit_genres, fit_genre_y, fit_genre_folds, [genres for _, genres in apply_exploded])

    fit_out = collapse(fit_rows, fit_values.to_numpy(), fit_df.index)
    apply_out = [collapse(rows, vals.to_numpy(), df.index)
                 for (rows, _), vals, df in zip(apply_exploded, apply_values, apply_dfs)]
    return fit_out, apply_out


# --------------------------------------------------------------------------
# 3. Put everything together
# --------------------------------------------------------------------------

def build_features(fit_df, apply_dfs, all_rows, n_top_genres=150):
    """
    fit_df     : labelled rows the models are trained on (needs 'music_hit' and 'fold')
    apply_dfs  : list of DataFrames to transform with what was learned on fit_df (e.g. [valid, test])
    all_rows   : train + test inputs together, only used for the label-free song counts

    Returns (X_fit, [X_apply_1, X_apply_2, ...]) with identical columns.
    """
    y, folds = fit_df["music_hit"], fit_df["fold"].to_numpy()

    # the most frequent genres, decided on the fitting data only
    top_genres = split_genres(fit_df["artist_genres"]).explode().dropna().value_counts().index[:n_top_genres]

    def label_free(df):
        return pd.concat([basic_features(df), count_features(df, all_rows), genre_multi_hot(df, top_genres)], axis=1)

    X_fit = label_free(fit_df)
    X_apply = [label_free(df) for df in apply_dfs]

    # target encodings for artist / label / album
    for col in TE_COLUMNS:
        fit_enc, apply_enc = target_encode(fit_df[col], y, folds, [df[col] for df in apply_dfs])
        X_fit[f"{col}_te"] = fit_enc
        for X, enc in zip(X_apply, apply_enc):
            X[f"{col}_te"] = enc

    # target encodings for genres
    fit_genre, apply_genre = genre_target_encode(fit_df, y, folds, apply_dfs)
    X_fit = X_fit.join(fit_genre)
    X_apply = [X.join(g) for X, g in zip(X_apply, apply_genre)]

    return X_fit, X_apply
