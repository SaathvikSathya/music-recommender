import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

path = "lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv"
cols = ["user", "timestamp", "artist_id", "artist", "track_id", "track"]

# 1. Load the data
print("Loading data (this takes a minute or two)...")
df = pd.read_csv(
    path, sep="\t", names=cols,
    usecols=["user", "artist", "track"],
    dtype="category", quoting=3, on_bad_lines="skip",
).dropna()

# 2. Give every (artist, song) pair its own number
n_tracks = len(df["track"].cat.categories)
df["song_key"] = (
    df["artist"].cat.codes.astype("int64") * n_tracks
    + df["track"].cat.codes.astype("int64")
)

# 3. Keep songs that at least 10 different people played
listeners = df.groupby("song_key")["user"].nunique()
df = df[df["song_key"].isin(listeners[listeners >= 10].index)]

# 4. Turn users and songs into row/column numbers
user_codes, _ = pd.factorize(df["user"])
song_codes, song_keys = pd.factorize(df["song_key"])
first = df.drop_duplicates("song_key").set_index("song_key")
song_artist = first.loc[song_keys, "artist"].astype(str).to_numpy()
song_title = first.loc[song_keys, "track"].astype(str).to_numpy()
artist_ids, _ = pd.factorize(pd.Series(song_artist).str.lower())
titles = pd.Series(song_title).str.lower()

# 5. Build the table: rows = users, columns = songs, values = how much they played
counts = (
    pd.DataFrame({"u": user_codes, "s": song_codes})
    .groupby(["u", "s"]).size().reset_index(name="plays")
)
X = csr_matrix(
    (np.log1p(counts["plays"]), (counts["u"], counts["s"])),
    shape=(user_codes.max() + 1, len(song_keys)),
)
listener_count = np.asarray((X > 0).sum(axis=0)).ravel()
Xn = normalize(X.T.tocsr())   # one row per song
del df
print(f"Ready: {len(song_keys)} songs to recommend from.\n")


def song_vector(i):
    return Xn[i].toarray().ravel()


def find_song(query):
    q = query.strip().lower()
    if not q:
        return None
    hits = np.where(titles.values == q)[0]
    if len(hits) == 0:
        hits = np.where(titles.str.contains(q, regex=False).values)[0]
    if len(hits) == 0:
        return None
    hits = hits[np.argsort(listener_count[hits])[::-1]][:5]
    if len(hits) == 1:
        return hits[0]
    print("  Which one?")
    for n, i in enumerate(hits, 1):
        print(f"   {n}. {song_title[i]} - {song_artist[i]}")
    while True:
        c = input("  Number (or Enter to cancel): ").strip()
        if c == "":
            return None
        if c.isdigit() and 1 <= int(c) <= len(hits):
            return hits[int(c) - 1]


def build_queue(taste, seen, blocked, n=5):
    scores = np.asarray(Xn @ taste).ravel()
    scores[list(seen)] = -np.inf
    scores[np.isin(artist_ids, list(blocked))] = -np.inf
    queue, used_artists = [], set()
    for i in np.argsort(scores)[::-1]:
        if scores[i] == -np.inf:
            break
        if artist_ids[i] in used_artists:      # max one song per artist
            continue
        used_artists.add(artist_ids[i])
        queue.append(i)
        if len(queue) == n:
            break
    return queue


# 6. The user picks 5 songs
print("Pick 5 songs you like (type a song title).")
picks = []
while len(picks) < 5:
    i = find_song(input(f"Song #{len(picks) + 1}: "))
    if i is None:
        print("  Not found. Try another song.")
    elif i in picks:
        print("  Already picked.")
    else:
        picks.append(i)
        print(f"  Added: {song_title[i]} - {song_artist[i]}")

# 7. Starting taste = the 5 picks. Block their artists from recommendations.
taste = np.zeros(Xn.shape[1])
for i in picks:
    taste += song_vector(i)
seen = set(picks)
blocked = {artist_ids[i] for i in picks}

# 8. The Jam loop
LIKE_WEIGHT = 1.0
SKIP_WEIGHT = 0.5
while True:
    queue = build_queue(taste, seen, blocked)
    if not queue:
        print("No more recommendations.")
        break
    print("\nUp next:")
    for rank, i in enumerate(queue, 1):
        print(f"  {rank}. {song_title[i]} - {song_artist[i]}")

    top = queue[0]
    choice = input(
        f"Playing '{song_title[top]}': [l]isten, [s]kip, [q]uit > "
    ).strip().lower()
    if choice == "q":
        break
    elif choice == "l":
        taste += LIKE_WEIGHT * song_vector(top)
        seen.add(top)
        blocked.add(artist_ids[top])
    elif choice == "s":
        taste -= SKIP_WEIGHT * song_vector(top)
        seen.add(top)