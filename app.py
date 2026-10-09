import html
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

DATA = (
    Path(__file__).parent
    / "lastfm-dataset-1K"
    / "userid-timestamp-artid-artname-traid-traname.tsv"
)
COLS = ["user", "timestamp", "artist_id", "artist", "track_id", "track"]
LIKE_WEIGHT = 3.0
SKIP_WEIGHT = 0.5

st.set_page_config(page_title="JAM!", page_icon="🤘", layout="wide")

# ---------- 0. Punk rock look: all the styling lives here ----------
# Colors: hot pink #ff2e88, acid green #c6ff00, yellow #ffe600, paper #f5f0e6
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Anton&family=Permanent+Marker&family=Special+Elite&display=swap');

.stApp {
  background-color: #0b0b0b;
  background-image:
    radial-gradient(rgba(255,46,136,.14) 1px, transparent 1.6px),
    radial-gradient(rgba(255,255,255,.06) 1px, transparent 1.6px);
  background-size: 16px 16px, 27px 27px;
  background-position: 0 0, 8px 8px;
}
.block-container { max-width: 1100px; padding-top: 1.2rem; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

.stMarkdown, .stMarkdown p, .stCaption, label p {
  font-family: 'Special Elite', monospace;
  color: #f5f0e6;
}

/* checkerboard strip */
.checker {
  height: 26px;
  background: repeating-conic-gradient(#fff 0% 25%, #111 0% 50%) 0 0 / 26px 26px;
  border: 3px solid #fff;
  margin-bottom: 14px;
}

/* ransom-note title */
.ransom { text-align: center; margin: 6px 0 4px; }
.ransom span {
  display: inline-block;
  font-size: 4.6rem; line-height: 1.05;
  padding: 0 .16em; margin: 0 .05em;
  box-shadow: 4px 4px 0 #000;
}
.ransom .r1 { background:#f5f0e6; color:#111; font-family:'Anton'; transform: rotate(-6deg); }
.ransom .r2 { background:#ff2e88; color:#fff; font-family:'Permanent Marker'; transform: rotate(4deg) translateY(4px); }
.ransom .r3 { background:#c6ff00; color:#111; font-family:'Special Elite'; font-weight:bold; transform: rotate(-3deg); }
.ransom .r4 { background:#111; color:#ffe600; border:3px solid #ffe600; font-family:'Anton'; transform: rotate(7deg) translateY(-3px); }

/* duct tape subtitle */
.duct {
  display: table; margin: 12px auto 8px;
  background: #b9b9b2; color: #111;
  font-family: 'Special Elite', monospace; letter-spacing: .08em;
  padding: 4px 18px; transform: rotate(-1.5deg);
  box-shadow: 3px 3px 0 #000;
}

/* caution tape with scrolling text */
.tape {
  background: repeating-linear-gradient(-45deg, #ffe600 0 12px, #111 12px 24px);
  padding: 6px 0; margin: 10px 0 18px; transform: rotate(-1deg);
}
.ticker { background: #111; overflow: hidden; white-space: nowrap; }
.ticker span {
  display: inline-block; padding-left: 100%;
  font-family: 'Anton', sans-serif; color: #ffe600;
  letter-spacing: .2em; font-size: 1.05rem;
  animation: scroll 24s linear infinite;
}
@keyframes scroll { to { transform: translateX(-100%); } }

/* section headings */
.hd {
  display: inline-block; font-family: 'Anton', sans-serif;
  font-size: 1.5rem; letter-spacing: .12em; text-transform: uppercase;
  padding: 2px 14px; margin: 18px 0 12px;
  transform: rotate(-1.5deg); border: 3px solid #111;
}
.hd.pink   { background:#ff2e88; color:#fff; box-shadow:5px 5px 0 #f5f0e6; }
.hd.green  { background:#c6ff00; color:#111; box-shadow:5px 5px 0 #ff2e88; }
.hd.yellow { background:#ffe600; color:#111; box-shadow:5px 5px 0 #f5f0e6; }

/* now-playing flyer */
.flyer {
  position: relative; background: #f5f0e6; color: #111;
  border: 3px solid #111; padding: 16px 20px 18px;
  transform: rotate(1deg); box-shadow: 8px 8px 0 #ff2e88;
  margin-bottom: 14px;
}
.flyer::before {
  content: ''; position: absolute; top: -13px; left: 40%;
  width: 90px; height: 24px; background: rgba(255,230,0,.85);
  transform: rotate(-4deg);
}
.flyer .tag { font-family:'Special Elite',monospace; letter-spacing:.3em; font-size:.8rem; color:#ff2e88; }
.flyer .song { font-family:'Anton',sans-serif; font-size:2.1rem; line-height:1.1; text-transform:uppercase; margin:4px 0; word-break:break-word; }
.flyer .by { font-family:'Permanent Marker',cursive; font-size:1.25rem; }
.flyer .none { font-family:'Special Elite',monospace; font-size:.85rem; margin-top:8px; }

/* animated equalizer */
.eq { display:flex; gap:5px; align-items:flex-end; height:36px; margin-top:12px; }
.eq i { display:block; width:9px; height:36px; background:#111; transform-origin:bottom; animation: eqb .8s ease-in-out infinite; }
.eq i:nth-child(odd) { background:#ff2e88; }
.eq i:nth-child(2) { animation-delay:.10s; }
.eq i:nth-child(3) { animation-delay:.25s; }
.eq i:nth-child(4) { animation-delay:.05s; }
.eq i:nth-child(5) { animation-delay:.35s; }
.eq i:nth-child(6) { animation-delay:.20s; }
.eq i:nth-child(7) { animation-delay:.40s; }
.eq i:nth-child(8) { animation-delay:.15s; }
@keyframes eqb { 0%,100% { transform: scaleY(.2); } 50% { transform: scaleY(1); } }

/* album art */
div[data-testid="stImage"] img {
  border: 4px solid #f5f0e6; box-shadow: 8px 8px 0 #c6ff00;
  transform: rotate(-3deg); filter: contrast(1.15) saturate(1.25);
}
audio { border: 3px solid #f5f0e6; }

/* torn-paper set list */
.setwrap { filter: drop-shadow(7px 7px 0 #ff2e88); transform: rotate(.8deg); margin: 8px 0 16px; }
.setlist {
  background: #f2ecd9; color: #111; padding: 16px 22px 28px;
  font-family: 'Special Elite', monospace;
  clip-path: polygon(0 0,100% 0,100% 97%,95% 100%,90% 97%,85% 100%,80% 97%,75% 100%,70% 97%,65% 100%,60% 97%,55% 100%,50% 97%,45% 100%,40% 97%,35% 100%,30% 97%,25% 100%,20% 97%,15% 100%,10% 97%,5% 100%,0 97%);
}
.setlist .sethead { font-family:'Anton',sans-serif; letter-spacing:.15em; font-size:1.2rem; border-bottom:3px double #111; margin-bottom:8px; }
.setlist ol { padding-left: 1.4rem; margin: 0; }
.setlist li { margin: 7px 0; font-size: 1.05rem; color: #111; }
.setlist li b { font-family:'Permanent Marker',cursive; font-weight: normal; }

/* stickers */
.stickers { display:flex; gap:14px; flex-wrap:wrap; margin: 10px 0 6px; }
.sticker {
  font-family:'Anton',sans-serif; letter-spacing:.08em;
  padding: 8px 18px; border: 3px solid #111; border-radius: 50px;
  box-shadow: 4px 4px 0 #f5f0e6;
}
.sticker b { font-size: 1.6rem; margin-left: 8px; }
.sticker.a { background:#c6ff00; color:#111; transform: rotate(-3deg); }
.sticker.b { background:#ff2e88; color:#fff; transform: rotate(2deg); }
.sticker.c { background:#ffe600; color:#111; transform: rotate(-1deg); }

/* paper strips (picks and history) */
.strip {
  background:#f5f0e6; color:#111; font-family:'Special Elite',monospace;
  padding: 5px 10px; margin: 7px 0; font-size: .92rem;
  border-left: 8px solid #c6ff00; transform: rotate(-.6deg);
}
.strip:nth-child(even) { transform: rotate(.7deg); }
.strip.skip { border-left-color:#ff2e88; text-decoration: line-through; opacity: .8; }

/* sidebar */
section[data-testid="stSidebar"] { background:#121212; border-right:4px solid #ff2e88; }

/* buttons */
.stButton > button {
  font-family:'Anton',sans-serif !important; letter-spacing:.12em;
  text-transform:uppercase; font-size:1.1rem !important;
  border: 3px solid #f5f0e6 !important; border-radius: 0 !important;
  background:#111 !important; color:#fff !important;
  box-shadow: 5px 5px 0 #f5f0e6; transition: transform .08s;
}
.stButton > button:hover { transform: translate(-2px,-2px) rotate(-1deg); box-shadow: 7px 7px 0 #ff2e88; }
.stButton > button:active { transform: translate(3px,3px); box-shadow: 1px 1px 0 #f5f0e6; }
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
  background:#c6ff00 !important; color:#111 !important; border-color:#111 !important; box-shadow: 5px 5px 0 #ff2e88;
}
.stButton > button[kind="secondary"], .stButton > button[data-testid="stBaseButton-secondary"] {
  background:#ff2e88 !important; color:#fff !important; border-color:#111 !important; box-shadow: 5px 5px 0 #f5f0e6;
}

/* text input and dropdown */
div[data-baseweb="input"] { border-radius:0 !important; border:3px solid #f5f0e6 !important; background:#f5f0e6 !important; box-shadow: 5px 5px 0 #ff2e88; }
div[data-baseweb="input"] input { color:#111 !important; font-family:'Special Elite',monospace !important; background:#f5f0e6 !important; }
div[data-baseweb="select"] > div { border-radius:0 !important; border:3px solid #f5f0e6 !important; background:#f5f0e6 !important; }
div[data-baseweb="select"] div, div[data-baseweb="select"] span { color:#111 !important; font-family:'Special Elite',monospace; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def banner():
    st.markdown(
        '<div class="checker"></div>'
        '<div class="ransom"><span class="r1">J</span><span class="r2">A</span>'
        '<span class="r3">M</span><span class="r4">!</span></div>'
        '<div class="duct">pick 5 songs // listen // skip // repeat</div>'
        '<div class="tape"><div class="ticker"><span>'
        "★ PICK 5 SONGS ★ HIT LISTEN OR SKIP ★ THE QUEUE LEARNS YOU ★ "
        "NO BORING PLAYLISTS ★ PLAY IT LOUD ★ "
        "</span></div></div>",
        unsafe_allow_html=True,
    )


def heading(text, color="pink"):
    st.markdown(
        f'<div class="hd {color}">{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def strip(text, cls=""):
    return f'<div class="strip {cls}">{html.escape(text)}</div>'


banner()


# ---------- 1. Load the data once and build the model ----------
@st.cache_resource(
    show_spinner="Loading 19 million listening records (1-2 minutes, first time only)..."
)
def load_engine():
    df = pd.read_csv(
        DATA, sep="\t", names=COLS,
        usecols=["user", "artist", "track"],
        dtype="category", quoting=3, on_bad_lines="skip",
    ).dropna()
    n_tracks = len(df["track"].cat.categories)
    df["song_key"] = (
        df["artist"].cat.codes.astype("int64") * n_tracks
        + df["track"].cat.codes.astype("int64")
    )
    listeners = df.groupby("song_key")["user"].nunique()
    df = df[df["song_key"].isin(listeners[listeners >= 10].index)]

    user_codes, _ = pd.factorize(df["user"])
    song_codes, song_keys = pd.factorize(df["song_key"])
    first = df.drop_duplicates("song_key").set_index("song_key")
    artist = first.loc[song_keys, "artist"].astype(str).to_numpy()
    title = first.loc[song_keys, "track"].astype(str).to_numpy()
    artist_ids, _ = pd.factorize(pd.Series(artist).str.lower())

    counts = (
        pd.DataFrame({"u": user_codes, "s": song_codes})
        .groupby(["u", "s"]).size().reset_index(name="plays")
    )
    X = csr_matrix(
        (np.log1p(counts["plays"]), (counts["u"], counts["s"])),
        shape=(user_codes.max() + 1, len(song_keys)),
    )
    return {
        "Xn": normalize(X.T.tocsr()),
        "artist": artist,
        "title": title,
        "artist_ids": artist_ids,
        "titles_lower": pd.Series(title).str.lower(),
        "listener_count": np.asarray((X > 0).sum(axis=0)).ravel(),
    }


E = load_engine()
Xn = E["Xn"]


@st.cache_resource(show_spinner="Building the taste map...")
def build_map():
    svd = TruncatedSVD(n_components=2, random_state=0)
    return svd, svd.fit_transform(Xn)


if HAS_PLOTLY:
    svd, coords = build_map()


def song_vector(i):
    return Xn[i].toarray().ravel()


def search_songs(query, n=8):
    q = query.strip().lower()
    if not q:
        return []
    t = E["titles_lower"]
    hits = np.where(t.values == q)[0]
    if len(hits) == 0:
        hits = np.where(t.str.contains(q, regex=False).values)[0]
    if len(hits) == 0:
        return []
    return hits[np.argsort(E["listener_count"][hits])[::-1]][:n]


def build_queue(taste, seen, blocked, n=5):
    scores = np.asarray(Xn @ taste).ravel()
    scores[list(seen)] = -np.inf
    scores[np.isin(E["artist_ids"], list(blocked))] = -np.inf
    queue, used = [], set()
    for i in np.argsort(scores)[::-1]:
        if scores[i] == -np.inf:
            break
        if E["artist_ids"][i] in used:      # at most one song per artist
            continue
        used.add(E["artist_ids"][i])
        queue.append(int(i))
        if len(queue) == n:
            break
    return queue


# ---------- 2. Look up a 30-second preview and album art ----------
@st.cache_data(show_spinner=False)
def get_preview(title, artist):
    try:
        r = requests.get(
            "https://itunes.apple.com/search",
            params={"term": f"{title} {artist}", "entity": "song", "limit": 10},
            timeout=8,
        )
        results = r.json().get("results", [])
    except Exception:
        return None
    a = artist.lower().replace("the ", "").strip()
    for item in results:
        name = item.get("artistName", "").lower().replace("the ", "").strip()
        if item.get("previewUrl") and name and (a in name or name in a):
            return {
                "preview": item["previewUrl"],
                "art": item.get("artworkUrl100", "").replace("100x100", "400x400"),
            }
    return None


# ---------- 3. App state and button actions ----------
ss = st.session_state
if "picks" not in ss:
    ss.picks = []
    ss.started = False
    ss.history = []


def add_song(i):
    if i not in ss.picks:
        ss.picks.append(i)
    ss.query = ""


def clear_picks():
    ss.picks = []


def start_jam():
    taste = np.zeros(Xn.shape[1])
    for i in ss.picks:
        taste += song_vector(i)
    ss.taste = taste
    ss.seen = set(ss.picks)
    ss.blocked = {int(E["artist_ids"][i]) for i in ss.picks}
    ss.started = True


def react(i, liked):
    if liked:
        ss.taste = ss.taste + LIKE_WEIGHT * song_vector(i)
        ss.blocked.add(int(E["artist_ids"][i]))
    else:
        ss.taste = ss.taste - SKIP_WEIGHT * song_vector(i)
    ss.seen.add(i)
    ss.history.append(("like" if liked else "skip", i))


def start_over():
    ss.started = False
    ss.picks = []
    ss.history = []


# ---------- 4. The taste map ("The Pit") ----------
def taste_map(queue):
    rng = np.random.default_rng(0)
    bg = rng.choice(len(coords), size=min(4000, len(coords)), replace=False)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=coords[bg, 0], y=coords[bg, 1], mode="markers",
        marker=dict(size=3, color="#888", opacity=0.35),
        hoverinfo="skip", name="All songs",
    ))

    def layer(idx, name, color, size, symbol="circle"):
        if not idx:
            return
        fig.add_trace(go.Scatter(
            x=coords[idx, 0], y=coords[idx, 1], mode="markers",
            marker=dict(size=size, color=color, symbol=symbol,
                        line=dict(width=1, color="white")),
            text=[f"{E['title'][i]} — {E['artist'][i]}" for i in idx],
            hoverinfo="text", name=name,
        ))

    layer(list(ss.picks), "Your picks", "#ff2e88", 16, "star")
    layer([i for a, i in ss.history if a == "like"], "Liked", "#c6ff00", 11)
    layer([i for a, i in ss.history if a == "skip"], "Skipped", "#ff4d4d", 11, "x")
    layer(list(queue), "Up next", "#ffe600", 11, "diamond")

    tv = ss.taste / (np.linalg.norm(ss.taste) or 1)
    t = svd.transform(tv.reshape(1, -1))[0]
    fig.add_trace(go.Scatter(
        x=[t[0]], y=[t[1]], mode="markers",
        marker=dict(size=22, color="rgba(0,0,0,0)", symbol="circle",
                    line=dict(width=3, color="white")),
        hoverinfo="name", name="Your taste",
    ))
    fig.update_layout(
        template="plotly_dark", height=460,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.35)",
        font=dict(family="Special Elite, monospace", color="#f5f0e6"),
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        legend=dict(orientation="h"),
    )
    return fig


# ---------- 5. The pages ----------
if not ss.started:
    heading(f"Pick 5 songs ({len(ss.picks)}/5)", "pink")
    if ss.picks:
        st.markdown(
            "".join(strip(f"{E['title'][i]} — {E['artist'][i]}") for i in ss.picks),
            unsafe_allow_html=True,
        )

    if len(ss.picks) < 5:
        query = st.text_input("Search for a song title", key="query")
        if query:
            hits = search_songs(query)
            if len(hits) == 0:
                st.warning("Not found. Try another song (this data is from 2009).")
            else:
                labels = {int(i): f"{E['title'][i]} — {E['artist'][i]}" for i in hits}
                choice = st.selectbox(
                    "Which one?", list(labels), format_func=lambda i: labels[i]
                )
                st.button("Add song", on_click=add_song, args=(choice,))
    else:
        st.button("Start the Jam ▶", on_click=start_jam, type="primary")

    if ss.picks:
        st.button("Clear picks", on_click=clear_picks)
    st.stop()

# --- Jam screen ---
queue = build_queue(ss.taste, ss.seen, ss.blocked)
if not queue:
    st.info("No more recommendations.")
    st.button("Start over", on_click=start_over)
    st.stop()

top = queue[0]
info = get_preview(E["title"][top], E["artist"][top])

c1, c2 = st.columns([1, 2])
with c1:
    if info and info["art"]:
        st.image(info["art"])
with c2:
    eq = "<div class='eq'>" + "<i></i>" * 8 + "</div>" if info else ""
    note = "" if info else "<div class='none'>no preview found for this one</div>"
    st.markdown(
        '<div class="flyer"><div class="tag">NOW PLAYING</div>'
        f'<div class="song">{html.escape(E["title"][top])}</div>'
        f'<div class="by">{html.escape(E["artist"][top])}</div>{eq}{note}</div>',
        unsafe_allow_html=True,
    )
    if info:
        st.audio(info["preview"], autoplay=True)

b1, b2 = st.columns(2)
b1.button("🤘 Listen", on_click=react, args=(top, True), type="primary")
b2.button("✖ Skip", on_click=react, args=(top, False))

likes = sum(1 for a, _ in ss.history if a == "like")
skips = sum(1 for a, _ in ss.history if a == "skip")
rate = f"{round(100 * likes / (likes + skips))}%" if likes + skips else "--"
st.markdown(
    '<div class="stickers">'
    f'<span class="sticker a">🤘 LOUD<b>{likes}</b></span>'
    f'<span class="sticker b">✖ SKIPPED<b>{skips}</b></span>'
    f'<span class="sticker c">HIT RATE<b>{rate}</b></span></div>',
    unsafe_allow_html=True,
)

heading("Set list: up next", "green")
items = "".join(
    f"<li><b>{html.escape(E['title'][i])}</b>"
    f"<span> — {html.escape(E['artist'][i])}</span></li>"
    for i in queue[1:]
)
st.markdown(
    f'<div class="setwrap"><div class="setlist"><div class="sethead">TONIGHT\'S SET</div>'
    f"<ol>{items}</ol></div></div>",
    unsafe_allow_html=True,
)

if HAS_PLOTLY:
    heading("The Pit (your taste map)", "yellow")
    st.plotly_chart(taste_map(queue))
else:
    st.caption("Want the taste map? Run: pip3 install plotly")

with st.sidebar:
    heading("Guest list", "pink")
    st.markdown(
        "".join(strip(f"{E['title'][i]} — {E['artist'][i]}") for i in ss.picks),
        unsafe_allow_html=True,
    )
    heading("Mosh log", "green")
    if ss.history:
        st.markdown(
            "".join(
                strip(f"{E['title'][i]} — {E['artist'][i]}", "" if a == "like" else "skip")
                for a, i in reversed(ss.history)
            ),
            unsafe_allow_html=True,
        )
    st.button("Start over", on_click=start_over)

st.caption("Previews courtesy of Apple's iTunes Search API.")