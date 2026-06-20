"""
1_build_dataset.py
Step 1: Pull song names from Spotify playlists → songs.csv
Step 2: Download audio from YouTube → data/<class>/raw/<song>.wav
Step 3: Chop into 30s clips → data/<class>/clips/<song>_000.wav ...
Step 4: Delete raw full-song files to save disk space
"""

import os
import re
import time
import subprocess
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SPOTIFY_CLIENT_ID     = "e96f45ac0d3349e69c29b6e526f40fd9"
SPOTIFY_CLIENT_SECRET = "392226fc500b431bb2ab7e89ea1bbee4"
SPOTIFY_REDIRECT_URI  = "http://127.0.0.1:3000"

PLAYLISTS = {
    "Sabrina": "6G2b4PJZtkK3FYfRX03Rli",
    "Taylor":  "62qKGvKdsDvcbepOW02ndW",
}

MAX_SONGS_PER_LANGUAGE = None

CLIP_DURATION   = 30
CLIPS_PER_SONG  = 3
OUTPUT_DIR      = "data"
CSV_PATH        = "songs.csv"
DELETE_RAW_AFTER_CHOPPING = True


# ─────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────
def get_spotify_client():
    return spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope="playlist-read-private playlist-read-collaborative",
        open_browser=True,
        cache_path=".spotify_token_cache"
    ))


# ─────────────────────────────────────────────
# STEP 1: Pull track names from Spotify
# ─────────────────────────────────────────────
def get_playlist_tracks(sp, playlist_id, language):
    tracks = []
    results = sp.playlist_tracks(playlist_id)

    while results:
        for item in results["items"]:
            track = item.get("item")   # this API response nests track data under 'item'
            if track is None:
                continue

            artist = track["artists"][0]["name"] if track["artists"] else "Unknown"
            tracks.append({
                "song_name": track["name"],
                "artist":    artist,
                "language":  language,
            })
        results = sp.next(results) if results["next"] else None

    return tracks


def build_csv():
    sp = get_spotify_client()

    all_tracks = []
    for language, playlist_id in PLAYLISTS.items():
        print(f"  Fetching {language} playlist...")
        tracks = get_playlist_tracks(sp, playlist_id, language)

        if MAX_SONGS_PER_LANGUAGE is not None:
            tracks = tracks[:MAX_SONGS_PER_LANGUAGE]

        print(f"  → using {len(tracks)} tracks")
        all_tracks.extend(tracks)

    df = pd.DataFrame(all_tracks).drop_duplicates(subset=["song_name"])
    df.to_csv(CSV_PATH, index=False)
    print(f"\n✓ Saved {len(df)} songs to {CSV_PATH}")
    return df


# ─────────────────────────────────────────────
# STEP 2: Download audio from YouTube
# ─────────────────────────────────────────────
def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()


def download_songs(df):
    import yt_dlp

    for _, row in df.iterrows():
        lang      = row["language"]
        song_name = sanitize_filename(row["song_name"])
        artist    = sanitize_filename(row["artist"])
        out_dir   = os.path.join(OUTPUT_DIR, lang, "raw")
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, f"{song_name}.wav")
        if os.path.exists(out_path):
            print(f"  SKIP (exists): {song_name}")
            continue

        query = f"{song_name} {artist} official audio"
        print(f"  Downloading: {song_name} ({lang})")

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(out_dir, f"{song_name}.%(ext)s"),
            "default_search": "ytsearch1",
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }],
            "quiet": True,
            "no_warnings": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([query])
            time.sleep(1)
        except Exception as e:
            print(f"  ERROR downloading {song_name}: {e}")
            continue


# ─────────────────────────────────────────────
# STEP 3+4: Chop into clips + delete raw files
# ─────────────────────────────────────────────
def chop_songs(df):
    for lang in df["language"].unique():
        raw_dir  = os.path.join(OUTPUT_DIR, lang, "raw")
        clip_dir = os.path.join(OUTPUT_DIR, lang, "clips")
        os.makedirs(clip_dir, exist_ok=True)

        if not os.path.exists(raw_dir):
            continue

        for fname in os.listdir(raw_dir):
            if not fname.endswith(".wav"):
                continue

            base       = fname.replace(".wav", "")
            input_path = os.path.join(raw_dir, fname)

            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries",
                 "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                 input_path],
                capture_output=True, text=True
            )
            try:
                duration = float(result.stdout.strip())
            except ValueError:
                print(f"  SKIP (can't read duration): {fname}")
                continue

            start_offset = 30
            clips_made   = 0

            for i in range(CLIPS_PER_SONG):
                start = start_offset + i * CLIP_DURATION
                if start + CLIP_DURATION > duration:
                    break

                out_clip = os.path.join(clip_dir, f"{base}_{i:03d}.wav")
                if os.path.exists(out_clip):
                    continue

                subprocess.run([
                    "ffmpeg", "-y",
                    "-ss", str(start),
                    "-i", input_path,
                    "-t", str(CLIP_DURATION),
                    "-ar", "22050",
                    "-ac", "1",
                    out_clip
                ], capture_output=True)
                clips_made += 1

            if clips_made:
                print(f"  Chopped {clips_made} clips: {base} ({lang})")

            if DELETE_RAW_AFTER_CHOPPING and clips_made > 0:
                os.remove(input_path)
                print(f"    Deleted raw file: {fname}")


# ─────────────────────────────────────────────
# STANDALONE AUTH TEST
# ─────────────────────────────────────────────
def test_auth_only():
    sp = get_spotify_client()
    me = sp.current_user()
    print(f"✓ Logged in as: {me['display_name']} ({me['id']})")

    # fix: was hardcoded to "Tamil", now uses the first key in PLAYLISTS dynamically
    first_class = list(PLAYLISTS.keys())[0]
    result = sp.playlist_tracks(PLAYLISTS[first_class], limit=1)
    first_track = result["items"][0]["item"]["name"]
    print(f"✓ Successfully read playlist. First track: {first_track}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # test_auth_only()
    # exit()

    print("=" * 50)
    print("STEP 1: Building song list from Spotify...")
    print("=" * 50)
    df = build_csv()

    print("\n" + "=" * 50)
    print("STEP 2: Downloading audio from YouTube...")
    print("=" * 50)
    download_songs(df)

    print("\n" + "=" * 50)
    print("STEP 3+4: Chopping into clips + deleting raw files...")
    print("=" * 50)
    chop_songs(df)

    print("\n✓ Done! Dataset structure:")
    for lang in df["language"].unique():
        clip_dir = os.path.join(OUTPUT_DIR, lang, "clips")
        if os.path.exists(clip_dir):
            n = len([f for f in os.listdir(clip_dir) if f.endswith(".wav")])
            print(f"  {lang}: {n} clips")