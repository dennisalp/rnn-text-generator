# python -i scrape.py "Taylor Swift" 250

import os
import sys
import subprocess
from pdb import set_trace as st

from lyricsgenius import Genius


artist = sys.argv[1]
max_songs = 9999 if len(sys.argv) == 2 else int(sys.argv[2])
token = open(".token", "r").read()

genius = Genius(token)
genius.remove_section_headers = True
genius.skip_non_songs = True
genius.excluded_terms = ['1989 ', '2021 BRIT Global', 'Grammys', "30 Things I Learned Before Turning 30", "Speeches", "(Acoustic)", "BBC ", "Billboard ", "(Radio Edit)"]

songs = genius.search_artist(artist, max_songs=max_songs, sort='popularity', get_full_info=True)
out = artist.lower().replace(' ', '_')
songs.save_lyrics(out, extension='txt', overwrite=True)

subprocess.call(["sed", "-i", "", "s/ *[0-9]*Embed */\\\n\\\n/g", out + '.txt'])
subprocess.call(["sed", "-i", "", "s/ Lyrics//g", out + '.txt'])
subprocess.call(["sed", "-i", "", "/^.\{128\}./d", out + '.txt'])
st()
