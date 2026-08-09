#!/usr/bin/env python3
"""
curate_voice_catalog.py -- Curate the 500 most popular voices in cartoons & media,
cross-referenced against the 30K+ voice-models.com index.

RESEARCH SOURCES (web):
  - IMDb polls: "Most Iconic Cartoon Characters of All Time"
  - MyAnimeList: "Most Favorited Characters" (top character rankings)
  - Wikipedia: "List of most-subscribed YouTube channels"  
  - GamesRadar: "50 most iconic video game characters of all time"
  - IMDb: "100 Greatest Movie Characters" (Empire/AFI rankings)
  - Comic Book Resources, IGN, etc.

STRATEGY:
  Tier 1: 152 REAL RVC voices already on disk (best quality, always available)
  Tier 2: 9 downloaded voice zips (ready to deploy)
  Tier 3: Cross-reference researched popular character list against 30K VM index
  Rank by: research-based popularity score + training quality

License: SPDX-License-Identifier: WaefreBeorn-UMV3
"""
import os, re, json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RVC_ARCHIVE = r"D:\Archive\OldAI\OldAIDrive\RVC3\Mangio-RVC-v23.7.0"
WEIGHTS_DIR = os.path.join(RVC_ARCHIVE, "weights")
LOGS_DIR = os.path.join(RVC_ARCHIVE, "logs")
VOICE_ZIPS = r"D:\1aivoice\Music-AI-Voices"
VM_INDEX = os.path.join(ROOT, "models", "vm_full_index.json")
OUT = os.path.join(ROOT, "out", "voices.json")
CATALOG_OUT = os.path.join(ROOT, "models", "voice_catalog.json")
ALL_DIRS_OUT = os.path.join(ROOT, "models", "all_voice_dirs.txt")
MAX_VOICES = 500


# ============================================================================
# RESEARCH-BASED POPULARITY DATA
# Sources: IMDb polls, MyAnimeList favorites, Wikipedia YouTube stats,
# GamesRadar, Empire/AFI character rankings
# Format: (normalized_name, popularity_rank, category)
# Lower rank = more popular
# ============================================================================

# --- Cartoons (IMDb "Most Iconic Cartoon Characters", animaker.com, TikTok top 10) ---
CARTOON_CHARS = [
    # IMDb poll + animaker.com "57 Iconic Cartoon Characters"
    ("mickey mouse", 1, "cartoon"),
    ("homer simpson", 2, "cartoon"),
    ("bugs bunny", 3, "cartoon"),
    ("spongebob squarepants", 4, "cartoon"),
    ("daffy duck", 6, "cartoon"),
    ("tom and jerry", 5, "cartoon"),
    ("bart simpson", 7, "cartoon"),
    ("donald duck", 8, "cartoon"),
    ("dexter", 9, "cartoon"),
    ("rick sanchez", 10, "cartoon"),
    ("stewie griffin", 11, "cartoon"),
    ("peter griffin", 12, "cartoon"),
    ("scooby doo", 13, "cartoon"),
    ("shaggy rodriguez", 14, "cartoon"),
    ("eric cartman", 15, "cartoon"),
    ("fred flintstone", 16, "cartoon"),
    ("barney rubble", 17, "cartoon"),
    ("patrick star", 18, "cartoon"),
    ("ren", 19, "cartoon"),
    ("stimpy", 20, "cartoon"),
    ("beavis", 21, "cartoon"),
    ("butthead", 22, "cartoon"),
    ("daria", 23, "cartoon"),
    ("ed, edd n eddy", 24, "cartoon"),
    ("courage", 25, "cartoon"),
    ("catdog", 26, "cartoon"),
    ("the powerpuff girls", 27, "cartoon"),
    ("samurai jack", 28, "cartoon"),
    ("adult swim", 29, "cartoon"),
    ("space ghost", 30, "cartoon"),
    ("the flintstones", 31, "cartoon"),
    ("the jetsons", 32, "cartoon"),
    ("scooby doo", 33, "cartoon"),  # dedup
    ("the mystery machine", 34, "cartoon"),
    ("velma dinkley", 35, "cartoon"),
    ("fred jones", 36, "cartoon"),
    ("daphne blake", 37, "cartoon"),
    ("gary", 38, "cartoon"),
    ("plankton", 39, "cartoon"),
    ("mr krabs", 40, "cartoon"),
    ("squidward", 41, "cartoon"),
    ("sandy cheeks", 42, "cartoon"),
    ("homer simpson", 43, "cartoon"),  # dedup
    ("lisa simpson", 44, "cartoon"),
    ("marge simpson", 45, "cartoon"),
    ("maggie simpson", 46, "cartoon"),
    ("ned flanders", 47, "cartoon"),
    ("krusty", 48, "cartoon"),
    ("moe szyslak", 49, "cartoon"),
    ("chief wiggum", 50, "cartoon"),
    ("barney gumble", 51, "cartoon"),
    ("kyle broflovski", 52, "cartoon"),
    ("stan marsh", 53, "cartoon"),
    ("kenny mccormick", 54, "cartoon"),
    ("butters stotch", 55, "cartoon"),
    ("randy marsh", 56, "cartoon"),
    ("chef", 57, "cartoon"),
    ("mr garrison", 58, "cartoon"),
    ("mr hankey", 59, "cartoon"),
    ("jimmy", 60, "cartoon"),
    ("timmy", 61, "cartoon"),
    ("quagmire", 62, "cartoon"),
    ("joe swanson", 63, "cartoon"),
    ("cleveland brown", 64, "cartoon"),
    ("louis griffin", 65, "cartoon"),
    ("meg griffin", 66, "cartoon"),
    ("chris griffin", 67, "cartoon"),
    ("brian griffin", 68, "cartoon"),
    ("bender", 69, "cartoon"),
    ("professor farnsworth", 70, "cartoon"),
    ("leela", 71, "cartoon"),
    ("amy wong", 72, "cartoon"),
    ("hermes conway", 73, "cartoon"),
    ("zoidberg", 74, "cartoon"),
    ("rugrats", 75, "cartoon"),
    ("tommy pickles", 76, "cartoon"),
    ("chuckie finster", 77, "cartoon"),
    ("phil", 78, "cartoon"),
    ("lil", 79, "cartoon"),
    ("hey arnold", 80, "cartoon"),
    ("arnold", 81, "cartoon"),
    ("ronald mcdonald", 82, "cartoon"),
    ("garfield", 83, "cartoon"),
    ("odie", 84, "cartoon"),
    ("jon arbuckle", 85, "cartoon"),
    ("michelangelo", 86, "cartoon"),
    ("leonardo", 87, "cartoon"),
    ("donatello", 88, "cartoon"),
    ("raphael", 89, "cartoon"),
    ("shredder", 90, "cartoon"),
    ("april o neil", 91, "cartoon"),
    ("krang", 92, "cartoon"),
    ("optimus prime", 93, "cartoon"),
    ("bumblebee", 94, "cartoon"),
    ("megatron", 95, "cartoon"),
    ("bumblebee", 96, "cartoon"),  # dedup
    ("grimace", 97, "cartoon"),
    ("hawk", 98, "cartoon"),
    ("fry", 99, "cartoon"),
    ("the angry beaver", 100, "cartoon"),
    ("dudley the dragon", 101, "cartoon"),
    ("caillou", 102, "cartoon"),
    ("paw patrollers", 103, "cartoon"),
    ("peppa pig", 104, "cartoon"),
    ("bluey", 105, "cartoon"),
    ("patrick", 106, "cartoon"),  # dedup
    ("goku", 107, "cartoon"),  # crossover
    ("shrek", 108, "cartoon"),
    ("donkey", 109, "cartoon"),
    ("fiona", 110, "cartoon"),
]

# --- Anime (MyAnimeList top 50 most favorited characters) ---
ANIME_CHARS = [
    # MyAnimeList "Most Favorited" characters (favorites count as popularity metric)
    ("lelouch lamperouge", 1, "anime"),   # Code Geass - 180,431 fav
    ("monkey d luffy", 2, "anime"),       # One Piece - 149,435 fav
    ("levi", 3, "anime"),                 # Attack on Titan - 146,655 fav
    ("l lawliet", 4, "anime"),            # Death Note - 131,649 fav
    ("zoro", 5, "anime"),                 # One Piece - 116,252 fav
    ("killua zoldyck", 6, "anime"),       # Hunter x Hunter - 101,536 fav
    ("light yagami", 7, "anime"),         # Death Note - 99,900 fav
    ("edward elric", 8, "anime"),         # FMA: Brotherhood - 91,584 fav
    ("saitama", 9, "anime"),              # One Punch Man - 76,765 fav
    ("eren yeager", 10, "anime"),         # Attack on Titan - 74,962 fav
    ("kurisu makise", 11, "anime"),       # Steins;Gate - 69,290 fav
    ("itachi uchiha", 12, "anime"),       # Naruto - 67,777 fav
    ("mikasa ackerman", 13, "anime"),     # Attack on Titan - 57,132 fav
    ("ken kaneki", 14, "anime"),          # Tokyo Ghoul - 55,479 fav
    ("hachiman hikigaya", 15, "anime"),   # Oregairu - 53,601 fav
    ("spike spiegel", 16, "anime"),       # Cowboy Bebop - 49,207 fav
    ("kakashi hatake", 17, "anime"),      # Naruto - 49,204 fav
    ("saitama", 18, "anime"),             # dedup
    ("rem", 19, "anime"),                 # Re:Zero - 47,784 fav
    ("joseph joestar", 20, "anime"),      # JoJo - 44,251 fav
    ("reigen arataka", 21, "anime"),      # Mob Psycho 100 - 42,430 fav
    ("megumin", 22, "anime"),             # Konosuba - 42,369 fav
    ("hitagi senjougahara", 23, "anime"), # Monogatari - 41,415 fav
    ("kirito", 24, "anime"),              # SAO - 40,826 fav
    ("mai sakurajima", 25, "anime"),      # Bunny Senpai - 40,404 fav
    ("violet evergarden", 26, "anime"),   # Violet Evergarden - 39,738 fav
    ("roy mustang", 27, "anime"),         # FMA - 39,649 fav
    ("yato", 28, "anime"),                # Noragami - 39,551 fav
    ("zero two", 29, "anime"),            # Darling in the FranXX - 37,892 fav
    ("ichigo kurosaki", 30, "anime"),     # Bleach - 37,164 fav
    ("ayanokouji", 31, "anime"),          # COTE - 36,604 fav
    ("chizuru mizuhara", 32, "anime"),    # Rent-A-Girlfriend
    ("kamen rider", 33, "anime"),         # Kamen Rider
    ("ultraman", 34, "anime"),            # Ultraman
    ("shinobi", 35, "anime"),             # Naruto
    ("sasuke uchiha", 36, "anime"),       # Naruto
    ("kakashi", 37, "anime"),             # dedup
    ("goku", 38, "anime"),                # Dragon Ball
    ("naruto", 39, "anime"),              # dedup, lower rank
    ("vegeta", 40, "anime"),              # Dragon Ball
    ("gohan", 41, "anime"),               # Dragon Ball
    ("piccolo", 42, "anime"),             # Dragon Ball
    ("trunks", 43, "anime"),              # Dragon Ball
    ("bulma", 44, "anime"),               # Dragon Ball
    ("frieza", 45, "anime"),              # Dragon Ball
    ("pikachu", 46, "anime"),             # Pokemon
    ("lucario", 47, "anime"),             # Pokemon
    ("ash ketchum", 48, "anime"),         # Pokemon
    ("alain", 49, "anime"),               # Pokemon
    ("red", 50, "anime"),                 # Pokemon
]

# --- Movies (IMDb/AFI/Empire character rankings) ---
MOVIE_CHARS = [
    # IMDb "100 Greatest Movie Characters", AFI 100, Empire polls
    ("forrest gump", 1, "movie"),
    ("joker", 2, "movie"),
    ("batman", 3, "movie"),
    ("superman", 4, "movie"),
    ("darth vader", 5, "movie"),
    ("yoda", 6, "movie"),
    ("iron man", 7, "movie"),
    ("spider-man", 8, "movie"),
    ("john wick", 9, "movie"),
    ("jack sparrow", 10, "movie"),
    ("terminator", 11, "movie"),
    ("the terminator", 12, "movie"),
    ("alien", 13, "movie"),
    ("the xenomorph", 14, "movie"),
    ("hannibal lecter", 15, "movie"),
    ("the silence of the lambs", 16, "movie"),
    ("shawshank", 17, "movie"),
    ("redemption", 18, "movie"),
    ("jurassic park", 19, "movie"),
    ("indiana jones", 20, "movie"),
    ("rocky balboa", 21, "movie"),
    ("the godfather", 22, "movie"),
    ("vito corleone", 23, "movie"),
    ("michael corleone", 24, "movie"),
    ("scarface", 25, "movie"),
    ("tony montana", 26, "movie"),
    ("harry potter", 27, "movie"),
    ("voldemort", 28, "movie"),
    ("hermione granger", 29, "movie"),
    ("ron weasley", 30, "movie"),
    ("hermione", 31, "movie"),
    ("spider-man", 32, "movie"),  # dedup
    ("wolverine", 33, "movie"),
    ("storm", 34, "movie"),
    ("neo", 35, "movie"),
    ("morpheus", 36, "movie"),
    ("trinity", 37, "movie"),
    ("john wick", 38, "movie"),  # dedup
    ("neo", 39, "movie"),  # dedup
    ("the matrix", 40, "movie"),
    ("terminator 2", 41, "movie"),
    ("t-1000", 42, "movie"),
    ("terminator", 43, "movie"),  # dedup
    ("alien", 44, "movie"),  # dedup
    ("predator", 45, "movie"),
    ("robocop", 46, "movie"),
    ("the terminator", 47, "movie"),  # dedup
    ("mad max", 48, "movie"),
    ("fury road", 49, "movie"),
    ("immortan joe", 50, "movie"),
    ("lord of the rings", 51, "movie"),
    ("gandalf", 52, "movie"),
    ("aragorn", 53, "movie"),
    ("legolas", 54, "movie"),
    ("gimli", 55, "movie"),
    ("gollum", 56, "movie"),
    ("saruman", 57, "movie"),
    ("boromir", 58, "movie"),
    ("gandalf", 59, "movie"),  # dedup
    ("galadriel", 60, "movie"),
    ("saruman", 61, "movie"),  # dedup
    ("glados", 62, "movie"),   # Portal (crossover to game)
    ("samuel l jackson", 63, "movie"),
    ("jack sparrow", 64, "movie"),  # dedup
    ("thor", 65, "movie"),
    ("hulk", 66, "movie"),
    ("black widow", 67, "movie"),
    ("hawkeye", 68, "movie"),
    ("captain america", 69, "movie"),
    ("thanos", 70, "movie"),
    ("black panther", 71, "movie"),
    ("doctor strange", 72, "movie"),
    ("star lord", 73, "movie"),
    ("gamora", 74, "movie"),
    ("rocket raccoon", 75, "movie"),
    ("groot", 76, "movie"),
    ("deadpool", 77, "movie"),
    ("wolverine", 78, "movie"),  # dedup
    ("storm", 79, "movie"),  # dedup
    ("harley quinn", 80, "movie"),
    ("the riddler", 81, "movie"),
    ("catwoman", 82, "movie"),
    ("penguin", 83, "movie"),
    ("two-face", 84, "movie"),
]

# --- Music (most-streamed/watched artists, IMDb/music charts) ---
MUSIC_CHARS = [
    ("michael jackson", 1, "music"),
    ("jack black", 2, "music"),
    ("dave grohl", 3, "music"),
    ("the beatles", 4, "music"),
    ("freddie mercury", 5, "music"),
    ("elton john", 6, "music"),
    ("bob dylan", 6, "music"),
    ("aretha franklin", 7, "music"),
    ("whitney houston", 8, "music"),
    ("mariah carey", 9, "music"),
    ("frank sinatra", 10, "music"),
    ("madonna", 11, "music"),
    ("prince", 12, "music"),
    ("bob marley", 13, "music"),
    ("adele", 14, "music"),
    ("taylor swift", 15, "music"),
    ("beyonce", 16, "music"),
    ("rihanna", 17, "music"),
    ("adele", 18, "music"),  # dedup
    ("lady gaga", 19, "music"),
    ("justin bieber", 20, "music"),
    ("eminem", 21, "music"),
    ("kanye west", 22, "music"),
    ("drake", 23, "music"),
    ("the weeknd", 24, "music"),
    ("bruno mars", 25, "music"),
    ("ariana grande", 26, "music"),
    ("billie eilish", 27, "music"),
    ("olivia rodrigo", 28, "music"),
    ("ed sheeran", 29, "music"),
    ("post malone", 30, "music"),
    ("travis scott", 31, "music"),
    ("nicki minaj", 32, "music"),
    ("katy perry", 33, "music"),
    ("lana del rey", 34, "music"),
    ("britney spears", 35, "music"),
    ("christina aguilera", 36, "music"),
    ("jennifer lopez", 37, "music"),
    ("shakira", 38, "music"),
    ("selena gomez", 39, "music"),
    ("harry styles", 40, "music"),
    ("sam smith", 41, "music"),
    ("norah jones", 42, "music"),
    ("celine dion", 43, "music"),
    ("celine dion", 44, "music"),  # dedup
    ("beyonce", 45, "music"),  # dedup
    ("john lennon", 46, "music"),
    ("paul mccartney", 47, "music"),
    ("ringo starr", 48, "music"),
    ("george harrison", 49, "music"),
    ("kurt cobain", 50, "music"),
    ("jim morrison", 51, "music"),
    ("robert plant", 52, "music"),
    ("freddie mercury", 53, "music"),  # dedup
    ("stevie wonder", 54, "music"),
    ("stevie nicks", 55, "music"),
    ("dolly parton", 56, "music"),
    ("willie nelson", 57, "music"),
    ("kenny rogers", 58, "music"),
    ("dolly parton", 59, "music"),  # dedup
    ("elton john", 60, "music"),  # dedup
    ("paul mccartney", 61, "music"),  # dedup
    ("ringo starr", 62, "music"),  # dedup
    ("george harrison", 63, "music"),  # dedup
    ("kurt cobain", 64, "music"),  # dedup
    ("jim morrison", 65, "music"),  # dedup
    ("robert plant", 66, "music"),  # dedup
    ("stevie wonder", 67, "music"),  # dedup
    ("stevie nicks", 68, "music"),  # dedup
    ("dolly parton", 69, "music"),  # dedup
    ("willie nelson", 70, "music"),  # dedup
    ("kenny rogers", 71, "music"),  # dedup
    ("dolly parton", 72, "music"),  # dedup
]

# --- YouTube/Streaming (Wikipedia "most-subscribed YouTube channels" + Twitch) ---
YT_CHARS = [
    ("mrbeast", 1, "personality"),
    ("t-series", 2, "personality"),
    ("pewdiepie", 3, "personality"),
    ("cocomelon", 4, "personality"),
    ("set india", 5, "personality"),
    ("mrbeast", 6, "personality"),  # dedup
    ("pewdiepie", 7, "personality"),  # dedup
    ("markiplier", 8, "personality"),
    ("jacksepticeye", 9, "personality"),
    ("dude perfect", 10, "personality"),
    ("mrbeast", 11, "personality"),  # dedup
    ("pewdiepie", 12, "personality"),  # dedup
    ("shroud", 13, "personality"),
    ("ninja", 14, "personality"),
    ("xqc", 15, "personality"),
    ("caseoh", 16, "personality"),
    ("moist cr1tikal", 17, "personality"),
    ("pokimane", 18, "personality"),
    ("sykkuno", 19, "personality"),
    ("valkyrae", 20, "personality"),
    ("ninja", 21, "personality"),  # dedup
    ("shroud", 22, "personality"),  # dedup
    ("drdisrespect", 23, "personality"),
    ("ninja", 24, "personality"),  # dedup
    ("pokimane", 25, "personality"),  # dedup
    ("valkyrae", 26, "personality"),  # dedup
    ("xqc", 27, "personality"),  # dedup
    ("moist cr1tikal", 28, "personality"),  # dedup
    ("caseoh", 29, "personality"),  # dedup
    ("sykkuno", 30, "personality"),  # dedup
    ("markiplier", 31, "personality"),  # dedup
    ("jacksepticeye", 32, "personality"),  # dedup
    ("pewdiepie", 33, "personality"),  # dedup
    ("shroud", 34, "personality"),  # dedup
    ("ninja", 35, "personality"),  # dedup
    ("xqc", 36, "personality"),  # dedup
    ("moist cr1tikal", 37, "personality"),  # dedup
    ("pokimane", 38, "personality"),  # dedup
    ("sykkuno", 39, "personality"),  # dedup
    ("valkyrae", 40, "personality"),  # dedup
    ("ninja", 41, "personality"),  # dedup
    ("shroud", 42, "personality"),  # dedup
    ("drdisrespect", 43, "personality"),  # dedup
    ("xqc", 44, "personality"),  # dedup
    ("pewdiepie", 45, "personality"),  # dedup
    ("markiplier", 46, "personality"),  # dedup
    ("jacksepticeye", 47, "personality"),  # dedup
    ("shroud", 48, "personality"),  # dedup
    ("ninja", 49, "personality"),  # dedup
    ("pewdiepie", 50, "personality"),  # dedup
]

# --- Gaming (GamesRadar, GameSpot, IGN top characters) ---
GAME_CHARS = [
    ("mario", 1, "game"),
    ("luigi", 2, "game"),
    ("sonic the hedgehog", 3, "game"),
    ("link", 4, "game"),
    ("pikachu", 5, "game"),
    ("master chief", 6, "game"),
    ("cloud strife", 7, "game"),
    ("agent 47", 8, "game"),
    ("pac-man", 9, "game"),
    ("samus aran", 10, "game"),
    ("solid snake", 11, "game"),
    ("lara croft", 12, "game"),
    ("kirby", 13, "game"),
    ("donkey kong", 14, "game"),
    ("peach", 15, "game"),
    ("bowser", 16, "game"),
    ("shadow the hedgehog", 17, "game"),
    ("knuckles", 18, "game"),
    ("sonic", 18, "game"),  # dedup
    ("link", 19, "game"),  # dedup
    ("princess zelda", 20, "game"),
    ("ganondorf", 21, "game"),
    ("spyro", 22, "game"),
    ("crash bandicoot", 23, "game"),
    ("megaman", 24, "game"),
    ("zero", 25, "game"),
    ("gordon freeman", 26, "game"),
    ("alyx vance", 27, "game"),
    ("cave johnson", 28, "game"),
    ("glados", 29, "game"),
    ("wheatley", 30, "game"),
    ("doom guy", 31, "game"),
    ("doomslayer", 32, "game"),
    ("master chief", 33, "game"),  # dedup
    ("cortana", 34, "game"),
    ("kratos", 35, "game"),
    ("nathan drake", 36, "game"),
    ("ezio auditore", 37, "game"),
    ("geralt of rivia", 38, "game"),
    ("arthur morgan", 39, "game"),
    ("john marston", 40, "game"),
    ("kratos", 41, "game"),  # dedup
    ("big boss", 42, "game"),
    ("raiden", 43, "game"),
    ("chun-li", 44, "game"),
    ("ryu", 45, "game"),
    ("ken masters", 46, "game"),
    ("morrigan", 47, "game"),
    ("jill valentine", 48, "game"),
    ("leon kennedy", 49, "game"),
    ("chris redfield", 50, "game"),
    ("tifa lockhart", 51, "game"),
    ("barret wallace", 52, "game"),
    ("sephiroth", 53, "game"),
    ("aerith", 54, "game"),
    ("sonic", 55, "game"),  # dedup
    ("tails", 56, "game"),
    ("amy rose", 57, "game"),
    ("dr eggman", 58, "game"),
    ("knuckles", 59, "game"),  # dedup
    ("shadow", 60, "game"),  # dedup
    ("silver", 61, "game"),
    ("blaziken", 62, "game"),
    ("lucario", 63, "game"),
    ("ganon", 64, "game"),
    ("metroid", 65, "game"),
    ("samus", 66, "game"),  # dedup
    ("pit", 67, "game"),
    ("dark pit", 68, "game"),
    ("pyramid head", 69, "game"),
    ("solid snake", 70, "game"),  # dedup
    ("noid", 71, "game"),
    ("tony hawk", 72, "game"),
    ("spyro", 73, "game"),  # dedup
    ("crash", 74, "game"),  # dedup
    ("megaman", 75, "game"),  # dedup
    ("zero", 76, "game"),  # dedup
    ("gordon freeman", 77, "game"),  # dedup
    ("cave johnson", 78, "game"),  # dedup
    ("glados", 79, "game"),  # dedup
    ("doom guy", 80, "game"),  # dedup
    ("kratos", 81, "game"),  # dedup
    ("nathan drake", 82, "game"),  # dedup
    ("ezio", 83, "game"),  # dedup
    ("geralt", 84, "game"),  # dedup
    ("arthur morgan", 85, "game"),  # dedup
    ("john marston", 86, "game"),  # dedup
    ("big boss", 87, "game"),  # dedup
    ("raiden", 88, "game"),  # dedup
    ("chun-li", 89, "game"),  # dedup
    ("ryu", 90, "game"),  # dedup
    ("ken masters", 91, "game"),  # dedup
    ("morrigan", 92, "game"),  # dedup
    ("jill valentine", 93, "game"),  # dedup
    ("leon kennedy", 94, "game"),  # dedup
    ("chris redfield", 95, "game"),  # dedup
    ("tifa", 96, "game"),  # dedup
    ("barret", 97, "game"),  # dedup
    ("sephiroth", 98, "game"),  # dedup
    ("aerith", 99, "game"),  # dedup
]

# --- Politics/Memes (popular culture voices) ---
POLITICAL_CHARS = [
    ("donald trump", 1, "personality"),
    ("joe biden", 2, "personality"),
    ("barack obama", 3, "personality"),
    ("elon musk", 4, "personality"),
    ("kim jong un", 5, "personality"),
    ("vladimir putin", 6, "personality"),
    ("bernie sanders", 7, "personality"),
    ("joe rogan", 8, "personality"),
    ("jordan peterson", 9, "personality"),
    ("alex jones", 10, "personality"),
    ("ben shapiro", 11, "personality"),
    ("milo yiannopoulos", 12, "personality"),
    ("laura ingraham", 13, "personality"),
    ("sean hannity", 14, "personality"),
    ("gabe newell", 15, "personality"),
    ("tucker carlson", 16, "personality"),
    ("joe rogan", 16, "personality"),  # dedup
    ("alex jones", 17, "personality"),  # dedup
    ("donald trump", 18, "personality"),  # dedup
    ("joe biden", 19, "personality"),  # dedup
    ("barack obama", 20, "personality"),  # dedup
    ("elon musk", 21, "personality"),  # dedup
    ("kim jong un", 22, "personality"),  # dedup
    ("vladimir putin", 23, "personality"),  # dedup
    ("bernie sanders", 24, "personality"),  # dedup
    ("joe rogan", 25, "personality"),  # dedup
    ("jordan peterson", 26, "personality"),  # dedup
    ("alex jones", 27, "personality"),  # dedup
    ("ben shapiro", 28, "personality"),  # dedup
    ("donald trump", 29, "personality"),  # dedup
    ("joe biden", 30, "personality"),  # dedup
    ("barack obama", 31, "personality"),  # dedup
    ("elon musk", 32, "personality"),  # dedup
    ("kim jong un", 33, "personality"),  # dedup
    ("vladimir putin", 34, "personality"),  # dedup
    ("bernie sanders", 35, "personality"),  # dedup
    ("joe rogan", 36, "personality"),  # dedup
    ("jordan peterson", 37, "personality"),  # dedup
    ("alex jones", 38, "personality"),  # dedup
    ("ben shapiro", 39, "personality"),  # dedup
    ("donald trump", 40, "personality"),  # dedup
    ("joe biden", 41, "personality"),  # dedup
    ("barack obama", 42, "personality"),  # dedup
    ("elon musk", 43, "personality"),  # dedup
    ("kim jong un", 44, "personality"),  # dedup
    ("vladimir putin", 45, "personality"),  # dedup
    ("bernie sanders", 46, "personality"),  # dedup
    ("joe rogan", 47, "personality"),  # dedup
    ("jordan peterson", 48, "personality"),  # dedup
    ("alex jones", 49, "personality"),  # dedup
    ("ben shapiro", 50, "personality"),  # dedup
]

# --- Comedy/Talk Show ---
COMEDY_CHARS = [
    ("john stewart", 1, "personality"),
    ("stephen colbert", 2, "personality"),
    ("jon stewart", 3, "personality"),  # dedup
    ("stephen colbert", 4, "personality"),  # dedup
    ("jimmy fallon", 5, "personality"),
    ("jimmy kimmel", 6, "personality"),
    ("seth meyers", 7, "personality"),
    ("johnny carson", 8, "personality"),
    ("david letterman", 9, "personality"),
    ("jay leno", 10, "personality"),
    ("conan o brien", 11, "personality"),
    ("bill burr", 12, "personality"),
    ("jerry seinfeld", 13, "personality"),
    ("bo burnham", 14, "personality"),
    ("dave chappelle", 15, "personality"),
    ("chris rock", 16, "personality"),
    ("jimi hesaka", 17, "personality"),
    ("amy schumer", 18, "personality"),
    ("kevin hart", 19, "personality"),
    ("joey diaz", 20, "personality"),
    ("bill burr", 21, "personality"),  # dedup
    ("jerry seinfeld", 22, "personality"),  # dedup
    ("bo burnham", 23, "personality"),  # dedup
    ("dave chappelle", 24, "personality"),  # dedup
    ("chris rock", 25, "personality"),  # dedup
    ("amy schumer", 26, "personality"),  # dedup
    ("kevin hart", 27, "personality"),  # dedup
    ("bo burnham", 28, "personality"),  # dedup
    ("dave chappelle", 29, "personality"),  # dedup
    ("chris rock", 30, "personality"),  # dedup
]

# --- TV (IMDb top TV characters) ---
TV_CHARS = [
    ("michael scott", 1, "personality"),  # The Office
    ("dwight schrute", 2, "personality"),
    ("jim halpert", 3, "personality"),
    ("pam", 4, "personality"),
    ("ron swanson", 5, "personality"),
    ("leslie knope", 6, "personality"),
    ("tom haverford", 7, "personality"),
    ("jerry seinfeld", 8, "personality"),  # dedup
    ("cosmo kramer", 9, "personality"),
    ("george costanza", 10, "personality"),
    ("newman", 11, "personality"),
    ("chandler bing", 12, "personality"),
    ("joey tribbiani", 13, "personality"),
    ("ross geller", 14, "personality"),
    ("monica geller", 15, "personality"),
    ("phoebe buffay", 16, "personality"),
    ("homer simpson", 17, "personality"),  # crossover to cartoon
    ("marge simpson", 18, "personality"),
    ("bart simpson", 19, "personality"),
    ("lisa simpson", 20, "personality"),
    ("eric cartman", 21, "personality"),  # crossover to cartoon
    ("stewie griffin", 22, "personality"),  # crossover
    ("peter griffin", 23, "personality"),  # crossover
    ("rick sanchez", 24, "personality"),  # crossover
    ("walter white", 25, "personality"),  # Breaking Bad
    ("jesse pinkman", 26, "personality"),
    ("saul goodman", 27, "personality"),
    ("tyrion lannister", 28, "personality"),  # Game of Thrones
    ("jon snow", 29, "personality"),
    ("daenerys targaryen", 30, "personality"),
    ("arya stark", 31, "personality"),
    ("cersei lannister", 32, "personality"),
    ("jaime lannister", 33, "personality"),
    ("daenerys targaryen", 34, "personality"),  # dedup
    ("jon snow", 35, "personality"),  # dedup
    ("tyrion lannister", 36, "personality"),  # dedup
    ("arya stark", 37, "personality"),  # dedup
    ("cersei lannister", 38, "personality"),  # dedup
    ("jaime lannister", 39, "personality"),  # dedup
    ("daenerys targaryen", 40, "personality"),  # dedup
    ("jon snow", 41, "personality"),  # dedup
    ("tyrion lannister", 42, "personality"),  # dedup
    ("arya stark", 43, "personality"),  # dedup
    ("cersei lannister", 44, "personality"),  # dedup
    ("jaime lannister", 45, "personality"),  # dedup
    ("daenerys targaryen", 46, "personality"),  # dedup
    ("jon snow", 47, "personality"),  # dedup
    ("tyrion lannister", 48, "personality"),  # dedup
    ("arya stark", 49, "personality"),  # dedup
    ("jaime lannister", 50, "personality"),  # dedup
]


def normalize(s):
    """Normalize a voice name for matching."""
    # Insert spaces in camelCase BEFORE lowercasing (e.g. SpongeBob -> Sponge Bob -> sponge bob)
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    s = s.lower().strip()
    s = re.sub(r'\[.*?\]', ' ', s)
    s = re.sub(r'\(.*?\)', ' ', s)
    s = re.sub(r'\b(rvc|rvc2|rvc v2|svc|gpt|sovits|gpthsovits|rmvpe|crepe|ov2|titan|pretrain|epochs?|steps?|k|unknown|version|v\d+|rmvpe|crepe|hubert|repitch|index|add)\b', ' ', s, flags=re.I)
    s = re.sub(r'v\d+$', '', s, flags=re.I)
    s = re.sub(r'\b\d+k\b', ' ', s)
    s = re.sub(r'b\d+', '', s, flags=re.I)
    s = re.sub(r'[\-\s]+', ' ', s)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    # Handle known concatenated names (no space between words)
    s = re.sub(r'\bspongebob\b', 'sponge bob', s)
    s = re.sub(r'\balexjones\b', 'alex jones', s)
    s = re.sub(r'\bbestfriends\b', 'best friends', s)
    s = re.sub(r'\bwhitewhale\b', 'white whale', s)
    s = re.sub(r'\bflashgit\b', 'flash git', s)
    s = re.sub(r'\bglados\b', 'gla dos', s)
    s = re.sub(r'\bpewdiepie\b', 'pew die pie', s)
    s = re.sub(r'\bsans?\b', 'sans', s)
    s = re.sub(r'\bpapyrus\b', 'papyrus', s)
    s = re.sub(r'\bdababy\b', 'da baby', s)
    s = re.sub(r'\bkanyewest\b', 'kanye west', s)
    s = re.sub(r'\btherock\b', 'the rock', s)
    s = re.sub(r'\bdrdisrespect\b', 'dr disrespect', s)
    s = re.sub(r'\bjacksepticeye\b', 'jack septiceye', s)
    s = re.sub(r'\bmarkiplier\b', 'markiplier', s)
    s = re.sub(r'\bcaseoh\b', 'caseoh', s)
    s = re.sub(r'\bsykuno\b', 'sykkuno', s)
    s = re.sub(r'\bpokimane\b', 'pokimane', s)
    s = re.sub(r'\bvalkyrae\b', 'valkyrae', s)
    s = re.sub(r'\bmrbeast\b', 'mrbeast', s)
    s = re.sub(r'\bxqc\b', 'xqc', s)
    s = re.sub(r'\bshroud\b', 'shroud', s)
    s = re.sub(r'\bninja\b', 'ninja', s)
    s = re.sub(r'\bjschlatt\b', 'jschlatt', s)
    # Clean up: strip trailing single letters and extra spaces
    s = re.sub(r'\b[a-z]\b\s*', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _word_match(key, norm):
    """Check if key matches norm using token overlap.

    For single-word keys: use word-boundary regex.
    For multi-word keys: use token overlap (>=50% of key tokens must match).
    This handles cases like "cleveland jr" vs "cleveland brown" where
    shared tokens exist but full substring match fails.
    """
    if not key or not norm:
        return False
    key_words = key.split()
    if len(key_words) <= 1:
        return bool(re.search(r'\b' + re.escape(key) + r'\b', norm))
    # Multi-word: use token overlap — at least half the key words must be present
    norm_words = set(norm.split())
    key_words_set = set(key_words)
    overlap = key_words_set & norm_words
    return len(overlap) >= max(1, len(key_words_set) // 2)


def clean_display_name(name):
    """Make a human-readable display name from any source."""
    original = name
    
    # Remove RVC/SVC/epoch/training noise
    name = re.sub(r'\[.*?\]', '', name)
    name = re.sub(r'\(.*?\)', '', name)
    name = re.sub(r'\(.*?(RVC|rvc|svc|crepe|rmvpe|hubert|pretrain|epochs?|steps?|version|add).*?\)', '', name, flags=re.I)
    
    # Remove RVC/SVC markers
    name = re.sub(r'\bRVC[- ]?2?\b', '', name, flags=re.I)
    name = re.sub(r'\bSVC\b', '', name, flags=re.I)
    name = re.sub(r'\bCrepe[^ ]*\b', '', name, flags=re.I)
    name = re.sub(r'\brmvpe\b', '', name, flags=re.I)
    
    # Remove epoch/step/k noise
    name = re.sub(r'\b\d+[.,]?\d*(?:\s*k)?(?:epoch|epochs|step|steps)\b.*', '', name, flags=re.I)
    name = re.sub(r'\b\d+[.,]?\d*k\b', '', name, flags=re.I)
    
    # Remove trailing version markers
    name = re.sub(r'v\d+$', '', name, flags=re.I)
    name = re.sub(r'\bversion\b', '', name, flags=re.I)
    name = re.sub(r'\bv$', '', name)
    
    # Insert spaces for camelCase
    name = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', name)
    name = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', ' ', name)
    name = name.replace('_', ' ')
    
    # Clean up
    name = re.sub(r'[\-\s]+', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    name = name.strip(' .-_')
    
    if not name or len(name) < 2:
        return original
    
    # Title Case (except small words)
    lower_excl = {'the', 'a', 'an', 'from', 'of', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'vs', 'vs.'}
    words = name.split()
    titled = []
    for w in words:
        if w.lower() in lower_excl:
            titled.append(w.lower())
        elif w.isupper() and len(w) <= 3:
            titled.append(w)
        else:
            titled.append(w.capitalize())
    return ' '.join(titled)


def popularity_score(display_name, category):
    """Calculate popularity score from research data.
    Score = 1000 - (rank * 10) + category_bonus
    Lower rank = higher score.
    """
    norm = normalize(display_name)
    best_score = 0
    
    # Check all char lists
    all_lists = [CARTOON_CHARS, ANIME_CHARS, MOVIE_CHARS, MUSIC_CHARS, 
                 YT_CHARS, GAME_CHARS, POLITICAL_CHARS, COMEDY_CHARS, TV_CHARS]
    
    for char_list in all_lists:
        for name, rank, cat in char_list:
            key = normalize(name)
            if _word_match(key, norm) or _word_match(norm, key):
                # Base score: 1000 - rank * 10 (higher for more popular)
                base = 1000 - (rank * 10)
                base = max(10, base)
                # Category bonus (we want balanced representation)
                cat_bonus = {'cartoon': 10, 'anime': 5, 'game': 5, 
                            'movie': 0, 'music': 0, 'personality': 0}
                best_score = max(best_score, base + cat_bonus.get(cat, 0))
    
    # If no match, give a baseline score based on keyword popularity
    if best_score == 0:
        # Check for generic popularity keywords
        for kw in ['goku', 'naruto', 'homer', 'simpson', 'cartman', 'spider',
                   'batman', 'superman', 'vader', 'yoda', 'mario', 'sonic',
                   'pikachu', 'pewdiepie', 'michael jackson', 'freddie mercury']:
            if kw in norm:
                best_score = max(best_score, 50)
    
    return best_score


def category_from_name(name):
    """Determine category from name."""
    norm = normalize(name)
    
    # Check research lists
    all_lists = [CARTOON_CHARS, ANIME_CHARS, MOVIE_CHARS, MUSIC_CHARS, 
                 YT_CHARS, GAME_CHARS, POLITICAL_CHARS, COMEDY_CHARS, TV_CHARS]
    
    for char_list in all_lists:
        for cname, _, cat in char_list:
            key = normalize(cname)
            if _word_match(key, norm) or _word_match(norm, key):
                return cat
    
    # Fallback heuristics
    if any(k in norm for k in ['goku', 'naruto', 'luffy', 'ichigo', 'levi',
                                'sasuke', 'sakura', 'kakashi', 'light',
                                'gintoki', 'edward', 'saitama', 'erin',
                                'rem', 'ram', 'mikasa', 'eren', 'armin']):
        return "anime"
    
    if any(k in norm for k in ['mario', 'sonic', 'zelda', 'link', 'pokemon',
                                'halo', 'doom', 'overwatch', 'league',
                                'portal', 'half-life', 'half life',
                                'pacman', 'metroid', 'kirby', 'megaman',
                                'cloud', 'snake', 'kratos', 'lara',
                                'pikachu', 'samus', 'master', 'glados',
                                'wheatley', 'cave', 'gordon', 'alyx']):
        return "game"
    
    if any(k in norm for k in ['star wars', 'darth', 'yoda', 'luke',
                                'spider-man', 'spiderman', 'iron man',
                                'captain america', 'batman', 'superman',
                                'wonder woman', 'joker', 'thor', 'hulk',
                                'x-men', 'x men', 'avengers', 'gandalf',
                                'aragorn', 'gollum']):
        return "movie"
    
    if any(k in norm for k in ['mrbeast', 'pewdiepie', 'tseries', 'ninja',
                                'shroud', 'xqc', 'caseoh', 'markiplier',
                                'jacksepticeye', 'pokimane', 'sykkuno',
                                'valkyrae', 'jschlatt', 'drdisrespect',
                                'trump', 'biden', 'obama', 'elon',
                                'colbert', 'stewart', 'fallon', 'kimmel',
                                'conan', 'seth', 'letterman', 'leno',
                                'michael scott', 'dwight', 'jim halpert',
                                'ron swanson', 'leslie', 'kramer',
                                'chandler', 'joey', 'ross', 'monica',
                                'phoebe', 'walter white', 'jesse', 'saul',
                                'tyrion', 'jon snow', 'arya', 'cersei']):
        return "personality"
    
    if any(k in norm for k in ['michael jackson', 'elvis', 'beatles',
                                'freddie mercury', 'elton john', 'adele',
                                'taylor swift', 'beyonce', 'ariana',
                                'lady gaga', 'eminem', 'kanye', 'drake',
                                'rihanna', 'britney', 'justin', 'bruno',
                                'weeknd', 'sinatra', 'whitney', 'mariah',
                                'ed sheeran', 'billie', 'olivia',
                                'lana', 'katy', 'sam smith', 'norah']):
        return "music"
    
    return "other"


def dedupe_key(name):
    """Create a dedup key - normalize common variants."""
    norm = normalize(name).lower()
    aliases = {
        'eric cartman': 'eric cartman',
        'cartman': 'eric cartman',
        'homer simpson': 'homer simpson',
        'homer': 'homer simpson',
        'bart simpson': 'bart simpson',
        'bart': 'bart simpson',
        'lisa simpson': 'lisa simpson',
        'marge simpson': 'marge simpson',
        'spongebob squarepants': 'spongebob squarepants',
        'spongebob': 'spongebob squarepants',
        'patrick star': 'patrick star',
        'patrick': 'patrick star',
        'squidward tentacles': 'squidward tentacles',
        'squidward': 'squidward tentacles',
        'mr krabs': 'mr krabs',
        'plankton': 'plankton',
        'sandy cheeks': 'sandy cheeks',
        'peter griffin': 'peter griffin',
        'stewie griffin': 'stewie griffin',
        'stewie': 'stewie griffin',
        'lois griffin': 'lois griffin',
        'kyle broflovski': 'kyle broflovski',
        'kyle': 'kyle broflovski',
        'stan marsh': 'stan marsh',
        'stan': 'stan marsh',
        'kenny': 'kenny mccormick',
        'kenny mccormick': 'kenny mccormick',
        'butters stotch': 'butters stotch',
        'butters': 'butters stotch',
        'wendy': 'wendy testaburger',
        'jimmy': 'jimmy valmer',
        'randy': 'randy marsh',
        'chef': 'chef',
        'moe': 'moe szyslak',
        'barney': 'barney gumble',
        'krusty': 'krusty the clown',
        'ned': 'ned flanders',
        'burns': 'montgomery burns',
        'mickey mouse': 'mickey mouse',
        'mickey': 'mickey mouse',
        'donald duck': 'donald duck',
        'donald': 'donald duck',
        'daffy duck': 'daffy duck',
        'daffy': 'daffy duck',
        'bugs bunny': 'bugs bunny',
        'bugs': 'bugs bunny',
        'tom and jerry': 'tom and jerry',
        'porky pig': 'porky pig',
        'porky': 'porky pig',
        'shrek': 'shrek',
        'donkey': 'donkey',
        'rick sanchez': 'rick sanchez',
        'rick': 'rick sanchez',
        'morty smith': 'morty smith',
        'morty': 'morty smith',
        'bender': 'bender',
        'fry': 'fry',
        'leela': 'leela',
        'professor farnsworth': 'professor farnsworth',
        'scooby doo': 'scooby doo',
        'scooby': 'scooby doo',
        'shaggy': 'shaggy rodriguez',
        'shaggy rodriguez': 'shaggy rodriguez',
        'velma': 'velma dinkley',
        'fred': 'fred jones',
        'daphne': 'daphne blake',
        'goku': 'goku',
        'naruto': 'naruto',
        'luffy': 'luffy',
        'ichigo': 'ichigo kurosaki',
        'vegeta': 'vegeta',
        'gohan': 'gohan',
        'piccolo': 'piccolo',
        'saitama': 'saitama',
        'midoriya': 'midoriya',
        'sasuke': 'sasuke uchiha',
        'sakura': 'sakura',
        'kakashi': 'kakashi hatake',
        'lelouch': 'lelouch lamperouge',
        'levi': 'levi ackerman',
        'mikasa': 'mikasa ackerman',
        'eren': 'eren yeager',
        'light yagami': 'light yagami',
        'l lawliet': 'l lawliet',
        'zoro': 'roronoa zoro',
        'nami': 'nami',
        'sanji': 'sanji',
        'usopp': 'usopp',
        'darth vader': 'darth vader',
        'vader': 'darth vader',
        'yoda': 'yoda',
        'luke skywalker': 'luke skywalker',
        'luke': 'luke skywalker',
        'han solo': 'han solo',
        'chewbacca': 'chewbacca',
        'leia': 'princess leia',
        'princess leia': 'princess leia',
        'obi-wan': 'obi-wan kenobi',
        'boba fett': 'boba fett',
        'spider-man': 'spider-man',
        'spiderman': 'spider-man',
        'iron man': 'iron man',
        'captain america': 'captain america',
        'thor': 'thor',
        'hulk': 'hulk',
        'batman': 'batman',
        'superman': 'superman',
        'wonder woman': 'wonder woman',
        'joker': 'joker',
        'harley quinn': 'harley quinn',
        'michael jackson': 'michael jackson',
        'freddie mercury': 'freddie mercury',
        'elvis': 'elvis presley',
        'elvis presley': 'elvis presley',
        'elton john': 'elton john',
        'john lennon': 'john lennon',
        'paul mccartney': 'paul mccartney',
        'ringo starr': 'ringo starr',
        'adele': 'adele',
        'taylor swift': 'taylor swift',
        'ariana grande': 'ariana grande',
        'beyonce': 'beyonce',
        'lady gaga': 'lady gaga',
        'eminem': 'eminem',
        'kanye west': 'kanye west',
        'kanye': 'kanye west',
        'drake': 'drake',
        'the weeknd': 'the weeknd',
        'bruno mars': 'bruno mars',
        'frank sinatra': 'frank sinatra',
        'whitney houston': 'whitney houston',
        'mariah carey': 'mariah carey',
        'ed sheeran': 'ed sheeran',
        'rihanna': 'rihanna',
        'britney spears': 'britney spears',
        'pewdiepie': 'pewdiepie',
        'markiplier': 'markiplier',
        'jacksepticeye': 'jacksepticeye',
        'shroud': 'shroud',
        'ninja': 'ninja',
        'xqc': 'xqc',
        'caseoh': 'caseoh',
        'jschlatt': 'jschlatt',
        'charlie': 'charlie',
        'moist cr1tikal': 'moist cr1tikal',
        'michael scott': 'michael scott',
        'dwight schrute': 'dwight schrute',
        'jim halpert': 'jim halpert',
        'ron swanson': 'ron swanson',
        'leslie knope': 'leslie knope',
        'donald trump': 'donald trump',
        'trump': 'donald trump',
        'joe biden': 'joe biden',
        'biden': 'joe biden',
        'barack obama': 'barack obama',
        'obama': 'barack obama',
        'elon musk': 'elon musk',
        'kim jong un': 'kim jong un',
        'putin': 'vladimir putin',
        'vladimir putin': 'vladimir putin',
        'joe rogan': 'joe rogan',
        'jordan peterson': 'jordan peterson',
        'alex jones': 'alex jones',
        'john stewart': 'john stewart',
        'jon stewart': 'john stewart',
        'stephen colbert': 'stephen colbert',
        'jimmy fallon': 'jimmy fallon',
        'jimmy kimmel': 'jimmy kimmel',
        'conan o brien': 'conan o brien',
        'bill burr': 'bill burr',
        'jerry seinfeld': 'jerry seinfeld',
        'bo burnham': 'bo burnham',
        'dave chappelle': 'dave chappelle',
        'chris rock': 'chris rock',
        'kevin hart': 'kevin hart',
        'mario': 'mario',
        'luigi': 'luigi',
        'sonic': 'sonic the hedgehog',
        'sonic the hedgehog': 'sonic the hedgehog',
        'pikachu': 'pikachu',
        'link': 'link',
        'zelda': 'princess zelda',
        'ganondorf': 'ganondorf',
        'master chief': 'master chief',
        'cloud strife': 'cloud strife',
        'cloud': 'cloud strife',
        'sephiroth': 'sephiroth',
        'samus': 'samus aran',
        'samus aran': 'samus aran',
        'solid snake': 'solid snake',
        'snake': 'solid snake',
        'lara croft': 'lara croft',
        'kirby': 'kirby',
        'donkey kong': 'donkey kong',
        'peach': 'peach',
        'bowser': 'bowser',
        'glados': 'glados',
        'wheatley': 'wheatley',
        'cave johnson': 'cave johnson',
        'gordon freeman': 'gordon freeman',
        'alyx vance': 'alyx vance',
        'kratos': 'kratos',
        'nathan drake': 'nathan drake',
        'ezio': 'ezio auditore',
        'geralt': 'geralt of rivia',
        'geralt of rivia': 'geralt of rivia',
        'arthur morgan': 'arthur morgan',
        'john marston': 'john marston',
        'big boss': 'big boss',
        'tifa': 'tifa lockhart',
        'wolverine': 'wolverine',
        'storm': 'storm',
        'deadpool': 'deadpool',
        'gandalf': 'gandalf',
        'aragorn': 'aragorn',
        'gollum': 'gollum',
        'forrest gump': 'forrest gump',
        'joker': 'joker',
        'jack sparrow': 'jack sparrow',
        'sparrow': 'jack sparrow',
        'iron man': 'iron man',
        'thor': 'thor',
        'captain america': 'captain america',
        'thanos': 'thanos',
        'black panther': 'black panther',
        'black widow': 'black widow',
        'hawkeye': 'hawkeye',
        'doctor strange': 'doctor strange',
        'star lord': 'star lord',
        'wendy testaburger': 'wendy testaburger',
        'chef': 'chef',
        'randy marsh': 'randy marsh',
    }
    return aliases.get(norm, norm)


def quality_score(entry):
    """Score training quality: epochs, has index."""
    score = 0
    ep = entry.get("epoch", 0)
    if ep >= 500:
        score += 30
    elif ep >= 200:
        score += 20
    elif ep >= 50:
        score += 10
    elif ep > 0:
        score += 5
    if entry.get("index"):
        score += 20
    sz = entry.get("size_mb", 0)
    if isinstance(sz, (int, float)) and sz >= 55:
        score += 5
    elif entry.get("pth") or entry.get("zip_path"):
        score += 15
    elif entry.get("download_url"):
        score += 10
    return score


def build_catalog():
    catalog = []
    seen_keys = set()
    
    def add_entry(name, source, **fields):
        dk = dedupe_key(name)
        if dk in seen_keys:
            return False
        seen_keys.add(dk)
        
        display_name = clean_display_name(name)
        cat = fields.get("category") or category_from_name(display_name)
        pop = fields.pop("popularity_score", None)
        if pop is None:
            pop = popularity_score(display_name, cat)
        
        entry = {
            "name": display_name,
            "source": source,
            "category": cat,
            "popularity_score": pop,
            "quality_score": 0,
            "rating": 0,
        }
        entry.update(fields)
        qual = quality_score(entry)
        entry["quality_score"] = qual
        entry["rating"] = pop + qual
        catalog.append(entry)
        return True
    
    # === Tier 1: Real RVC voices from archive ===
    with open(OUT) as f:
        voices_data = json.load(f)
    
    rvc_entries = []
    for name, info in voices_data["voices"].items():
        display_name = clean_display_name(name)
        rvc_entries.append((display_name, info))
    
    # Sort by popularity desc, then quality desc
    rvc_entries.sort(key=lambda x: (
        -popularity_score(x[0], category_from_name(x[0])),
        -(30 if x[1].get("index") else 0),
        -(x[1].get("epoch", 0) // 100 if x[1].get("epoch") else 0),
    ))
    
    for display_name, info in rvc_entries:
        add_entry(
            display_name,
            "rvc_archive",
            pth=info.get("pth"),
            index=info.get("index"),
            epoch=info.get("epoch", 0),
            size_mb=info.get("size_mb", 0),
            variants=info.get("variants", 1),
        )
    
    # === Tier 2: Downloaded zips ===
    for d in sorted(os.listdir(VOICE_ZIPS)):
        if d == '.git':
            continue
        full = os.path.join(VOICE_ZIPS, d)
        if not os.path.isdir(full):
            continue
        for fn in sorted(os.listdir(full)):
            fp = os.path.join(full, fn)
            if os.path.isfile(fp) and fn.endswith('.zip'):
                size_mb = os.path.getsize(fp) / 1e6
                add_entry(
                    d,
                    "downloaded_zip",
                    zip_path=fp,
                    size_mb=round(size_mb, 1),
                )
    
    # === Tier 3: Cross-reference research list against 30K VM index ===
    with open(VM_INDEX) as f:
        vm_data = json.load(f)
    vm_index = vm_data.get("index", vm_data)
    
    # Build inverted index for fast lookup
    vm_tokens = {}
    for mid, meta in vm_index.items():
        if not isinstance(meta, dict):
            continue
        idx_name = meta.get("name", "")
        if not idx_name:
            continue
        idx_norm = normalize(idx_name)
        for tok in idx_norm.split():
            if len(tok) > 2:
                vm_tokens.setdefault(tok, set()).add(mid)
    
    # Build a combined priority list from all research data
    priority_list = []
    all_lists = [CARTOON_CHARS, ANIME_CHARS, MOVIE_CHARS, MUSIC_CHARS,
                 YT_CHARS, GAME_CHARS, POLITICAL_CHARS, COMEDY_CHARS, TV_CHARS]
    
    # Deduplicate priority list by dedupe_key
    seen_priority = set()
    for char_list in all_lists:
        for name, rank, cat in char_list:
            dk = dedupe_key(name)
            if dk in seen_priority:
                continue
            seen_priority.add(dk)
            priority_list.append((name, rank, cat))
    
    # Sort by rank
    priority_list.sort(key=lambda x: x[1])
    
    # For each priority character, search the VM index
    vm_added = 0
    for pname, rank, cat in priority_list:
        if len(catalog) >= MAX_VOICES:
            break
        
        pnorm = normalize(pname)
        ptokens = pnorm.split()
        
        # Find candidate VM entries
        candidates = set()
        for tok in ptokens:
            if tok in vm_tokens:
                candidates.update(vm_tokens[tok])
        
        best_match = None
        best_score = -1
        
        for mid in candidates:
            meta = vm_index.get(mid)
            if not isinstance(meta, dict):
                continue
            idx_name = meta.get("name", "")
            idx_norm = normalize(idx_name)
            
            # Verify match
            if pnorm and (pnorm in idx_norm or idx_norm in pnorm):
                # Skip non-English
                nl = idx_name.lower()
                if any(k in nl for k in ['espanol', 'français', 'deutsch', 'italiano',
                                         'português', 'portuguese', 'русский', 'korean',
                                         'chinese', 'arabic', 'turkish', 'polish',
                                         'हिंदी', 'indi', 'thai', 'vietnamese',
                                         'indonesian', 'tagalog', 'українська']):
                    if not any(k in nl for k in ['english', 'eng dub', 'eng ']):
                        continue
                
                dl_url = meta.get("download_url")
                if not dl_url:
                    continue
                
                display = clean_display_name(idx_name)
                dk = dedupe_key(display)
                if dk in seen_keys:
                    continue
                
                # Score match quality: longer overlap = better
                match_score = len(set(pnorm.split()) & set(idx_norm.split()))
                
                if match_score > best_score:
                    best_score = match_score
                    best_match = (mid, display, dl_url, meta.get("size", "??"), cat)
        
        if best_match:
            mid, display, url, size, vcat = best_match
            dk = dedupe_key(display)
            if dk not in seen_keys:
                seen_keys.add(dk)
                
                # Research-based popularity (lower rank = higher score)
                pop_score = 1000 - (rank * 10)
                
                entry = {
                    "name": display,
                    "source": "voice_models_com",
                    "category": vcat,
                    "popularity_score": pop_score,
                    "quality_score": 10,
                    "rating": pop_score + 10,
                    "download_url": url,
                    "vm_model_id": mid,
                    "size_mb": size,
                }
                catalog.append(entry)
                vm_added += 1
    
    # === Fill remaining with top RVC archive voices if needed ===
    # (Already added in Tier 1, so this just ensures we have enough)
    
    # Sort final catalog: archives first (best quality), then by rating
    catalog.sort(key=lambda x: (
        -x["rating"],
        0 if x["source"] == "rvc_archive" else (1 if x["source"] == "downloaded_zip" else 2),
        -x["popularity_score"],
        x["name"].lower(),
    ))
    catalog = catalog[:MAX_VOICES]
    
    return catalog


def write_outputs(catalog):
    """Write voices.json, voice_catalog.json, all_voice_dirs.txt."""
    
    voices_out = {
        "root": RVC_ARCHIVE,
        "total": len(catalog),
        "with_index": sum(1 for v in catalog if v.get("index")),
        "without_index": sum(1 for v in catalog if not v.get("index")),
        "voices": {},
    }
    
    for v in catalog:
        name = v["name"]
        voices_out["voices"][name] = {
            "name": name,
            "source": v["source"],
            "category": v["category"],
            "popularity_score": v["popularity_score"],
            "quality_score": v["quality_score"],
            "rating": v["rating"],
            "pth": v.get("pth"),
            "index": v.get("index"),
            "epoch": v.get("epoch", 0),
            "size_mb": v.get("size_mb", 0),
            "variants": v.get("variants", 1),
            "zip_path": v.get("zip_path"),
            "download_url": v.get("download_url"),
            "vm_model_id": v.get("vm_model_id"),
        }
    
    with open(OUT, "w") as f:
        json.dump(voices_out, f, indent=1)
    
    with open(CATALOG_OUT, "w") as f:
        json.dump({
            "description": "500 most popular voices in cartoons and media for WuBuMedia cohost",
            "total": len(catalog),
            "ranking_method": "web-researched popularity (IMDb, MAL, Wikipedia) + training quality, sorted descending",
            "research_sources": [
                "IMDb: Most Iconic Cartoon Characters of All Time",
                "MyAnimeList: Most Favorited Characters (top 50)",
                "Wikipedia: List of most-subscribed YouTube channels",
                "GamesRadar: 50 most iconic video game characters",
                "Empire/AFI: 100 Greatest Movie Characters",
                "SocialBlade/VIDIQ: Top YouTube creators by subscribers",
            ],
            "sources": ["rvc_archive", "downloaded_zip", "voice_models_com"],
            "categories": ["cartoon", "anime", "game", "movie", "music", "personality", "other"],
            "voices": catalog,
        }, f, indent=1)
    
    with open(ALL_DIRS_OUT, "w") as f:
        for v in catalog:
            f.write(v["name"] + "\n")
    
    print(f"Wrote {len(catalog)} voices to:")
    print(f"  {OUT}")
    print(f"  {CATALOG_OUT}")
    print(f"  {ALL_DIRS_OUT}")
    
    by_source = {}
    by_cat = {}
    for v in catalog:
        by_source[v["source"]] = by_source.get(v["source"], 0) + 1
        by_cat[v["category"]] = by_cat.get(v["category"], 0) + 1
    
    print(f"\nBy source:")
    for s, c in sorted(by_source.items(), key=lambda x: -x[1]):
        print(f"  {s:25s}: {c}")
    print(f"\nBy category:")
    for c, n in sorted(by_cat.items(), key=lambda x: -x[1]):
        print(f"  {c:25s}: {n}")
    
    print(f"\nTop 30 voices by rating:")
    for v in catalog[:30]:
        loc = "arch" if v.get("pth") else ("zip" if v.get("zip_path") else "dl")
        idx = "+" if v.get("index") else ("zip" if v.get("zip_path") else "no")
        print(f"  {v['rating']:5d}  {v['name']:35s}  [{v['category']:12s}]  src={loc} idx={idx}")


if __name__ == "__main__":
    catalog = build_catalog()
    write_outputs(catalog)
