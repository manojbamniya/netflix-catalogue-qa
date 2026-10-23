# NOTES

> DRAFT — sections marked **[TODO]** need to be filled in or checked by me.

## 1. Problems found in the CSV, and what I did

`data/netflix_titles.csv`: 6,234 rows, 12 columns. I cleaned it into
`data/netflix_movies.csv` (6,180 rows, 7 columns), which `ingest.py` loads.
**[TODO: say how — script/notebook/by hand — and include it in the repo.]**

| Problem | Count | What I did |
|---|---|---|
| Same title listed twice under two `show_id`s, every other column identical (*Sarkar*, 81075235 / 81072516) | 1 | Dropped 81072516. A plain duplicate check finds 0, because `show_id` differs — you have to compare all columns except `show_id`. |
| Same name, different works (*Tiger* 2016: India vs Argentina; *Kakegurui* 2019: anime vs live action) | 2 pairs | Kept both. Deduping by title would wrongly merge them. 60 titles appear more than once overall, all legitimate. |
| Missing `rating` | 10 | Dropped. **[TODO: why]** |
| Missing director, cast **and** country | 42 | Dropped. **[TODO: why — and is it worth losing them? e.g. the 1925 *Pioneers: First Women Filmmakers* is one of them]** |
| *Kikoriki* (80159925): no director, country or date added, but has cast | 1 | Dropped. **[TODO: check — doesn't fit the rule above]** |
| Missing `country` (in the rows I kept) | 429 | Kept, stored as NULL. See section 2. |
| Missing `director` / `cast` | 1,969 / 570 | Kept; normal for TV shows, docs and stand-up. |
| Country with a trailing comma (`"Poland,"`, `"United States,"`, `"United Kingdom,"`) | 3 | Removed the comma. Otherwise splitting creates an empty country. |
| `date_added` with a leading space (`" June 13, 2019"`) | 640 | Not relevant — I dropped `date_added`. |
| `release_year` later than the year in `date_added` | 9 | Left as is; can't tell which date is wrong. |
| Descriptions with double spaces or line breaks | 10 | Collapsed to single spaces. |
| `duration` mixes units: `"90 min"` for movies, `"2 Seasons"` for TV | all | Kept as text inside `movie_description`. |

Other changes in `netflix_movies.csv`: renamed `release_year` → `year`; dropped
`date_added`; merged `listed_in`, `duration`, `cast`, `director` into
`movie_description` (spaces removed, most commas in the description removed).
**[TODO: why remove spaces and commas]**

## 2. Schema decisions

SQLite, two tables (built by `ingest.py`):

- **`titles`** — `show_id` (PK), `type`, `title`, `country`, `year`, `rating`,
  `movie_description`. Indexes on `type`, `year`, `rating`.
- **`title_countries`** — one row per (title, country), so
  `"United States, India"` becomes two rows. Used for `?country=` and the
  top-10 stats. Exact match on an indexed column, instead of `LIKE '%India%'`
  on a comma string.

**Missing country → NULL, not `"Unknown"` or `""`.** A placeholder looks like a
real country, so it shows up in the top 10 and every query has to remember to
exclude it. With NULL, those titles simply have no row in `title_countries`, so
they can't appear in the top 10. They're still returned by unfiltered
`/titles` and by `/ask`. The API accepts `?country=Unknown` and maps it to
"titles with no country" (429), so they're still filterable.

**Cast, director, genres** are not separate tables — they live only inside
`movie_description`. Trade-off: simpler schema, but you can't filter by
actor or genre. None of the required endpoints need that.

**Readable names.** In `netflix_movies.csv` names have no spaces
(`AlanMarriott`), which hurts search ("Alan Marriott" won't match). `ingest.py`
reads genres, duration, cast and director from the original CSV by `show_id`
and appends `Genres: … Duration: … Cast: … Director: …` to
`movie_description`.

## 3. RAG choices

**[TODO: check and reword]**

- **What I embed:** one text per title, no chunking (titles are short):
  `"<title> (<type>, <year>). Country: <country>. Rating: <rating>. <movie_description>"`.
  Title, country and year go first because the model only reads the first 256
  word pieces — 154 of 6,180 texts are longer and lose the end of the cast list.
- **Embedding model:** `all-MiniLM-L6-v2` (sentence-transformers). Small
  (~90 MB), free, runs locally; 6,180 titles take ~13 s on my laptop.
- **Vector store:** none — numpy. 6,180 × 384 floats is ~9 MB; a full dot
  product per question takes milliseconds. Vectors are normalised, so the dot
  product is cosine similarity. Built once by `python -m rag.index`, saved to
  `data/embeddings.npz`.
- **Retrieval:** top 8 titles (`TOP_K`), passed to Gemini
  (`gemini-3.5-flash-lite`, temperature 0.2) with a prompt that says to use
  only those titles, each labelled with its `show_id`.
- **Grounding:** Gemini must reply in JSON (`answer`, `show_ids`). The code
  then drops any `show_id` that wasn't among the 8 retrieved — so a made-up
  title can never be returned as a source. If no valid source is left, the
  answer is replaced with a fixed "couldn't find a title" message. Tested with
  fake LLMs that return made-up ids.
- **What I saw when testing retrieval:** good for themes ("anime about gambling
  at school" → *Kakegurui*; "Transformers" → all 5 Transformers shows). Weak
  for names: "movies with Shah Rukh Khan" → *Rukh* first, because embeddings
  match meaning, not exact words. Hybrid search (add BM25 keyword scores) would
  fix it. *City of Joy* didn't make the top 5 for "women healing after war in
  Congo" — maybe the no-space duplicate text in `movie_description` adds noise.
- **With Gemini:** "comedy from India with a strong female lead" → *Aiyyaa*;
  "anime about gambling at school" → *Kakegurui*; "time travel on Mars with
  Tom Cruise" (not in the catalogue) → the no-match message, no sources.

## 4. What would break at 100,000 titles

- **Hand cleaning.** `netflix_movies.csv` was made outside `ingest.py`. At 100k
  rows (and with re-deliveries) the cleaning has to be code that reruns.
- **Building the index.** 6,180 titles take ~13 s, so 100k take ~4 min — fine
  as a separate step, but every rebuild re-embeds everything. I'd embed only
  new or changed titles.
- **Vector search.** numpy over 100k × 384 floats is ~150 MB in memory and
  ~16× slower per question; I'd move to FAISS.
- **`/ask` loads the whole index into each server process** — with several
  workers that's 150 MB each.
- **Deep pagination.** `LIMIT/OFFSET` scans all skipped rows; page 2,000 gets
  slow. Keyset pagination (`WHERE show_id > ?`) fixes it.
- **`/stats` per request.** Counting by country on every call — cache it or
  compute at ingest time.
- **Exact-match dedup** misses near-duplicates (a typo, a different cast
  order), which become more common at scale.

## 5. AI usage

**[TODO — check and rewrite in my own words.]** I used Claude Code to:
- Profile both CSVs and compare them (found the *Sarkar* duplicate, the
  *Tiger* / *Kakegurui* same-name pairs, the trailing commas).
- Write `ingest.py`, plus the pandas snippets I used to check the data.
- Write the FastAPI app (`api/`), the RAG code (`rag/`) and the tests.
- Debug a scipy crash on macOS 27 (fixed by using Python 3.13).

Where I overrode it:
- It recommended `ingest.py` load `netflix_titles.csv` and do all cleaning
  there. I chose to load my own `netflix_movies.csv`.
- It suggested `release_year`; I used `year` (the README overview says "year").
- It suggested keeping all 12 columns; I kept only the 7 I need.
- It suggested Groq for the LLM; I picked Gemini.

Accepted without fully understanding: **[TODO — be honest]**

## 6. With another 4 hours

**[TODO]** Ideas: move my cleaning into code so it reruns from the original
CSV; reconsider dropping the 53 rows; separate cast/genre tables to filter by
actor or genre; evaluate retrieval on a few hand-written questions.
