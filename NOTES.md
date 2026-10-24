# NOTES

> DRAFT — reword in my own words; **[TODO]** = still to write.

## 1. Problems in the CSV, and what I did

`netflix_titles.csv` has 6,234 rows. I cleaned it in a Kaggle notebook into
`netflix_movies.csv` (6,180 rows, 7 columns), which `ingest.py` loads.
The Kaggle notebook I used for the cleaning is `netflix-movies.ipynb`.

| Problem | Rows | What I did |
|---|---|---|
| *Sarkar* listed twice (81075235, 81072516), identical except `show_id` | 1 | Dropped one. A normal duplicate check finds nothing — you must ignore `show_id`. |
| Same name, different titles (*Tiger* 2016: India vs Argentina; *Kakegurui*: anime vs live action) | 55 names | Kept. Deduping by title would merge different shows. |
| No `rating` | 10 | Dropped. It's a very small number, so it won't affect the suggestions. Rating is also one of the filters, and I don't want a filter column to be empty. |
| 3 or more empty fields (e.g. no director, cast and country) | 43 | Dropped. With 3 or more columns empty, there's too little left to be useful for suggestions. Includes *Kikoriki* (no director, country, date added) and the 1925 *Pioneers: First Women Filmmakers* — losing those is the cost of the rule. |
| No `country` (kept rows) | 429 | Kept, as NULL (section 2). They're about 7% of the data, so dropping them would lose too much. I didn't fill in a placeholder like an empty string either, because country is a filter and a fake value would cause problems there. |
| No director / no cast | 1,969 / 570 | Kept: normal for TV, docs, stand-up. I replaced them with an empty string and combined them into `movie_description`, so the rest of the row isn't lost. |
| Trailing comma in country (`"Poland,"`) | 3 | Removed; otherwise splitting makes an empty country. |
| `release_year` after `date_added` | 9 | Left: I can't tell which is wrong. |
| `date_added` with a leading space | 640 | Not used anywhere. |

I also renamed `release_year` → `year` and merged genres, duration, cast and
director into one `movie_description`:
`"<description> Genres: … Duration: … Cast: … Director: …"`. My first version
removed the spaces inside names (`AlanMarriott`), which broke search ("Alan
Marriott" didn't match), so I redid it with spaces. **[TODO: the notebook still
removes commas, so names run together — "Alan Marriott Andrew Toth"; keep
commas inside `movie_description`]**

`ingest.py` loads that file, removes the 3 trailing commas, and compares it
with the original CSV to print what was dropped and why.

## 2. Schema decisions

- **`titles`**: `show_id` (key), `type`, `title`, `country`, `year`, `rating`,
  `movie_description`; indexes on `type`, `year`, `rating`.
- **`title_countries`**: one row per title and country, so
  `"United States, India"` is two rows. The country filter and top 10 use it:
  an exact match, so "India" doesn't match "British Indian Ocean Territory".
- **Missing country is NULL, not `"Unknown"` or `""`.** A placeholder looks like
  a real country and would show up in the top 10. `?country=Unknown` still
  finds those 429 titles: the API maps it to "no row in `title_countries`".
- **Cast, director, genres** live only in `movie_description`. Simpler, but
  you can't filter by actor or genre — no endpoint needs that.
- The API takes `?release_year=` (the README's name) and filters the `year`
  column. Bad input returns 400 naming the parameter.

## 3. RAG choices

- **Text per title** (no chunking — descriptions are short):
  `"<title> (<type>, <year>). Country: <country>. Rating: <rating>. <movie_description>"`.
  Title and country first, because the model reads only 256 word pieces
  (only 2 texts are longer).
- **Model:** `all-MiniLM-L6-v2` — free, local, ~90 MB; all titles in ~13 s.
- **Store:** plain numpy. 6,180 × 384 numbers is ~9 MB, searched in
  milliseconds. Built once by `python -m rag.index`.
- **Answer:** the top 15 titles go to Gemini (`gemini-3.5-flash-lite`), each
  with its `show_id`, with a prompt to use only those (partial matches are
  allowed if it says what doesn't fit). Gemini replies in JSON (`answer`,
  `show_ids`); the code drops any `show_id` it wasn't given, and if none are
  left returns "I couldn't find a title…" with no sources.
- **Why 15, not 8:** with 8, *City of Joy* (rank 11 for "women healing after
  war in Congo") and every Shah Rukh Khan film (first at rank 15) were never
  shown to Gemini. Embeddings match meaning, not exact names.
- **Results:** "comedy from India with a strong female lead" → *Mumbai Delhi
  Mumbai*; "anime about gambling at school" → *Kakegurui*; "Congo" →
  *City of Joy*; "Shah Rukh Khan" → *Ram Jaane*; "time travel on Mars with Tom
  Cruise" → says nothing fits all three, suggests related titles.
- **Gemini limits:** the free key allows only ~20 requests a day per model on
  `gemini-3.6-flash` (I used it up while testing), so I use the lite model.
  Busy errors (5xx) are retried 3 times; "quota used up" returns a clear 503.

## 4. What breaks at 100,000 titles

- **Cleaning in a notebook** doesn't rerun automatically; at this size it
  must be code in the pipeline.
- **Index build** grows to ~4 min and re-embeds everything; embed only new
  titles.
- **numpy search** needs ~150 MB per server process and gets ~16× slower;
  use FAISS.
- **`LIMIT/OFFSET` pagination** slows on deep pages; use
  `WHERE show_id > ?`.
- **`/stats`** recounts on every call; cache it.
- **Exact-match dedup** misses near-duplicates, which get more common.

## 5. AI usage

I used Claude Code to profile and compare the CSVs (it found the *Sarkar*
duplicate, the same-name pairs), and to write `ingest.py`, the API,
the RAG code and the tests. It also debugged a scipy crash on macOS 27 (fixed
by using Python 3.13).

Where I overrode it: it wanted `ingest.py` to clean `netflix_titles.csv`
itself — I load my own cleaned file. It suggested `release_year`, all 12
columns and Groq; I chose `year`, 7 columns and Gemini.

Accepted without fully understanding: the test file, and some of the code I
didn't understand well. For that code, I asked Claude to rewrite it more simply.

## 6. With another 4 hours

With another 4 hours, I would test more by myself and also build a UI for it.
