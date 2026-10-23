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

**[TODO — not built yet.]** Planned: embed `movie_description` per title (one
chunk per title — descriptions are short, so no splitting).
Known issues to decide on:
- Each description contains everything twice (no-space + readable version).
- Country and title aren't in the text, so "a comedy from India" relies on the
  description. Could add `Country:` or filter by country before retrieval.
- Model: **[TODO]**. Vector store: **[TODO]**.

## 4. What would break at 100,000 titles

- **Hand cleaning.** `netflix_movies.csv` was made outside `ingest.py`. At 100k
  rows (and with re-deliveries) the cleaning has to be code that reruns.
- **Embedding at startup.** Embedding 100k texts on CPU takes many minutes;
  embeddings must be computed once in a separate step and saved to disk.
- **Vector search.** numpy brute force over 100k × 384 floats (~150 MB) still
  works but is slow per query; I'd move to FAISS.
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

Where I overrode it:
- It recommended `ingest.py` load `netflix_titles.csv` and do all cleaning
  there. I chose to load my own `netflix_movies.csv`.
- It suggested `release_year`; I used `year` (the README overview says "year").
- It suggested keeping all 12 columns; I kept only the 7 I need.

Accepted without fully understanding: **[TODO — be honest]**

## 6. With another 4 hours

**[TODO]** Ideas: move my cleaning into code so it reruns from the original
CSV; reconsider dropping the 53 rows; separate cast/genre tables to filter by
actor or genre; evaluate retrieval on a few hand-written questions.
