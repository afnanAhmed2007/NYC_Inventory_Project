# NYC Infrastructure Intelligence

A CS 210 data-management project for DOT borough desks. It loads street-maintenance 311 requests, community-district population, borough pavement ratings, and Central Park weather into SQLite, trains two models, and opens a Streamlit desk view.

The screen answers three questions for a borough, community district, complaint group, and week:

- Where are requests likely to be opened?
- Which districts already have requests past due?
- Which factors traveled with the forecast: week of year, last year's count, freeze-thaw days, rain, snow, fair or poor pavement, and population?

## Run it

`pdftotext` (poppler) is required for the census profile.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python pipeline.py
python -m streamlit run app.py --server.port 8765 --server.address 0.0.0.0
```

The first pipeline run downloads about 1.06 million 311 requests and the pavement file. Later runs reuse the files in `data/raw/`. The database is `data/nyc_infrastructure.sqlite`. Bulk files are not committed.

Open the app at `http://127.0.0.1:8765`.

## What is stored

`schema.sql` creates the tables. `queries.sql` lists the management queries the dashboard runs. Labels and the district-week table are SQL updates and inserts, not a second copy of the numbers in the app.

A request is past due when it is still open after its due date, or after 14 days if the due date is missing. A request that closed after that deadline counts as past due too. A request that is still inside the deadline is left unlabeled. Came back means the same complaint group returned to the same cross streets within 90 days of an earlier close.

Weeks with no requests are stored as zeros. Borough totals on the screen are the sum of the district rows.

Pavement is a length-weighted borough share from Street Pavement Ratings. Every district in a borough uses that borough figure, and the screen says so. Good is 8 to 10, fair is 4 up to 8, and poor is below 4.

Population is the 2020 Census total for each community district, parsed from the Department of City Planning profile in `reference/nyc-cd-decennial-2020-2010.pdf` (September 2021).

## Models

The last 12 complete weeks are the holdout. Nothing is split at random.

The request forecast compares three numbers on that holdout: the same district and complaint group 52 weeks earlier, a Poisson regression, and a histogram gradient boosting model with Poisson loss. The stored forecast is the one with the lowest mean absolute error. If neither model beats the seasonal count, the seasonal count is what the screen shows. The first week after the data ends uses that model with last year's weather. The next three weeks are the seasonal count.

The past-due score compares each complaint group's historical rate, a logistic regression, and a gradient boosting classifier. The stored score is the one with the higher average precision. The classifier is fit on a stratified sample of 250,000 labeled requests from before the holdout, and it is scored on every labeled request in the holdout.

## Limits

311 is who called, not every defect. One weather station stands in for the city. Pavement is the latest survey, not that morning's street. A high score is not a reason to ignore a new call.
