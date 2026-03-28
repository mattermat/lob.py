# Support to ITCH
### Expected goal
Make `lobpy` able to support loading of Nasdaq's `ITCH` data.
To do so, we need to create a parser for `ITCH` messages and the ingestion of information into a `TL`.

### Test (in the github workflow)
With the command: `wget <uri> -P test_data/`

Download all the following:
- `https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/01302019.NASDAQ_ITCH50.gz`
- `https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/01302020.NASDAQ_ITCH50.gz`
- `https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/03272019.NASDAQ_ITCH50.gz`
- `https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/07302019.NASDAQ_ITCH50.gz`
- `https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/08302019.NASDAQ_ITCH50.gz`
- `https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/10302019.NASDAQ_ITCH50.gz`
- `https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/12302019.NASDAQ_ITCH50.gz`
- `https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/S010303-v2.zip`
- `https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/S071321-v50.txt.gz`
- `https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/S081321-v50.txt.gz`
- `https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/S101819-v50.txt.gz`

### Script (on local machine)

Run the script: `.env/bin/python.py examples/itch.py test_data/01302019.NASDAQ_ITCH50.gz`