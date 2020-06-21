import requests
import pathlib
from bs4 import BeautifulSoup as bs

dir_transcript = pathlib.Path.cwd().parent.joinpath('transcripts')
dir_transcript.mkdir(parents=True, exist_ok=True)

for n in range(31373, 31600):
    url = 'https://transcripts.foreverdreaming.org/viewtopic.php?f=845&t=' + str(n) + '&view=print'
    r = requests.get(url)
    if r.status_code == 200:
        ep = bs(r.content, 'html.parser').select('.topic')[0].string.split(' ')[0]
        with open(dir_transcript.joinpath(ep + ".html"), 'wb') as f:
            f.write(r.content)
        print("Downloaded " + ep)
