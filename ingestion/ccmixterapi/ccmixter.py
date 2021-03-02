"""
The CCMixter API is documented at http://ccmixter.org/query-api
Here, we save the results as jsonl
for further processing.
"""
import os
import time
from datetime import date
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import bs4 as bs
import jsonlines

today = date.today().strftime("%Y-%b-%d")

class CCMixterRequester:
    def __init__(self, licenses=["sa"], limit=20, max_offset=150) -> None:
        self.filename = f'ccmixter-sa-{today}.jsonl'
        if os.path.exists(self.filename):
            raise ValueError("Attempting to write over existing output!")        
        retry_strategy = Retry(
            total=5, status_forcelist=[429, 500, 502, 503, 504], backoff_factor=5
        )
        self.adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session = requests.Session()
        self.session.mount("http://ccmixter.org/", self.adapter)
        self.licenses = licenses
        self.limit = limit
        self.max_offset = max_offset
        self.params = {
            "f": "rss",
            "limit": self.limit,
            "offset": 0,
            "dataview": "uploads",
            "sort": "rank",
        }

    def start_requests(self):
        for lic in self.licenses:
            self.params["lic"] = lic
            for offset in range(0, self.max_offset, self.limit):
                self.params["offset"] = offset
                response = self.session.get(
                    "http://ccmixter.org/api/query?", params=self.params, timeout=5
                )
                if response.status_code == 200:
                    xml = bs.BeautifulSoup(response.content, 'xml')
                    
                    contents = []
                    
                    for item in xml.findAll('item'):
                        try:
                            item_stuff = {}
                            item_stuff['title'] = item.findAll('title')[0].contents[0]
                            item_stuff['artist'] = item.findAll('dc:creator')[0].contents[0]
                            item_stuff['media_url'] = item.findAll('enclosure')[0]['url']
                            item_stuff['link'] = item.findAll('link')[0].contents[0]
                            item_stuff['tags'] = [c.contents[0] for c in item.findAll('category')]
                            item_stuff['license'] = item.findAll('cc:license')[0].contents[0]
                            contents.append(item_stuff)
                        except:
                            pass
                    self.write_out(contents)
                    print("Wrote out", lic, offset)
                time.sleep(1)
                    
    def write_out(self, contents):
        with jsonlines.open(self.filename, mode='a') as writer:
            writer.write_all(contents)
            
if __name__ == "__main__":
    crawler = CCMixterRequester()
    crawler.start_requests()