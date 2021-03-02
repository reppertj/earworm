import os
import json
from datetime import date
import jsonlines

import scrapy
from scrapy_splash import SplashRequest

today = date.today().strftime("%Y-%b-%d")

class FMAMetadataSpider(scrapy.Spider):
    name = "fma_metadata"

    def start_requests(self):
        self.filename = f'fma_metadata-{today}.jsonl'
        if os.path.exists(self.filename):
            raise ValueError("Attempting to write over existing output!")
        urls = (
            f'https://freemusicarchive.org/music/charts/all?page={p}' for p in range(1, 6243)
        )
        for url in urls:
            yield SplashRequest(url=url, callback=self.parse, args={'wait': 0.3})

    def parse(self, response):
        all_track_info = []
        for item in response.css(".play-item"):
            all_track_info.append(json.loads(item.attrib['data-track-info']))
        with jsonlines.open(self.filename, mode='a') as writer:
            writer.write_all(all_track_info)
        self.log(f'Scraped {response.url}')