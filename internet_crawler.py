import requests
import json
import os
from datetime import datetime
import time

CRAWL_DIR = "data/internet_data"
os.makedirs(CRAWL_DIR, exist_ok=True)

def crawl_programming_data():
    """Crawl programming data from public APIs."""
    sources = [
        {
            "name": "GitHub Trending",
            "url": "https://api.github.com/search/repositories?q=stars:>50000&sort=stars&order=desc&per_page=10",
            "type": "code"
        },
        {
            "name": "Stack Overflow Tags",
            "url": "https://api.stackexchange.com/2.3/tags?site=stackoverflow&order=desc&sort=popular&pagesize=10",
            "type": "qa"
        },
        {
            "name": "Dev.to Articles",
            "url": "https://dev.to/api/articles?top=7",
            "type": "article"
        }
    ]
    
    all_data = []
    
    for source in sources:
        try:
            print(f"[*] Crawling {source['name']}...")
            resp = requests.get(source['url'], timeout=5)
            
            if resp.status_code == 200:
                data = resp.json()
                
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "source": source['name'],
                    "type": source['type'],
                    "data": data
                }
                all_data.append(entry)
                print(f"[+] {source['name']}: OK")
            else:
                print(f"[!] {source['name']}: Failed ({resp.status_code})")
        
        except Exception as e:
            print(f"[!] {source['name']}: Error - {e}")
        
        time.sleep(1)  # Rate limiting
    
    # Save crawled data
    crawl_file = os.path.join(CRAWL_DIR, f"crawl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
    with open(crawl_file, "w", encoding="utf-8") as f:
        for entry in all_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"\n[+] Crawled {len(all_data)} sources -> {crawl_file}")
    return all_data

if __name__ == "__main__":
    crawl_programming_data()