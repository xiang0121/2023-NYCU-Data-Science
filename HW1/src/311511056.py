import sys

import requests
import json
from bs4 import BeautifulSoup
import re

import queue
import threading
import time
from tqdm import tqdm

class Crawler():
    def __init__(self) -> None:
        self.url_index  = 3500
        self.get_soup_by_index(self.url_index)
        
        
    def get_soup_by_index(self, url_index):
        url = f'https://www.ptt.cc/bbs/Beauty/index{url_index}.html'
        self.response = requests.get(url, cookies={'over18': '1'})
        self.soup = BeautifulSoup(self.response.text, 'html.parser')   
        return self.soup
    
    def get_soup_by_url(self, url):
        self.response = requests.get(url, cookies={'over18': '1'})
        self.soup = BeautifulSoup(self.response.text, 'html.parser') 
        return self.soup

    

    def get_all_articles(self):
        year = 2021
        url_idx = self.url_index
        previous_date = -1
        crawl_dict = {}
        divs = self.get_soup_by_index(url_idx).find_all('div', class_='r-ent')
        all_article_fp = open(f'./all_article.jsonl', 'w', encoding="utf8")
        all_popular_fp = open(f'./all_popular.jsonl', 'w', encoding="utf8")
        while (year < 2023):
            divs = self.get_soup_by_index(url_idx).find_all('div', class_='r-ent')
            pbar = tqdm(divs)
            for div in pbar:
                pbar.set_description(f'Processing page {url_idx}')
                # crawl date
                date = div.find('div', class_='date').text.split('/')
                crawl_dict['date'] = f'{int(date[0]):02d}'+ date[1]
                if (previous_date == '1231' and crawl_dict['date'] == '0101'):
                    year += 1
                previous_date = crawl_dict['date'] 
                
                #crwal title
                crawl_dict['title' ] = div.find('div', class_='title').text[1:-1]
                if '[公告]' in crawl_dict['title']:
                    continue
                
                # crawl url
                try :
                    href = div.find('a')['href']
                    crawl_dict['url'] = r'https://www.ptt.cc/' + href
                    author = div.find('div', class_='author').text
                except:
                    continue

                # output jsonl
                if year == 2022:
                    all_article_fp.write(json.dumps(crawl_dict, ensure_ascii=False))
                    all_article_fp.write('\n')
                    if div.find('span', class_='hl f1'):
                      all_popular_fp.write(json.dumps(crawl_dict, ensure_ascii=False))
                      all_popular_fp.write('\n')
            
            url_idx += 1 # next page

    

    def count_pushs(self, url, push_ids, boo_ids, lock):
        
        soup = self.get_soup_by_url(url)
        pushs = soup.find_all('span', class_='hl push-tag')
        boos = soup.find_all('span', class_='f1 hl push-tag')

        # count pushs and boos
        for push in pushs:
            if push.text == '推 ':
                lock.acquire()
                if push.next_sibling.text not in push_ids:
                    push_ids[push.next_sibling.text] = 1
                elif push.next_sibling.text in push_ids:
                    push_ids[push.next_sibling.text] += 1
                lock.release()

        for boo in boos:
            if boo.text == '噓 ':
                lock.acquire()
                if boo.next_sibling.text not in boo_ids:
                    boo_ids[boo.next_sibling.text] = 1
                elif boo.next_sibling.text in boo_ids:
                    boo_ids[boo.next_sibling.text] += 1
                lock.release()


    @classmethod    
    def get_push(cls, push_ids, boo_ids, output_dict):
           

        output_dict['all_like'] = sum(push_ids.values())
        output_dict['all_boo'] = sum(boo_ids.values())

        push_ids = sorted(push_ids.items(), key=lambda x: (-x[1], x[0]))[:10]
        boo_ids = sorted(boo_ids.items(), key=lambda x: (-x[1], x[0]))[:10]
        

        for i in range(10):
            user_dict = {}
            user_dict['user_id'] = push_ids[i][0]
            user_dict['count'] = push_ids[i][1]
            output_dict[f'like {i+1}'] = user_dict
        
        for i in range(10):
            user_dict = {}
            user_dict['user_id'] = boo_ids[i][0]
            user_dict['count'] = boo_ids[i][1]
            output_dict[f'boo {i+1}'] = user_dict

        # print(f'all-like: {push_count}')
        # print(f'all-boo: {boo_count}')
        # print(f'like-ids: {push_ids}')
        # print(f'boo-ids: {boo_ids}')
 
        

    def get_popular(self, url, image_urls, lock):
        
        self.soup = self.get_soup_by_url(url)
        contents = self.soup.select('a')
        for content in contents:
            lock.acquire()
            if content.get('href').endswith(('jpg' , 'png' , 'gif' , 'jpeg')) and content.get('href').startswith('http'):
                image_urls.append(content.text)
            lock.release()

    

    def get_keyword(self, url, keyword, image_urls, key_output_dict, lock):
        self.soup = self.get_soup_by_url(url)
        main_container = self.soup.find('div', id='main-container')
        ip = "※ 發信站: 批踢踢實業坊\(ptt\.cc\), 來自: [0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}"
        doc = re.split(ip, main_container.text)[0]
        lock.acquire()
        if (keyword in doc):
            
            contents = main_container.select('a')
            for content in contents:
                
                if content.get('href').endswith(('jpg' , 'png' , 'gif' , 'jpeg')) and content.get('href').startswith('http'):
                    image_urls.append(content.text)
                    
            key_output_dict['image_urls'] = image_urls
        lock.release()          

class Worker(threading.Thread):
    push_ids = {}
    boo_ids = {}
    output_dict = {}
    lock = threading.Lock()
    crawler = Crawler()
    bar = None
    image_urls = []
    popular_dict = {}
    keyword = None
    def __init__(self, queue, func):
        threading.Thread.__init__(self)
        self.queue = queue  # urls
        self.func = func

    def run(self):      
        while self.queue.qsize() > 0:         
            self.url = self.queue.get()
            self.bar.update(1)
            self.func_call()
    
    def func_call(self):
        if self.func == 'push':
            return self.push()
        elif self.func == 'popular':
            return self.popular()
        elif self.func == 'keyword':
            return self._keyword()
        else:
            print('function error')
            exit(1)    

    def push(self):
        self.crawler.count_pushs(self.url, self.push_ids, self.boo_ids, self.lock)
    
    def popular(self):
        self.crawler.get_popular(self.url, self.image_urls, self.lock)

    def _keyword(self):
        self.crawler.get_keyword(self.url, self.keyword, self.image_urls, self.output_dict, self.lock)
        

def get_urls(filename, start_date, end_date):
    urls = []
    with open(f'{filename}', 'r', encoding="utf8") as fp:
        for line in fp:
            data = json.loads(line)
            if data['date'] >= start_date and data['date'] <= end_date:
                urls.append(data['url'])
    return urls

def output_jsonl(func, output_dict, start_date, end_date):
        with open(f'./{func}_{start_date}_{end_date}.json', 'w', encoding="utf8") as op:
            op.write(json.dumps(output_dict, indent=2))

if __name__ == '__main__':
    
    num_threads = 12
    urls_queue = queue.Queue()

    if len(sys.argv) < 2:
        print('Usage: python 311511056.py crawl')
        print('Usage: python 311511056.py push {start_data} {end_date}')
        print('Usage: python 311511056.py popular {start_data} {end_date}')
        print('Usage: python 311511056.py keyword {keyword} {start_date} {end_date}')
        exit(1)
    
    if sys.argv[1] == 'crawl':
        crawler = Crawler()
        print('Crawling all articles ...')
        crawler.get_all_articles()

    elif sys.argv[1] == 'push':   
        print('Crawling push data ...')
        start_date = sys.argv[2]
        end_date = sys.argv[3]
        urls = get_urls('all_article.jsonl', start_date, end_date)
        
        threads = [Worker(urls_queue, 'push') for i in range(num_threads)]
        for url in urls:
            urls_queue.put(url)
        Worker.bar = tqdm(total=urls_queue.qsize())
        
        
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        Worker.bar.close()

        Worker.crawler.get_push(Worker.push_ids, Worker.boo_ids, Worker.output_dict)
        output_jsonl('push', Worker.output_dict, start_date, end_date)

    elif sys.argv[1] == 'popular':
        print('Crawling popular data ...')

        start_date = sys.argv[2]
        end_date = sys.argv[3]
        urls = get_urls('all_popular.jsonl', start_date, end_date)
        
        for url in urls:
            urls_queue.put(url)
        
        threads = [Worker(urls_queue, 'popular') for i in range(num_threads)]
        Worker.bar = tqdm(total=urls_queue.qsize())
        Worker.popular_dict['number_of_popular_articles'] = urls_queue.qsize()
        
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        Worker.bar.close()
        Worker.popular_dict['image_urls'] = Worker.image_urls
        output_jsonl('popular', Worker.popular_dict, start_date, end_date)

    elif sys.argv[1] == 'keyword':
        print('Crawling keyword data ...')    

        keyword = sys.argv[2]
        start_date = sys.argv[3]
        end_date = sys.argv[4]

        urls = get_urls('all_article.jsonl', start_date, end_date)
        for url in urls:
            urls_queue.put(url)
        
        Worker.keyword = keyword
        threads = [Worker(urls_queue, 'keyword') for i in range(num_threads)]
        Worker.bar = tqdm(total=urls_queue.qsize())
        
        for thread in threads:
            thread.start()
        for thread in threads: 
            thread.join()
        Worker.bar.close()
        
        # output json
        with open(f'./keyword_{keyword}_{start_date}_{end_date}.json', 'w', encoding="utf8") as op:
            op.write(json.dumps(Worker.output_dict, indent=2)) 

    elif sys.argv[1] == 'test':
        crawler = Crawler()
        print('Testing ...')
        
    else:    
        print('Error: Wrong command!')
