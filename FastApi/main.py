import cloudscraper

scraper = cloudscraper.create_scraper()
response = scraper.get('https://www.atbmarket.com/catalog/461-do-velikodnya')
with open('answer.txt', 'w', encoding='utf-8') as file:
    file.write(response.text)