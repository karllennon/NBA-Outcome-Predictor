from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
driver.get("https://www.rotowire.com/basketball/injury-report.php")

# Wait longer for JS to render
time.sleep(5)

# Search for player names in the page
from bs4 import BeautifulSoup
soup = BeautifulSoup(driver.page_source, 'html.parser')

# Print all text that might contain player data
# Look for any divs/tables with injury-related classes
injury_divs = soup.find_all(class_=lambda x: x and 'injury' in x.lower())
print(f"Found {len(injury_divs)} injury-related elements")

for div in injury_divs[:5]:
    print(div.get('class'), ':', div.text.strip()[:100])

driver.quit()