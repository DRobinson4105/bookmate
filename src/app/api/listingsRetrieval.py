from selenium import webdriver
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

options = Options()
options.headless = False

scroll_time = 1

driver_path = 'C:/Users/maxpg/OneDrive/Desktop/BookMate/amazon_web_scraping'
amazon_link = "https://www.amazon.com/Dark-Secret-Wings-Graphic-Novel/dp/1338344218/ref=sr_1_17?dib=eyJ2IjoiMSJ9.I16HBQGzebuA2K7-wiT3OGDWCYB995UcRnK7cQmqzO3c8sV5CRY7zfKmx9C5JUIw8WCBadkdgZ00f8j-y34wrc0QagR-Goqf0pRL46ztHWqHMZJuWXEGEecKTLWZyZyEwWPL66XBccnGK3pqpxmdl1KmpFtUd0Y6GaQUhudU0-OTS0w08660YDU0THKxmkhw_mmjjywJzhJCpHk0MVrMla7G1WaplxOr0v5NmB9kFis.Nm5_GHTYz2ct8aGrvhuWWsQNeTtyoFaQx2E2bx7H4Yk&dib_tag=se&hvadid=580696694281&hvdev=c&hvlocphy=9011784&hvnetw=g&hvqmt=b&hvrand=73845424378692618&hvtargid=kwd-436022913818&hydadcr=22563_13493224&keywords=the+wings+of+fire+books&qid=1713659481&sr=8-17"
driver = webdriver.Chrome(options=options)

driver.get(amazon_link)
driver.implicitly_wait(10)

listings = driver.find_element(By.XPATH, "//*[@id='dynamic-aod-ingress-box']/div/div[2]/a/span/span[1]")

if(listings == None):
    print("not found")

listings.click()

driver.implicitly_wait(4 * scroll_time)

counter = 1

# Loads the first 30 items
#TODO Lower sleep time
while counter < 3:
    try:
        frame = driver.find_element(By.XPATH, f"//*[@id='aod-price-{counter*10}']/div/span/span[1]")
        driver.execute_script("arguments[0].scrollIntoView(true)", frame);
        time.sleep(2)
    except:
        break

    counter+=1

#Loads the remaining listings
while True:
    try:
        new = driver.find_element(By.XPATH, "//*[@id='aod-show-more-offers']")
        new.click()
    except:
        break;

count = 0

#Gets all the listings
#TODO takes too long to end the loop on the except
while True:
    print(f"item {count}")
    try:
        price = driver.find_element(
            By.XPATH, f"//*[@id='aod-price-{count}']/div/span/span[1]"
        ).get_attribute("textContent")
        print(price)
    except:
        break
    count += 1
