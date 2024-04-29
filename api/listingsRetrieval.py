from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import easyocr
from PIL import Image

reader = easyocr.Reader(['en'])

options = Options()
options.headless = False

scrollTime = 1

driver_path = r'C:\Users\maxpg\OneDrive\Desktop\BookMate\amazon_web_scraping'
link = "https://www.amazon.com/Six-Crows-Leigh-Bardugo/dp/125007696X/ref=sr_1_1?sr=8-1"
driver = webdriver.Chrome(options=options)

driver.get(link)

# Keep retrying the Captcha until a success is found/No Captcha, Every attempt changes the captcha on amazon
while(1):
    try:
        listings = driver.find_element(By.XPATH, "//*[@id='dynamic-aod-ingress-box']/div/div[2]/a/span/span[1]")
        break
    except:
        driver.save_screenshot(f"{driver_path}/captcha.png")
        img = Image.open(f"{driver_path}/captcha.png")

        # Gets the rectangle surrounding the Captcha
        letters = img.crop((950, 740, 1600, 925))
        letters.save(f"{driver_path}/cropped.png")

        # Read the screenshot and get the characters
        solution = reader.readtext(f"{driver_path}/cropped.png", detail=1)

        # Input the solution
        captcha_input = driver.find_element(By.XPATH, "//*[@id='captchacharacters']")
        captcha_input.send_keys(solution[0][1])

        button = driver.find_element(By.XPATH, "/html/body/div/div[1]/div[3]/div/div/form/div[2]/div/span")
        button.click()

if(listings == None):
    print("not found")
    exit()

listings.click()

driver.implicitly_wait(2*scrollTime)

counter = 1

# Loads the first 30 items
while counter < 3:
    try:
        frame = driver.find_element(By.XPATH, f"//*[@id='aod-price-{counter*10}']/div/span/span[1]")
        driver.execute_script("arguments[0].scrollIntoView(true)", frame);
        time.sleep(1.5)
    except:
        break

    counter+=1

# Loads the remaining items
while True:
    try:
        new = driver.find_element(By.XPATH, "//*[@id='aod-show-more-offers']")
        new.click()
    except:
        break;

count = 0

price_list = []

# Get all price listings
while True:
    try:
        price = driver.find_element(By.XPATH, f"//*[@id='aod-price-{count}']/div/span/span[1]").get_attribute("textContent")
        element = price.replace("$", "")
        price_list.append(element)
    except:
        break
    count += 1

quality = driver.find_elements(By.XPATH, "//*[@id='aod-offer-heading']/h5")
quality_list = []

for index, qual in enumerate(quality):

    # There is one extra element that has this XPATH, duplicating the quality of the first listing, so it's ignored
    if(index == 0):
        continue

    # Removes extra characters, all used books have an actual quality to them
    element = " ".join(qual.get_attribute("textContent").split())
    quality_list.append(element.replace("Used - ", ""))

book_list = []

# Combines price and quality
for a, b in zip(price_list, quality_list):
    book_list.append((a, b))

print(book_list)

driver.close()