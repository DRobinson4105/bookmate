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

def convert_isbn(isbn_13):
    isbn_string = str(isbn_13)
    check_digit = 0
    for i in range(9):
        check_digit += (10 - i) * int(isbn_string[3 + i])

    check_digit = str(11 - (check_digit % 11))

    if check_digit == 10:
        check_digit = "X"

    isbn_10 = isbn_string[slice(3, 12)] + check_digit

    return isbn_10

def get_listings(isbn_10):
    link = f"https://www.amazon.com/dp/{isbn_10}"
    driver = webdriver.Chrome(options=options)

    driver.get(link)
    # Keep retrying the Captcha until a success is found/No Captcha, Every attempt changes the captcha on amazon
    while(1):
        try:
            listings = driver.find_element(By.XPATH, "//*[@id='dynamic-aod-ingress-box']/div/div[2]/a/span/span[1]")
            break
        except:
            driver.save_screenshot("./captcha.png")
            img = Image.open("./captcha.png")

            # Crop center of image to get captcha
            width, height = img.size
            left = width / 4
            top = height / 4
            right = 3 * width / 4
            bottom = 3 * height / 4
            img = img.crop((left, top, right, bottom))
            img.save('./cropped.png')

            # Read the screenshot and get the characters
            solution = reader.readtext("./cropped.png", detail=1)

            # Input the solution
            captcha_input = driver.find_element(By.XPATH, "//*[@id='captchacharacters']")
            captcha_input.send_keys(solution[0][1])

            button = driver.find_element(By.XPATH, "/html/body/div/div[1]/div[3]/div/div/form/div[2]/div/span")
            button.click()

    if(listings == None):
        print("not found")
        exit()

    listings.click()

    # Get rank for item
    rank_element = driver.find_element(By.XPATH,
                                    "//*[@id='detailBulletsWrapper_feature_div']/ul[1]/li/span").get_attribute(
                                        "textContent")

    tokens = rank_element.strip().split()

    # Extract Rank
    for token in tokens:
        if token[0] == '#':
            rank = int(token.replace('#', "").replace(',', ""))
            break

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
            element = float(element)
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
        element = element.replace("Used - ", "")

        # Convert quality to numerical value
        if element == "New":
            quality_list.append(0)
        elif element == "Like New":
            quality_list.append(1)
        elif element == "Very Good":
            quality_list.append(2)
        elif element == "Good":
            quality_list.append(3)
        elif element == "Acceptable":
            quality_list.append(4)

    book_list = []

    # Combines price and quality
    for a, b in zip(price_list, quality_list):
        book_list.append((a, b))

    # Gets specific item data in tuple
    item = (price_list[0], rank)

    driver.close()

    return item, book_list

print(get_listings(convert_isbn(9781936806119)))