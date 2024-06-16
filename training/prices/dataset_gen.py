from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import easyocr
from PIL import Image
from price_predictor import BookDataset, Model
import pandas as pd
import torch

reader = easyocr.Reader(['en'])

options = Options()
options.headless = False

scrollTime = 1

condition_stoi = {'New': 0, 'Like New': 1, 'Very Good': 2, 'Good': 3, 'Acceptable': 4}

def get_listings(isbn_13):
    isbn_string = str(isbn_13)
    check_digit = 0
    for i in range(9):
        check_digit += (10 - i) * int(isbn_string[3 + i])

    check_digit = str(11 - (check_digit % 11))

    if check_digit == 10:
        check_digit = "X"

    isbn_10 = isbn_string[slice(3, 12)] + check_digit

    link = f"https://www.amazon.com/dp/{isbn_10}"
    driver = webdriver.Chrome(options=options)

    driver.get(link)

    # Keep retrying the Captcha until a success is found/No Captcha, Every attempt changes the captcha on amazon
    while True:
        try:
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
        except:
            return None, None, None

    if(listings == None): exit()

    # Get rank for item
    try:
        rank_element = driver.find_element(By.XPATH,
                                    "//*[@id='detailBulletsWrapper_feature_div']/ul[1]/li/span").get_attribute(
                                        "textContent")
        tokens = rank_element.strip().split()
    except:
        tokens = []

    listings.click()

    # Extract Rank
    rank = 0
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
    prices_list = []

    # Get all price listings
    while True:
        try:
            price = driver.find_element(By.XPATH, f"//*[@id='aod-price-{count}']/div/span/span[1]").get_attribute("textContent")
            element = price.replace("$", "")
            element = float(element)
            prices_list.append(element)
        except:
            break
        count += 1

    quality = driver.find_elements(By.XPATH, "//*[@id='aod-offer-heading']/h5")
    conditions_list = []

    for index, qual in enumerate(quality):
        # There is one extra element that has this XPATH, duplicating the quality of the first listing, so it's ignored
        if(index == 0):
            continue

        # Removes extra characters, all used books have an actual quality to them
        element = " ".join(qual.get_attribute("textContent").split())
        element = element.replace("Used - ", "").replace("Collectible - ", "")

        # Convert quality to numerical value
        conditions_list.append(condition_stoi[element])

    # Gets specific item data in tuple
    details_list = (prices_list[0], rank)

    driver.close()

    return details_list, prices_list, conditions_list

isbnFile = open('isbns.txt', 'r')
isbns = [int(line) for line in isbnFile.readlines()]
# isbns = [9781936806119, 9780805005028, 9781787139787]

def calc_price(details_list, prices_list, conditions_list):
    avg_listing, length = 0, len(prices_list)
    for i in range(length):
        avg_listing += prices_list[i] * (5 - conditions_list[i]) / 5
    avg_listing /= length

    return round(avg_listing * 0.9 + details_list[0] * 0.1, 2)

details_list, prices_list, conditions_list = [], [], []
temp = []
for isbn in isbns:
    a, b, c = get_listings(isbn)
    if a == None:
        print(isbn)
        continue
    temp.append(isbn)

    details_list.append(a)
    prices_list.append(b)
    conditions_list.append(c)

# Rebuild isbn list with only valid isbns
isbns = temp
length = len(isbns)

targets_list = [calc_price(details_list[i], prices_list[i], conditions_list[i]) for i in range(length)]
df = pd.DataFrame({
    'details': details_list,
    'prices': prices_list,
    'conditions': conditions_list,
    'targets': targets_list
})
df.to_csv('dataset.csv')

# Test
df2 = pd.read_csv('dataset.csv')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
price_model = Model().to(device)
price_model.load_state_dict(torch.load(f="model.pt"))

convert_to_tensor = lambda column: [[float(x) for x in list.strip(')(][').split(", ")] for list in df2[column].tolist()]
dataset = BookDataset(convert_to_tensor('details'), convert_to_tensor('prices'), convert_to_tensor('conditions'), df2['targets'].tolist())
details, prices, conditions, _ = dataset.__getitem__(0);
result = price_model(details, prices, conditions).tolist()
print(result)