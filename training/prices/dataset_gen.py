from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import easyocr
from PIL import Image
from utils import BookDataset, Model
import pandas as pd
import torch
from tqdm.auto import tqdm

reader = easyocr.Reader(['en'])

condition_stoi = {'New': 1, 'Like New': 0.85, 'Very Good': 0.7, 'Good': 0.6, 'Acceptable': 0.4}
condition_itos = {v: k for k, v in condition_stoi.items()}

def get_listings(isbn_13):
    # try:
    isbn_string = str(isbn_13)
    check_digit = 0
    for i in range(9):
        check_digit += (10 - i) * int(isbn_string[3 + i])

    check_digit = str(11 - (check_digit % 11))

    if check_digit == 10:
        check_digit = "X"

    isbn_10 = isbn_string[slice(3, 12)] + check_digit

    link = f"https://www.amazon.com/dp/{isbn_10}"
    driver = webdriver.Chrome()

    driver.get(link)

    # Keep retrying the Captcha until a success is found/No Captcha, Every attempt changes the captcha on amazon
    while(1):
        try:
            listings = driver.find_element(By.XPATH, '//*[@id="dynamic-aod-ingress-box"]/div/div[2]/a')
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
            try:
                captcha_input = driver.find_element(By.XPATH, "//*[@id='captchacharacters']")
            except:
                print('STOP')
                return None, None
            captcha_input.send_keys(solution[0][1])

            button = driver.find_element(By.XPATH, "/html/body/div/div[1]/div[3]/div/div/form/div[2]/div/span")
            button.click()

    if (listings == None):
        return None, None

    listings.click()
    driver.implicitly_wait(2)
    counter = 1

    # Loads the first 30 items
    while counter < 3:
        try:
            frame = driver.find_element(By.XPATH, f"//*[@id='aod-price-{counter*10}']/div/span/span[1]")
            driver.execute_script("arguments[0].scrollIntoView(true)", frame);
            time.sleep(1.5)
            counter += 1
        except:
            break

    # Loads the remaining items
    while True:
        try:
            new = driver.find_element(By.XPATH, "//*[@id='aod-show-more-offers']")
            new.click()
        except:
            break

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
    
    print(prices_list)

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

    driver.close()

    return prices_list, conditions_list

isbnFile = open('isbns.txt', 'r')
outputFile = open('filteredIsbns.txt', 'w')
isbns = [int(line) for line in isbnFile.readlines()]

def calc_price(prices_list, conditions_list):
    for price, condition in zip(prices_list, [condition_itos(x) for x in conditions_list]):
        print(price, condition, "\n")
    return input("Selling Price:")

prices_list, conditions_list = [], []
for isbn in tqdm(isbns):
    a, b = get_listings(isbn)
    if a == None or len(a) == 0: continue
    outputFile.write(str(isbn) + '\n')

    prices_list.append(a)
    conditions_list.append(b)

# Rebuild isbn list with only valid isbns
length = len(prices_list)

targets_list = [calc_price(prices_list[i], conditions_list[i]) for i in range(length)]

df = pd.DataFrame({
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
dataset = BookDataset(convert_to_tensor('prices'), convert_to_tensor('conditions'), df2['targets'].tolist(), device=device)
prices, conditions, target = dataset.__getitem__(0);
result = price_model(prices.unsqueeze(0), conditions.unsqueeze(0)).tolist()
print("Expected: ", target.item(), "Actual: ", result)