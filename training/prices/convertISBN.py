def isbn10_to_isbn13(isbn10):
    isbn10 = isbn10.replace('-', '')
    if len(isbn10) != 10:
        return 0

    isbn13 = '978' + isbn10[:-1]

    check_sum = 0
    for i, char in enumerate(isbn13):
        if i % 2 == 0:
            check_sum += int(char)
        else:
            check_sum += 3 * int(char)
    
    check_digit = 10 - (check_sum % 10)
    if check_digit == 10:
        check_digit = 0

    return isbn13 + str(check_digit)

isbn = input()
while True:
    try:
        print(isbn10_to_isbn13(isbn))
        isbn = input()
    except: break