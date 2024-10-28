from customer import Card, Account, add_customer, check_account_number, check_card_pin, customer_list, Customer
from atm import Screen, Printer

def login_required(func):
    def wrapper(*args, **kwargs):
        if not login:
            print("Giriş yapmanız gerekiyor.")
            return
        return func(*args, **kwargs)
    return wrapper

add_customer("Beyza", "İstanbul", "beyza@email.com", "5566778899", "Gümüş", "1122334455667788", "Beyza Yılmaz", "10/25", 1214, 456456)
add_customer("Zeynep", "İzmir", "zeynep@email.com", "5566112233", "Bronz", "1112223334445556", "Zeynep Mutlu", "09/27", 1213, 123456)
add_customer("Ali", "Edirne", "ali@email.com", "5544771155", "Altın", "1111222233334444", "Ali Özdemir", "08/26", 1215, 123123)

login = False
login_attempts = 3

screen_instance = Screen()
printer = Printer()

@login_required
def para_cek():
    global login
    global login_attempts
    global user_account

    select_withdraw = int(input(""" 
        1.10 ₺
        2.20 ₺
        3.30 ₺
        4.Özel tutar
        5.İptal et
        """))

    if select_withdraw == 4:
        withdraw_value = int(input("Ne kadar ₺ çekmek istersiniz?"))
    elif select_withdraw == 5:
        return

    else:
        withdraw_value = select_withdraw * 10

    if user_account.get_available_balance() < withdraw_value:
        print("Yeterli fon mevcut değil")
        return

    user_account.set_available_balance(user_account.get_available_balance() - withdraw_value)
    print(f"{withdraw_value}₺ çekildi. Yeni bakiye: {user_account.get_available_balance()}₺")

@login_required
def para_yatir():
    global user_account

    deposit_value = int(input("Kaç para yatırmak istiyorsunuz? "))
    user_account.set_available_balance(user_account.get_available_balance() + deposit_value)
    print(f"{deposit_value}₺ yatırıldı. Yeni bakiye: {user_account.get_available_balance()}₺")

@login_required
def bakiye_sorgula():
    global user_account

    print(f"Bakiyeniz: {user_account.get_available_balance()}₺")

@login_required
def aktarim_yap():
    global user_account

    transfer_amount = int(input("Kaç tl transfer etmek istersiniz? "))
    if user_account.get_available_balance() < transfer_amount:
        print("Yeterli fon mevcut değil ")
        return

    receiver_id = int(input("Alıcının hesap numarasını girin "))
    if check_account_number(receiver_id):
        user_account.set_available_balance(user_account.get_available_balance() - transfer_amount)
        print(f"{transfer_amount}₺ başarıyla gönderildi. Yeni bakiye: {user_account.get_available_balance()}₺")
        receipt = input("Makbuzu yazdır(evet/hayır)").lower()
        if receipt == 'evet':
            printer.print_receipt()
    else:
        print("Bu hesap bulunamadı.")

@login_required
def cikis_yap():
    print("Tekrar Görüşürüz")
    global login
    login = False

while True:
    if not login:
        account_number = screen_instance.get_input()
        user_account = Account(account_number)
        password = int(input("Hoş geldiniz! Lütfen şifrenizi girin."))

    if check_card_pin(password, account_number):
        login = True
        print("""
        1.Para Çek
        2.Para Yatır
        3.Bakiye Sorgulama
        4.Aktarım
        5.Çıkış
        """)
        choose = int(input("Bir işlem seçin "))

        if choose == 1:
            para_cek()

        elif choose == 2:
            para_yatir()

        elif choose == 3:
            bakiye_sorgula()

        elif choose == 4:
            aktarim_yap()

        elif choose == 5:
            cikis_yap()

        else:
            print("Lütfen arasında seçim yapın 1-5")

    else:
        login_attempts -= 1
        if login_attempts <= 0:
            print("Kartınız bloke oldu!")
            break
