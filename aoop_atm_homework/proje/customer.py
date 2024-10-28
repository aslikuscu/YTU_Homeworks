
class Customer:
    def __init__(self, name, address, email, phone, status, card_number, card_customer_name, card_expiry, card_pin, account_number):
        self.__name = name
        self.__address = address
        self.__email = email
        self.__phone = phone
        self.__status = status
        self.__card = Card(card_number, card_customer_name, card_expiry, card_pin)
        self.__account = Account(account_number)

    # Getter ve Setter metotları
    def get_name(self):
        return self.__name

    def set_name(self, new_name):
        self.__name = new_name

    def get_address(self):
        return self.__address

    def set_address(self, new_address):
        self.__address = new_address

    def get_email(self):
        return self.__email

    def set_email(self, new_email):
        self.__email = new_email

    def get_phone(self):
        return self.__phone

    def set_phone(self, new_phone):
        self.__phone = new_phone

    def get_status(self):
        return self.__status

    def set_status(self, new_status):
        self.__status = new_status

    def make_transaction(self, transaction):
        None

    def get_billing_address(self):
        None



class Card:
    def __init__(self, number, customer_name, expiry, pin):
        self.__card_number = number
        self.__customer_name = customer_name
        self.__card_expiry = expiry
        self.__pin = pin

    # Getter ve Setter metotları
    def get_card_number(self):
        return self.__card_number

    def set_card_number(self, new_number):
        self.__card_number = new_number

    def get_customer_name(self):
        return self.__customer_name

    def set_customer_name(self, new_name):
        self.__customer_name = new_name

    def get_card_expiry(self):
        return self.__card_expiry

    def set_card_expiry(self, new_expiry):
        self.__card_expiry = new_expiry

    def get_pin(self):
        return self.__pin

    def set_pin(self, new_pin):
        self.__pin = new_pin

    def get_billing_address(self):
        None


class Account:
    def __init__(self, account_number):
        self.__account_number = account_number
        self.__total_balance = 0.0
        self.__available_balance = 0.0

    # Getter ve Setter metotları
    def get_account_number(self):
        return self.__account_number

    def set_account_number(self, new_account_number):
        self.__account_number = new_account_number

    def get_total_balance(self):
        return self.__total_balance

    def set_total_balance(self, new_total_balance):
        self.__total_balance = new_total_balance

    def get_available_balance(self):
        return self.__available_balance

    def set_available_balance(self, new_available_balance):
        self.__available_balance = new_available_balance


customer_list = []

def add_customer(name, address, email, phone, status, card_number, card_customer_name, card_expiry, card_pin, account_number):
    new_customer = Customer(name, address, email, phone, status, card_number, card_customer_name, card_expiry, card_pin, account_number)
    customer_list.append(new_customer)

def check_account_number(account_number_to_check):
    for customer in customer_list:
        if customer._Customer__account._Account__account_number == account_number_to_check:
            return True  # Eğer hesap numarası eşleşiyorsa True döndür
    return False  # Eğer eşleşen hesap numarası yoksa False döndür

# customer_list içindeki her bir müşteri örneği için card_pin'i kontrol etme
def check_card_pin(pin_to_check, account_number):
    for customer in customer_list:
        if customer._Customer__card._Card__pin == pin_to_check and customer._Customer__account._Account__account_number == account_number:
            return True  # Eğer pin ve account_number eşleşiyorsa True döndür
    return False  # Eğer eşleşen pin veya account_number yoksa False döndür