class Bank:
    def __init__(self, name, bank_code):
        self.__name = name
        self.__bank_code = bank_code

    # Getter ve Setter metotları
    def get_name(self):
        return self.__name

    def set_name(self, new_name):
        self.__name = new_name

    def get_bank_code(self):
        return self.__bank_code

    def set_bank_code(self, new_bank_code):
        self.__bank_code = new_bank_code

    def add_atm(self, atm):
        # Buraya ATM ekleme işlemlerini ekleyebilirsiniz.
        pass


class ATM:
    def __init__(self, id, location):
        self.__atm_id = id
        self.__location = location

    # Getter ve Setter metotları
    def get_atm_id(self):
        return self.__atm_id

    def set_atm_id(self, new_id):
        self.__atm_id = new_id

    def get_location(self):
        return self.__location

    def set_location(self, new_location):
        self.__location = new_location

    def authenticate_user(self):
        # Kullanıcı doğrulama işlemlerini ekleyebilirsiniz.
        pass

    def make_transaction(self, customer, transaction):
        # İşlem yapma işlemlerini ekleyebilirsiniz.
        pass


class CashDispenser:
    def __init__(self):
        self.__total_five_dollar_bills = 100
        self.__total_twenty_dollar_bills = 50

    def dispense_cash(self, amount):
        # Bu metot, belirli bir miktar parayı ATM'den verme işlemini simgeliyor.
        # Gerçek bir uygulamada bu işlem daha karmaşık olacaktır.
        print(f"Dispensing {amount}$ in cash")

    def can_dispense_cash(self):
        # Bu metot, ATM'nin yeterli miktarda nakit para bulundurup bulunmadığını kontrol ediyor.
        # Gerçek bir uygulamada daha karmaşık bir kontrol yapısı düşünülmelidir.
        return self.__total_five_dollar_bills > 0 or self.__total_twenty_dollar_bills > 0


class Keypad:
    def get_input(self):
        # Kullanıcıdan giriş almak için gerekli işlemler
        pass


class Screen:
    def show_message(self, message):
        print(message)

    def get_input(self):
        try:
            return int(input("Hoş geldin! Lütfen hesap numaranızı giriniz:"))
        except ValueError:
            print("Geçersiz Giriş. Lütfen geçerli bir hesap numarası girin.")
            return None


# Makbuz yazdırma işlemleri
class Printer:
    def print_receipt(self):
        print("RECEIPT")



 # Çek yatırma işlemleri
class CheckDeposit:
    def __init__(self):
        pass


 # Nakit para yatırma işlemleri
class CashDeposit:
    def __init__(self):
        pass


class DepositSlot(ATM):
    def __init__(self):
        self.__total_amount = 0.0

    def get_total_amount(self):
        return self.__total_amount


# Çek miktarını alma işlemleri
class CheckDepositSlot(DepositSlot):
    def get_check_amount(self):
        pass



 # Tl banknot alma işlemleri
class CashDepositSlot(DepositSlot):
    def receive_dollar_bill(self):
        pass
