--Müşteri Tipine Göre En Çok Tercih Edilen Ödeme Yöntemi


select Customer_type, payment, count(*) as payment_count
from avm.dbo.[small-mall-dataset]
group by Payment, Customer_type
order by Payment, Customer_type








