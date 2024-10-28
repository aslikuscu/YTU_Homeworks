--musteri turune Göre Satış Miktarı:

select Customer_type, sum(s.Quantity) as total_sales
from avm.dbo.[small-mall-dataset] as s
group by s.Customer_type 