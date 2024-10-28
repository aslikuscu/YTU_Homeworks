--sehirlere Göre Satış Miktarı:

select city, sum(s.Quantity) as total_sales
from avm.dbo.[small-mall-dataset] as s
group by s.City 