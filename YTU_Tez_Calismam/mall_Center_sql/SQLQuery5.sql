--tarihlere Göre Satış Miktarı:

select Date, sum(s.Quantity) as total_sales
from avm.dbo.[small-mall-dataset] as s
group by s.Date 