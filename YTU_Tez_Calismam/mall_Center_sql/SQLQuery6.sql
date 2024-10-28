--cinsiyete Göre Satış Miktarı:

select Gender, sum(s.Quantity) as total_sales
from avm.dbo.[small-mall-dataset] as s
group by s.Gender 