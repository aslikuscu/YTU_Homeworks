--odeme turune Göre Satış Miktarı:

select payment, sum(s.Quantity) as total_sales
from avm.dbo.[small-mall-dataset] as s
group by s.Payment 