--Satışların Haftalık ve Aylık Trendleri:

select DATEPART(MONTH,Date), sum(Quantity)
from avm.dbo.[small-mall-dataset]
group by DATEPART(MONTH,Date)
order by DATEPART(MONTH,Date) asc






