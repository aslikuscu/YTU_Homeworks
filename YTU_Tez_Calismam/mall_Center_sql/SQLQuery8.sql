--Her Şehirdeki Toplam Satış Gelirinin Zaman İçindeki Değişimi


select city, date ,sum(Total)
from avm.dbo.[small-mall-dataset]
group by Date,City
order by Date,City desc






