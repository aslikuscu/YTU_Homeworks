--Her Şehirdeki en yuksek ve en dusuk urun turu ve satis miktari

select city,MAX(Product_line) as product_line, sum(Quantity) as max_productline, 
MIN(Product_line) as product_line, SUM(Quantity) as min_product_line
from avm.dbo.[small-mall-dataset]
group by City