--Her Şubede En Yüksek ve En Düşük Rating Değerine Sahip Müşteri Türü:

SELECT 
    Branch,
    MAX(Rating) AS Highest_Rating,
    MAX(Customer_type) AS Customer_Type_Highest_Rating,
    MIN(Rating) AS Lowest_Rating,
    MIN(Customer_type) AS Customer_Type_Lowest_Rating
FROM 
    avm.dbo.[small-mall-dataset] -- Tablo adını buraya girin
GROUP BY 
    Branch;