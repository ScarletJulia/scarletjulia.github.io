---
layout: post
title: "San Francisco Crimes"
featured-img: object_detection
categories: [Data Analysis]
---

# The goal is to do free-form data analysis in order to extract relevant information from the database.

## Topics:

> 1) Different categories of crime
> 2) Plot most common crimes
> 3) District with Most Crime 
>    1) Crimes by District 
> 4) Crime count on each day 
>    1) Crime on each day
> 5) Solved Crimes 
> 6) Top 15 Regions in Crime 
> 7) Density of crime in San Francisco

## What you'll see

## Entendimento dos dados:
- Tamanho do conjunto de dados usando a função dados.shape (150500 entradas e 13 colunas).
- Abaixo as colunas:
1. "IncidntNum" é o número do incidente, que é um identificador único para cada registro no conjunto de dados.
2. "Category" é a categoria geral do crime, como roubo, furto ou violência doméstica.
3. "Descript" é uma descrição mais detalhada do crime, como "roubo de veículo".
4. "DayOfWeek" é o dia da semana em que o crime ocorreu.
5. "Date" é a data em que o crime ocorreu.
6. "Time" é a hora em que o crime ocorreu.
7. "PdDistrict" é o distrito policial em que o crime ocorreu.
8. "Resolution" é o resultado do incidente, como "prender o suspeito".
9. "Address" é o endereço onde o crime ocorreu.
10. "X" e "Y" são as coordenadas geográficas do local do crime.
11. "Location" é a localização do crime, dada pelas coordenadas geográficas.
12. "PdId" é o ID do departamento de polícia atribuído ao incidente.

### Major crimes in San Francisco
![imagem](https://user-images.githubusercontent.com/114709169/204068253-cf5ca369-fd7d-4959-8361-43fc87c7f8db.png)

A categoria mais frequente é "LARCENY/THEFT" com 40.409 ocorrências, seguida por "OTHER OFFENSES" com 19.599 ocorrências e "NON-CRIMINAL" com 17.866 ocorrências.
A distribuição e frequência dos diferentes tipos de crimes no conjunto de dados e podem ser usados para priorizar áreas de policiamento ou investigação.

### District with Most Crime
![imagem](https://user-images.githubusercontent.com/114709169/204068273-bf100813-73cf-4ec9-9da9-70cdb8c60915.png)

Essa tabela apresenta a contagem de incidentes policiais por distrito. Cada linha representa um distrito policial e a coluna "IncidntNum" mostra a quantidade de incidentes registrados em cada distrito.

Por exemplo, o distrito com o maior número de incidentes é o Southern com 28.445 registros, seguido pelo Northern com 20.100 registros e Mission com 19.503 registros.

Esses resultados ajudam a entender a distribuição dos incidentes policiais por distrito e podem ser usados para avaliar a efetividade das políticas de segurança em diferentes áreas, identificar áreas de maior risco para a criminalidade e planejar estratégias de policiamento e prevenção de crimes.

### Crimes in the neighborhood
![imagem](https://user-images.githubusercontent.com/114709169/204068342-6b19bba0-6825-43a6-94d3-b727e37dcfe2.png)

Este gráfico apresenta a contagem de ocorrências de crimes por categoria em cada distrito policial. Cada linha representa um distrito policial e cada coluna representa uma categoria de crime. A célula em cada linha e coluna mostra a quantidade de ocorrências da categoria de crime no distrito policial correspondente.

Por exemplo, no distrito Tenderloin, a categoria de crime mais frequente é "LARCENY/THEFT" com 1.825 ocorrências, seguida por "NON-CRIMINAL" com 1.379 ocorrências e "OTHER OFFENSES" com 1.237 ocorrências.

Esses resultados ajudam a entender a distribuição de diferentes tipos de crimes em cada distrito policial e podem ser usados para identificar áreas de maior risco, avaliar a efetividade das políticas de segurança em diferentes áreas e planejar estratégias de policiamento e prevenção de crimes específicos para cada distrito policial.

### Top 15 Regions in Crime
![imagem](https://user-images.githubusercontent.com/114709169/204068386-88f0a78e-84f9-4440-a38a-ecc6c993f099.png)

Esse gráfico apresenta a contagem de ocorrências de crimes por categoria em diferentes dias da semana em um conjunto de dados. Cada barra no eixo x representa um dia da semana, e no eixo y representa uma categoria de crime e a célula em cada linha e coluna mostra a quantidade de ocorrências da categoria de crime no dia da semana correspondente.

Por exemplo, a categoria de crime mais frequente no dia da semana sexta-feira é "LARCENY/THEFT" com 6.477 ocorrências, seguida por "LARCENY/THEFT" novamente no sábado com 6.384 ocorrências e na quinta-feira com 5.538 ocorrências.

Esses resultados ajudam a entender a distribuição de diferentes tipos de crimes em diferentes dias da semana e podem ser usados para avaliar a efetividade das políticas de segurança em diferentes dias da semana. Por exemplo, pode-se identificar que o roubo é mais frequente nos finais de semana e que, portanto, pode ser necessário reforçar as medidas de segurança nesses dias. Além disso, esses resultados também podem ser usados para planejar estratégias de policiamento e prevenção de crimes específicos para cada dia da semana, com o objetivo de reduzir a incidência de crimes em dias específicos.

A análise desses dados também pode ajudar a identificar tendências ou padrões em diferentes dias da semana, como a ocorrência de crimes mais violentos em determinados dias ou a concentração de crimes em áreas específicas em dias específicos da semana. Isso pode levar a uma maior efetividade na prevenção e combate ao crime, além de promover a segurança dos cidadãos.

### Density of crime in San Francisco
![imagem](https://user-images.githubusercontent.com/114709169/204068442-d5f2070b-57d8-45f8-9756-70f929ef2702.png)

Esse gráfico apresenta a contagem de incidentes policiais por bairro em nosso conjunto de dados. Cada linha representa um bairro e a coluna "Count" mostra a quantidade de incidentes registrados em cada bairro.

Por exemplo, o bairro com o maior número de incidentes é o Southern com 28.445 registros, seguido pelo Bayview com 14.303 registros e Mission com 19.503 registros.

Esses resultados ajudam a entender a distribuição dos incidentes policiais por bairro e podem ser usados para avaliar a efetividade das políticas de segurança em diferentes bairros, identificar áreas de maior risco para a criminalidade e planejar estratégias de policiamento e prevenção de crimes. Além disso, esses dados podem ser úteis para as autoridades locais e para os moradores dessas áreas, pois podem fornecer informações importantes sobre a segurança em suas comunidades.

A análise de clusters pode ser usada para agrupar distritos policiais com características semelhantes, como a frequência de diferentes tipos de crimes, a densidade populacional, a presença de pontos turísticos ou áreas comerciais, entre outros fatores. Esses grupos podem ser usados para avaliar a efetividade das políticas de segurança em diferentes áreas e para planejar estratégias de policiamento e prevenção de crimes específicos para cada grupo.

Além disso, a análise de clusters pode ajudar a identificar áreas com altas taxas de criminalidade e a priorizar a alocação de recursos de segurança pública para essas áreas. Isso pode levar a uma maior efetividade na prevenção e combate ao crime, além de promover a segurança dos cidadãos.



## Libs

- [Python](https://www.python.org/doc/)
- [Jupyter Notebook](https://docs.jupyter.org/en/latest/)
- [Pandashttps://pandas.pydata.org/docs/user_guide/index.html]()
- [Numpy](https://numpy.org/doc/stable/)
- [Plotly](https://plotly.com/python/)
- [matplotlib](https://matplotlib.org/stable/users/index.html)
- [folium](https://python-visualization.github.io/folium/)

Font: https://thecleverprogrammer.com/2020/05/26/san-francisco-crime-analysis-with-data-science/

---

