---
layout: post
title: "Projetando Preços: Previsão de Valores de Carros Usados usando Regressão"
featured-img:
categories: [Machine Learning, Data Analysis, Python, Linear Regression, Random Forest]
---
Aqui, vamos explorar um dos mercados mais dinâmicos e emocionantes do Brasil: o mercado de anúncios de carros usados. Se você já se perguntou por que os preços dos veículos usados variam tanto e o que influencia essas flutuações, então este é o lugar certo para entender.
O mercado de carros usados no Brasil é imenso e diversificado, com milhões de veículos sendo vendidos e comprados todos os anos. É um setor que atrai tanto compradores em busca de boas ofertas como vendedores buscando maximizar o valor de seus automóveis.

No entanto, com uma ampla variedade de marcas, modelos, idades e condições, estabelecer um preço justo para um carro usado pode ser um desafio tanto para quem vende quanto para quem compra. Além disso, fatores externos, como a economia, a oferta e a demanda, também desempenham papéis cruciais na determinação dos valores.

Aqui, apresentaremos um projeto empolgante baseado em machine learning e regressão que visa desvendar esse intrigante mercado de anúncios de carros usados. Utilizando técnicas avançadas de aprendizado de máquina, vamos desenvolver um modelo capaz de prever os preços dos veículos usados com precisão, levando em consideração diversos fatores que influenciam nesse processo.

Ao longo deste projeto, mergulharemos nas etapas do desenvolvimento, explorando as nuances do machine learning, a importância dos dados de treinamento e como o modelo resultante pode ser uma ferramenta poderosa para compradores e vendedores de carros usados.
O projeto foi oferecido pela empresa Indicium, como um desafio na área de Ciência de Dados, desenvolvido e entregue no prazo de 7 dias.

> *Neste [Notebook](https://github.com/scarletjulia/Lighthouse_Indicium/blob/main/LH_CD_SCARLET_JULIA.ipynb) está toda minha análise e os tratamentos realizados.

Vamos realizar a análise da base dos anúncios de veículos usados, fazer a análise exploratória, responder 3 questões de negócios já definidas, criar e responder outras 3 hipóteses de negócios e criar um modelo preditivo de regressão a fim de prever o target "preço". 
Para iniciar fiz a importação da base de [treino](https://drive.google.com/file/d/1MrhCVFoU5eM32KoTZn0Hq0j7UZbTE5PS/view?usp=drive_link) e de [teste](https://drive.google.com/file/d/1z233Oh3TMsttalFXuI0scTHfNujO0NkM/view?usp=sharing)).

Antes disso, foi necessário implementar essa função a fim de identificar o encoding do arquivo CSV.

{% highlight python %} def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding'] {% endhighlight %}

Após, foi possível realizar a importação das bases.

{% highlight python %} treino = pd.read_csv('/home/bloom/Downloads/cars_train.csv', delimiter='\t', encoding='utf-16')
teste = pd.read_csv('/home/bloom/Downloads/cars_test.csv', delimiter='\t', encoding='utf-16') {% endhighlight %}

### Analisando a base

{% highlight python %} treino.describe() {% endhighlight %}

|          | num_fotos | ano_de_fabricacao | ano_modelo | hodometro   | num_portas | veiculo_alienado | preco         |
| -------- | --------- | ----------------- | ---------- | ----------- | ---------- | ---------------- | ------------- |
| count    | 29407.000 | 29584.000         | 29584.000  | 29584.000   | 29584.000  | 0.0              | 2.958400e+04  |
| mean     | 10.323834 | 2016.758552       | 2017.808985 | 58430.592077| 3.940677   | NaN              | 1.330239e+05  |
| std      | 3.487334  | 4.062422          | 2.673930   | 32561.769309| 0.338360   | NaN              | 8.166287e+04  |
| min      | 8.000000  | 1985.000000       | 1997.000000| 100.000000  | 2.000000   | NaN              | 9.869951e+03  |
| 25%      | 8.000000  | 2015.000000       | 2016.000000| 31214.000000| 4.000000   | NaN              | 7.657177e+04  |
| 50%      | 8.000000  | 2018.000000       | 2018.000000| 57434.000000| 4.000000   | NaN              | 1.143558e+05  |
| 75%      | 14.000000 | 2019.000000       | 2020.000000| 81953.500000| 4.000000   | NaN              | 1.636796e+05  |
| max      | 21.000000 | 2022.000000       | 2023.000000| 390065.000000| 4.000000   | NaN              | 1.359813e+06  |

{% highlight python %} teste.describe() {% endhighlight %}

|          | num_fotos | ano_de_fabricacao | ano_modelo | hodometro   | num_portas | veiculo_alienado |
| -------- | --------- | ----------------- | ---------- | ----------- | ---------- | ---------------- |
| count    | 9802.000  | 9862.000          | 9862.000   | 9862.000    | 9862.000   | 0.0              |
| mean     | 10.323811 | 2016.716893       | 2017.801663| 58237.207057| 3.942507   | NaN              |
| std      | 3.462367  | 4.151105          | 2.679667   | 32487.018991| 0.333749   | NaN              |
| min      | 8.000000  | 1988.000000       | 2007.000000| 100.000000  | 2.000000   | NaN              |
| 25%      | 8.000000  | 2015.000000       | 2016.000000| 31323.250000| 4.000000   | NaN              |
| 50%      | 8.000000  | 2018.000000       | 2018.000000| 56742.000000| 4.000000   | NaN              |
| 75%      | 14.000000 | 2019.000000       | 2020.000000| 81784.000000| 4.000000   | NaN              |
| max      | 21.000000 | 2022.000000       | 2023.000000| 381728.000000| 4.000000   | NaN              |

### Respondendo 3 questões de negócios:

#### a) Qual o melhor estado cadastrado na base de dados para se vender um carro de marca popular e por quê?


![carros_populares_por_estado](https://drive.google.com/uc?export=view&id=1RabD-ZTcA-__0CZqZnhMkDyaZoPLaAZM)


![boxplot_preco_pop_estado](https://drive.google.com/uc?export=view&id=19-A0l1lymTHrSzC0auYv2lY9B0R6Wsbu)

São Paulo possui a maior quantidade de carros populares anunciados para venda comparado a outros estados presentes na base de dados. A alta oferta pode ser um grande atrativo para pessoas interessadas em uma variedade ampla de opções. O preço médio dos carros populares em São Paulo é relativamente alto em comparação com outros estados, o que permite um melhor retorno financeiro. No cálculo do Índice de Desempenho, São Paulo teve o resultado mais alto dentre todos os estados com 0.739091, o que garante boa combinação de oferta (quantidade de carros anunciados) e demanda (preço médio relativamente alto).

#### b) Qual o melhor estado para se comprar uma picape com transmissão automática e por quê?

![boxplot_preco_picape_estado](https://drive.google.com/uc?export=view&id=1-BzcCnqnQFQUiuDbTVIdmBMUwZzM6Gky)


![boxplot_preco_medio_picape_estado](https://drive.google.com/uc?export=view&id=1TUHOlOIxGWISmh28FOa0jpBKfOirEg6I)


![boxplot_mediana_preco_picape_estado](https://drive.google.com/uc?export=view&id=10PljUGL0ibDQGJAtKoPPUJmCkrlo0qJF)

São Paulo foi considerado o melhor estado para compra de picapes automáticas. Com o preço médio aproximado de 188.427,48, fica em posição entre tods os estados porém se destaca potencialmente porque possui oferta significativa com 1712 unidades disponíveis, levando a uma competitivade de preços e variedade de escolha, fatores benéficos para compradores.

#### c) Qual o melhor estado para se comprar carros que ainda estejam dentro da garantia de fábrica e por quê?

![carros_preco_medio_estado](https://drive.google.com/uc?export=view&id=196eA9oiipCukIpcUBmxgIz_Moyz10ins)

Minas Gerais (MG) destaca-se no cenário nacional, posicionando-se entre os 25% dos estados com maior volume de carros que ainda detêm garantia de fábrica. Este indicativo sugere uma preferência ou tendência do mercado mineiro em adquirir carros novos ou recentemente lançados, refletindo possivelmente a confiança dos consumidores na aquisição de veículos com garantias de fábrica, uma vez que essas garantias geralmente são indicativos de confiabilidade e segurança na compra.

#### Hipóteses de negócio

#### a) Influência do Ano de Fabricação no Preço:

Veículos mais novos tendem a ter preços mais altos em comparação com os mais antigos. Especificamente, carros fabricados a partir de 2018 (mediana do ano de fabricação) podem ter um preço significativamente mais alto do que os fabricados antes desse ano. Justificativa: Normalmente, carros mais novos possuem características e tecnologias mais atualizadas, maior eficiência no consumo de combustível, menos desgaste e, em muitos casos, ainda estão sob garantia, o que pode justificar preços mais elevados.

{% highlight python %} X = sm.add_constant(treino['ano_de_fabricacao'])

model = sm.OLS(treino['preco'], X).fit()

print(model.summary()) {% endhighlight %}

OLS Regression Results
==============================================================================
Dep. Variable:                  preco   R-squared:                       0.057
Model:                            OLS   Adj. R-squared:                  0.057
Method:                 Least Squares   F-statistic:                     1795.
Date:                Sun, 23 Jul 2023   Prob (F-statistic):               0.00
Time:                        20:54:57   Log-Likelihood:            -3.7571e+05
No. Observations:               29584   AIC:                         7.514e+05
Df Residuals:                   29582   BIC:                         7.514e+05
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
=====================================================================================
                        coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------------
const             -9.563e+06   2.29e+05    -41.784      0.000      -1e+07   -9.11e+06
ano_de_fabricacao  4807.8364    113.484     42.366      0.000    4585.403    5030.270
==============================================================================
Omnibus:                    15778.592   Durbin-Watson:                   2.012
Prob(Omnibus):                  0.000   Jarque-Bera (JB):           214866.631
Skew:                           2.263   Prob(JB):                         0.00
Kurtosis:                      15.402   Cond. No.                     1.00e+06
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large,  1e+06. This might indicate that there are
strong multicollinearity or other numerical problems.

Resultado: O ano de fabricação tem uma relação positiva e estatisticamente significativa com o preço. No entanto, ele explica apenas 5,7% da variabilidade do preço. Além disso, a análise sugere a presença de outliers e potencial multicolinearidade, o que pode impactar a confiabilidade do modelo.

#### b) Impacto do histórico de Manutenção no Preço:

Veículos que passaram por revisões na concessionária e têm todas as suas revisões dentro da agenda são vendidos a preços mais elevados do que aqueles que não têm um histórico de manutenção claro. Justificativa: Revisões em concessionárias e manutenções programadas geralmente são vistas como indicadores de que o veículo foi bem cuidado. Compradores podem estar dispostos a pagar mais por carros que têm um histórico claro de manutenção, pois isso sugere que o carro pode ter menos problemas no futuro.




