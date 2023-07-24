---
layout: post
title: "Projetando Preços: Previsão de Valores de Carros Usados usando Regressão"
featured-img:
categories: [Machine Learning, Data Analysis, Python, Linear Regression, Random Forest]
---
Neste projeto, o objetivo é analisar o mercado de anúncios de carros usados no Brasil, que é dinâmico e diversificado, com milhões de veículos sendo vendidos e comprados anualmente. O desafio é entender as flutuações de preços e os fatores que as influenciam, como a economia, oferta e demanda.

Para enfrentar esse desafio, o projeto utiliza técnicas avançadas de machine learning e regressão para desenvolver um modelo preciso capaz de prever os preços dos carros usados. Ao longo do desenvolvimento, são exploradas as etapas do processo de aprendizado de máquina, destacando a importância dos dados de treinamento. O resultado final é uma poderosa ferramenta que pode beneficiar tanto compradores quanto vendedores de carros usados.

O projeto foi oferecido pela empresa Indicium como um desafio na área de Ciência de Dados e foi concluído dentro de um prazo de 7 dias.

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

![impacto_rev_concess_preco](https://drive.google.com/uc?export=view&id=1olnNMFrR6lGHHrSRBCzBFbq4Nni3IeGY)


![impacto_rev_agenda_preco](https://drive.google.com/uc?export=view&id=1kLDYar3I7AI0t3soqNkKx8pcp3jqiTgV)

Análise de revisões em concessionária:

Mediana (Valor Central)

Sem Revisão: 105.42k
Com Revisão: 136.21k

Há uma diferença notável na mediana dos preços entre os dois grupos. Os carros com revisões em concessionárias possuem uma mediana 30.79k maior, indicando que o valor central dos carros revisados em concessionárias tende a ser mais elevado.

Quartis (Dispersão dos Preços)

O Q1 (primeiro quartil) para carros com revisão é 20.46k mais alto, sugerindo que os 25% mais baratos dos carros com revisão em concessionária ainda são vendidos a preços mais elevados em comparação com os sem essa revisão.
O Q3 (terceiro quartil) para carros com revisão é 40.56k maior. Isso significa que, até mesmo nos segmentos de preços mais altos, os carros com revisões em concessionárias ainda são negociados a valores superiores.

Limites (Extremos dos Preços)

O preço mínimo para carros com revisão é maior por 3.71k.
O preço máximo para carros com revisão é menor por 219.7k. Este é um ponto interessante e poderia indicar que carros sem revisão em concessionárias, mas talvez com características especiais ou raridades, alcançam valores muito elevados no mercado.

Análise de Revisão dentro da agenda: Mediana (Valor Central)

Sem Revisão: 109.73k
Com Revisão: 134.59k

Observando as medianas, os carros com revisões feitas dentro da agenda têm uma mediana de preço 22.86k mais alta que os carros sem essas revisões. Isso indica que o valor central dos carros com revisões programadas é mais alto. O Q1 (primeiro quartil) para carros com revisões é 15.83k maior, indicando que 25% dos carros com revisões têm um preço acima de 89.73k, enquanto aqueles sem revisão têm preços acima de 73.90k. Curiosamente, o preço máximo para carros com revisão é significativamente menor (619.913k a menos). No entanto, é importante lembrar que outliers ou valores extremos podem ser influenciados por outros fatores, não apenas pela revisão.

#### c) Relação entre o Número de Fotos e o Preço:

Veículos que possuem um número de fotos acima da média (mais do que 10, que é a média fornecida) são vendidos a preços mais elevados do que os que possuem menos fotos. Justificativa: Um número maior de fotos pode indicar que o vendedor está mais engajado em vender o carro e está fornecendo detalhes abrangentes sobre o veículo. Isso pode transmitir mais confiança ao comprador potencial, permitindo que o vendedor defina um preço mais alto. Adicionalmente, veículos com características ou condições especiais (como interiores de alta qualidade ou customizações) podem ter mais fotos para destacar esses atributos.

![impacto_num_fotos_preco](https://drive.google.com/uc?export=view&id=1OM3G6yMjuQz78w3YfDkE8onE_0YbtU-X)

Análise Relação entre o número de fotos e o preço do veículo:

    Mediana: Veículos com fotos abaixo ou igual à média têm uma mediana de preço mais alta (117.74k) em comparação com aqueles com fotos acima da média (107.15k).
    Dispersion: A dispersão de preços (Q3 - Q1) é ligeiramente maior para veículos com fotos abaixo ou igual à média.
    Valores extremos: Enquanto veículos com mais fotos têm um valor máximo mais elevado (1.359813M vs 1.15436M), veículos com menos fotos apresentam uma mediana de preço mais elevada.

Contrário à hipótese inicial, veículos com um número de fotos abaixo ou igual à média tendem a ter preços medianos ligeiramente mais elevados do que aqueles com fotos acima da média. Embora possa parecer surpreendente à primeira vista, essa descoberta sugere que simplesmente ter mais fotos não é um forte indicativo de um preço mais alto.

### Correlação

{% highlight python %} df_numeric = df.query("Origem == 'treino'").select_dtypes(exclude=['object'])

corr_matrix = df_numeric.corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

corr_masked = corr_matrix.mask(mask)

fig = ff.create_annotated_heatmap(
    reversescale=True,
    z=corr_masked.to_numpy(),
    x=list(corr_masked.columns),
    y=list(corr_masked.index),
    annotation_text=corr_masked.round(2).to_numpy(),
    colorscale='Viridis',
    showscale=True
)
fig.show() {% endhighlight %}

![corr](https://drive.google.com/uc?export=view&id=1ARrM3iiuIADEmNKEHPFSjzcXfgfvc0vj)

### Modelo Baseline

{% highlight python %} df_numeric = df.select_dtypes(exclude=['object'])

# Selecionando as colunas para avaliar
features = list(df_numeric.columns)
features.remove('preco')  # Removendo 'preco' da lista

# Preparando os dados
X = df.query("Origem == 'treino'")[features]
y = df.query("Origem == 'treino'")['preco'] {% endhighlight %}


{% highlight python %} %%time
models = [
    ('Linear Regression', LinearRegression()),
    ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=0)),
    ('Gradient Boosting', GradientBoostingRegressor(random_state=0)),
    ('SVR', SVR()),
    ('Decision Tree', DecisionTreeRegressor()),
    ('Ridge Regression', Ridge()),
    ('Lasso Regression', Lasso()),
    ('ElasticNet Regression', ElasticNet()),
    ('KNN', KNeighborsRegressor()),
    ('XGBoost', XGBRegressor(random_state=0)),
    ('LightGBM', LGBMRegressor(random_state=0)),
    ('AdaBoost', AdaBoostRegressor(random_state=0))  
]

scoring = {
    'rmse': make_scorer(lambda y, y_pred: mean_squared_error(y, y_pred, squared=False)),
    'mae': make_scorer(mean_absolute_error),
    'r2': make_scorer(r2_score)
}

results = {}
for name, model in models:
    results.update(evaluate_model(X, y, model, name))

for metric, values in results.items():
    print(f"{metric}: {values.mean()} +/- {values.std()}") {% endhighlight %}

Abaixo estão os resultados de avaliação para vários modelos de regressão (baseline):

Linear Regression

    Fit Time: 0.054 +/- 0.019 seconds
    Score Time: 0.008 +/- 0.004 seconds
    Test RMSE: 71510.43 +/- 1629.10
    Test MAE: 51897.31 +/- 194.46
    Test R2: 0.233 +/- 0.013

Random Forest

    Fit Time: 6.956 +/- 0.137 seconds
    Score Time: 0.149 +/- 0.007 seconds
    Test RMSE: 69727.78 +/- 1230.40
    Test MAE: 49928.68 +/- 625.62
    Test R2: 0.270 +/- 0.027

Gradient Boosting

    Fit Time: 1.801 +/- 0.051 seconds
    Score Time: 0.012 +/- 0.003 seconds
    Test RMSE: 65766.41 +/- 1582.93
    Test MAE: 47402.39 +/- 191.97
    Test R2: 0.351 +/- 0.014

SVR (Support Vector Regression)

    Fit Time: 24.792 +/- 0.469 seconds
    Score Time: 5.769 +/- 0.089 seconds
    Test RMSE: 83346.04 +/- 1822.11
    Test MAE: 57001.75 +/- 445.60
    Test R2: -0.042 +/- 0.003

Decision Tree

    Fit Time: 0.122 +/- 0.006 seconds
    Score Time: 0.004 +/- 0.0001 seconds
    Test RMSE: 91477.49 +/- 1273.23
    Test MAE: 64093.99 +/- 527.74
    Test R2: -0.257 +/- 0.040

Ridge Regression

    Fit Time: 0.036 +/- 0.004 seconds
    Score Time: 0.005 +/- 0.001 seconds
    Test RMSE: 71510.35 +/- 1628.71
    Test MAE: 51897.23 +/- 194.02
    Test R2: 0.233 +/- 0.013

Lasso Regression

    Fit Time: 0.054 +/- 0.007 seconds
    Score Time: 0.005 +/- 0.001 seconds
    Test RMSE: 71510.43 +/- 1628.94
    Test MAE: 51897.26 +/- 194.19
    Test R2: 0.233 +/- 0.013

ElasticNet Regression

    Fit Time: 0.044 +/- 0.002 seconds
    Score Time: 0.005 +/- 0.001 seconds
    Test RMSE: 74292.49 +/- 1690.93
    Test MAE: 54207.42 +/- 201.98
    Test R2: 0.172 +/- 0.006

KNN (K-Nearest Neighbors)

    Fit Time: 0.028 +/- 0.005 seconds
    Score Time: 0.151 +/- 0.044 seconds
    Test RMSE: 82438.66 +/- 1836.55
    Test MAE: 59781.15 +/- 612.73
    Test R2: -0.020 +/- 0.010

XGBoost

    Fit Time: 0.864 +/- 0.070 seconds
    Score Time: 0.008 +/- 0.003 seconds
    Test RMSE: 65131.68 +/- 1083.52
    Test MAE: 46442.43 +/- 300.36
    Test R2: 0.363 +/- 0.016

LightGBM

    Fit Time: 0.115 +/- 0.005 seconds
    Score Time: 0.011 +/- 0.001 seconds
    Test RMSE: 64112.48 +/- 1297.37
    Test MAE: 45886.71 +/- 254.28
    Test R2: 0.383 +/- 0.017

AdaBoost

    Fit Time: 0.729 +/- 0.147 seconds
    Score Time: 0.018 +/- 0.003 seconds
    Test RMSE: 119815.81 +/- 21529.08
    Test MAE: 104369.18 +/- 21372.15
    Test R2: -1.241 +/- 0.796

### Resultados do modelo:

{% highlight python %} %%time
features = list(df_numeric.columns)
features.remove('preco')  

X = df_dummies.query("Origem == 'treino'")[features]
y = df_dummies.query("Origem == 'treino'")['preco']

y = np.log(y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.columns = [col.replace(" ", "_") for col in X_train.columns]
X_val.columns = [col.replace(" ", "_") for col in X_val.columns]
X_train = X_train.fillna(0)
y_train = y_train.fillna(0)
X_val = X_val.fillna(0)

model = LGBMRegressor(random_state=0, force_col_wise=True)
model.fit(X_train, y_train.ravel())

y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)

y_pred_train = np.exp(y_pred_train)
y_pred_val = np.exp(y_pred_val)

y_train = np.exp(y_train)
y_val = np.exp(y_val)

print("Train Set Evaluation:")
print(f"RMSE: {mean_squared_error(y_train, y_pred_train, squared=False)}")
print(f"MAE: {mean_absolute_error(y_train, y_pred_train)}")
print(f"R2: {r2_score(y_train, y_pred_train)}")

print("Validation Set Evaluation:")
print(f"RMSE: {mean_squared_error(y_val, y_pred_val, squared=False)}")
print(f"MAE: {mean_absolute_error(y_val, y_pred_val)}")
print(f"R2: {r2_score(y_val, y_pred_val)}")

std_dev = np.std(y_pred_val)

print(f'Standard Deviation of predictions: {std_dev}') {% endhighlight %}






