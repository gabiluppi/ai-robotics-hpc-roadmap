
# Part 1: The Basics of Machine Learning

### Chapter 1: The Machine Learning Scenario

O primeiro capítulo introduz sobre conceitos de machine learning.

> *É o campo de estudo que possibilita aos computadores a habilidade de aprender sem explicitamente programá-los*
> ***Arthur Samuel, 1959***

> *Alega-se que um programa de computador aprende pela experiência ==E== em relação a algum tipo de tarefa ==T== e alguma medida de desempenho ==P== se o seu desempenho em ==T==, conforme medido por ==P==, melhora com a experiência ==E==.*
> ***Tom Mitchell, 1997***

##### Quando utilizar o aprendizado de máquina?
- A solução de um problema requer muitos ajustes finos ou extensas listas de regras.
- O problema é complexo e não existe uma boa solução utilizando a abordagem tradicional
- O sistema requer uma boa adaptabilidade para novos dados
- Deseja-se entender um problema complexo ou grandes quantidades de dados



#### Tipos de Aprendizado:

---
##### Aprendizado Supervisionado

==O conjunto de treinamento oferecido ao algoritmo inclui as soluções desejadas, chamadas de ***rótulos*** ou ***labels***.==

Os tipos de problemas mais comuns para este tipo de aprendizado são **classificação** (e.g. marcar um e-mail como spam ou ham) e **regressão** (e.g. prever o valor de uma casa baseado nas suas características).

Alguns dos algoritmos mais importantes deste método são:
- K-Nearest Neighbors (KNN)
- Regressão linear
- Regressão logística
- Máquinas de vetores de suporte (SVM)
- Árvores de decisão
- Random Florests
- Redes neurais* (maioria)

---
##### Aprendizado Não Supervisionado

==O conjunto de treinamento não é rotulado.==

Alguns dos problemas e seus algoritmos mais importantes são:
- ==Clusterização==
	- K-Means
	- DBSCAN
	- Análise de cluster hierárquica
- ==Detecção de anomalias e novidades==
	- One-class SVM
	- Floresta de isolamento
- ==Visualização e redução de dimensionalidade==
	- Análise de Componentes Principais
	- Kernel ACP
	- LLE (Locally Linear Embedding)
	- t-SNE (Distributed Stochastic Neighbor Embedding)
- ==Aprendizado de regras por associação==
	- Apriori
	- Eclat

---
##### Aprendizado Semissupervisionado

==O conjunto de treinamento tem apenas uma parte das instâncias rotuladas (geralmente poucas).==

A maior parte dos algoritmos de aprendizado semissupervisionado são combinações de algoritmos supervisionados e não supervisionados:
- Rede neural de crenças profundas (DBN)
- Máquina restritas de Boltzmann (RBM)

---
##### Aprendizado por Reforço

==O sistema de aprendizado (agente) pode assistir o ambiente, selecionar e executar ações e obter recompensas (ou penalidades) em troca.== Ele deve aprender sozinho qual a melhor estratégia (política) para obter o maior número de recompensas. Uma política define o que o agente deve escolher em determinada situação.

---

### Aprendizado em batch vs online

##### Aprendizado em batch
- Incapaz de aprender de forma incremental
- Para se adaptar a novos dados deve ser gerado uma nova versão do modelo com o conjunto de treino atualizado
- Demanda muito tempo e recurso computacional

##### Aprendizado online
- Possível treinar de maneira incremental
- O sistema aprende com os dados em tempo real
- Pode ser utilizado para treinar com grandes conjuntos de dados que não cabem na memória principal


Demonstração do processo de aprendizado online.
![[IMG-20250623-WA0005~2.jpg]]

Demonstração do uso de aprendizado online para um conjunto de dados grande.
![[IMG-20250623-WA0006~2.jpg]]

---
### Aprendizado Baseado em Instâncias vs Aprendizado Baseado em Modelo

Classificar os sistemas de aprendizado por meio da ***generalização***.

##### Aprendizado baseado em Instâncias

O sistema emprega uma medida de ***similaridade*** entre dados novos e dados memorizados durante o treinamento.


##### Aprendizado baseado em modelo

O sistema aprende a tomar decisões a partir de um modelo explícito do ambiente baseado nas duas interações. Esse modelo captura a relação entre estados, ações e resultados.

---

### Principais Desafios do Aprendizado de Máquina

Os dois principais problemas são ***algoritmos ruins*** e ***dados ruins***.

No artigo ***The Unreasonable Effectiveness of Data  (2009)***, Peter Norvig traz a ideia de que os dados são mais importantes do que os algoritmos ao se tratar de problemas complexos.

##### Dados ruins
- Quantidade insuficiente de dados de treinamento
- Dados de treinamento não representativos
- Dados cheios de ruídos, erros e outliers
- Características irrelevantes
- Viés de amostragem

##### Algoritmos ruins
- Sobreajuste dos dados de treinamento (overfitting)
- Subajuste dos dados de treinamento (underfitting)

---

### Capítulo 2: Projeto de Aprendizado de Máquina Ponta a Ponta

Neste capítulo é introduzido o contexto de projeto para uma empresa. O objetivo do projeto é construir um modelo que faça a previsão de preço em regiões do estado da Califórnia.

Respondendo os questionamentos iniciais sobre o modelo, será utilizado:
- Aprendizado supervisionado
- Tarefa de regressão univariada (estou tentando prever um valor de saída a partir das características de entrada)
- Aprendizado em batch (o dataset não necessita que seja atualizado em tempo real)


  #### Medida de desempenho
  
A *raiz do erro quadrático médio* (RMSE) é uma métrica amplamente utilizada para avaliar o desempenho de modelos de regressão. Ela quantifica o erro médio de previsões, atribuindo penalidade maior a erros de maior magnitude:
$$ RMSE(X, h) = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (h(x^{(i)}) - y^{(i)})^2} $$

Quando o conjunto de dados contém muitos *outliers*, pode ser mais apropriado utilizar o *erro médio absoluto* (MAE), também conhecido como desvio médio absoluto, pois ele é menos sensível a valores extremos.
$$ MAE(X, h) = \frac{1}{n} \sum_{i=1}^{n} |h(x^{(i)}) - y^{(i)}| $$
[^1]: **Notações** 
- $n$ é o número de instâncias no conjunto de dados 
- $x^{(i)}$ representa o vetor de características da $i$-ésima instância (excluindo o rótulo), e $y^{(i)}$ é o rótulo correspondente 
- $X$ é a matriz contendo todas as características das instâncias 
- $h$ é a função preditiva (ou hipótese) do modelo, que gera a previsão $\hat{y}^{(i)} = h(x^{(i)})$ para a instância $x^{(i)}$ 
- $RMSE(X, h)$ e $MAE(X, h)$ são as funções de custo avaliadas sobre o conjunto de exemplos utilizando a hipótese $h$

O RMSE está relacionado à *norma euclidiana* (ou norma $L_2$), denotada por $\|\mathbf{x}\|_2$, enquanto o MAE está associado à *norma Manhattan* (ou norma $L_1$), denotada por $\|\mathbf{x}\|_1$. A norma $L_1$ mede a soma dos valores absolutos dos elementos do vetor, enquanto a norma $L_2$ mede a raiz quadrada da soma dos quadrados dos elementos.

De forma geral, a norma $L_k$ de um vetor $\mathbf{v}$ com $n$ elementos é definida por:
$$
\|\mathbf{v}\|_k = \left( \sum_{i=1}^n |v_i|^k \right)^{1/k}
$$

Para valores maiores de $k$, a norma $L_k$ torna-se cada vez mais sensível aos maiores valores absolutos do vetor, aproximando-se da norma $L_\infty$, que corresponde ao maior valor absoluto entre os elementos.
$$ \|\mathbf{v}\|_\infty = \max_{i} |v_i| $$

Assim, métricas baseadas em normas de ordem superior, como RMSE, são mais sensíveis a grandes erros do que métricas baseadas em ordem inferior, como o MAE. Por outro lado, quando os resíduos seguem uma distribuição aproximadamente normal (gaussiana) e outliers são raros, o RMSE é geralmente preferido, pois é estimador de máxima verossimilhança nesse contexto.


#### Verifique as hipóteses

Uma boa prática é enumerar e verificar as hipóteses feitas, facilitando a identificação de problemas logo no início. Por exemplo, verifique quais tipos de dados o consumidor do modelo espera receber.