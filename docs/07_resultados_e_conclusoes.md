# 7. Resultados e conclusões

Síntese do que foi descoberto com a coleta, a análise exploratória e o treinamento dos
modelos. Os detalhes de cada parte estão nos capítulos 1 a 6.

## 7.1 Resumo em números

| Item | Resultado |
|---|---|
| Base | 46.383 registros de 8 imobiliárias em 12 coletas (mar–ago/2026) → 4.266 imóveis únicos; 3.354 casas e apartamentos |
| Modelo de preço — apartamentos | R² 0,80; erro médio R$ 120 mil; erro percentual mediano 14,8%; 64% das previsões a até ±20% |
| Modelo de preço — casas | R² 0,56; erro médio R$ 291 mil; erro percentual mediano 19,6%; 51% das previsões a até ±20% |
| Melhor candidato individual (validação cruzada) | TabPFN v2, nos dois tipos |
| Variação de preço em 1 mês | Nenhum modelo supera "o preço não muda" |
| Variação de preço em 3 meses | CatBoost supera a referência (−6,7% de erro; R² 0,20) |
| Chance de redução de preço em 3 meses | Gradient Boosting, AUC 0,73 |
| Valorização anual do preço anunciado | Apartamentos +0,89%/ano; casas +0,38%/ano |
| Inflação (IPCA, média de 10 anos) | +4,89%/ano |

## 7.2 Principais descobertas

### 1. A qualidade dos dados pesou mais que a escolha do algoritmo

Os primeiros modelos, com a base bruta, tiveram R² perto de zero. O salto para R² 0,66
veio da limpeza (casas e apartamentos, remoção de outliers), não da troca de algoritmo.
A EDA mostrou por quê: preços digitados em milhões ("R$ 1,35" para uma casa de 420 m²),
áreas de 140 mil m², apartamento com 409 vagas e endereço ausente em 100% dos anúncios.

### 2. A avaliação correta mudou a conclusão sobre o desempenho

A primeira versão do modelo híbrido reportava R² 0,85, mas treinava o modelo com os
próprios dados de teste. Com a avaliação corrigida (teste separado antes de qualquer
tratamento, escolha dos modelos por validação cruzada no treino), o desempenho real é
**R² 0,68**. Mostrar essa correção é parte do resultado: evidencia o cuidado
metodológico e dá um número confiável para quem usar o modelo.

### 3. Apartamentos são muito mais previsíveis que casas

- R² 0,80 (apartamentos) contra 0,56 (casas).
- Correlação entre área e preço: 0,86 contra 0,69.
- Casas misturam padrões muito diferentes (sobrado, geminada, casa em terreno grande), e
  a base não tem variáveis que os distingam.

### 4. O TabPFN é o melhor candidato, mas a vantagem é pequena

O TabPFN v2 venceu a validação cruzada nos dois tipos de imóvel, sem nenhum ajuste de
hiperparâmetros — um resultado relevante para um modelo de fundação recente. No teste
final, porém, o ensemble com e sem TabPFN teve o mesmo desempenho (diferença de 1%), e o
TabPFN leva cerca de 2 minutos por previsão em CPU. Para a aplicação, os modelos de
*gradient boosting* (CatBoost, XGBoost) oferecem a mesma precisão com resposta imediata.

### 5. O preço anunciado quase não muda no curto prazo

96% dos anúncios mantêm o mesmo preço de uma coleta para a outra; quando mudam, a
mudança é em degraus (+4,7% ou −6,0% na mediana). Consequência: prever a variação em 1
mês não supera a regra "não vai mudar". Em 3 meses, os modelos de árvore já encontram
padrão (CatBoost 6,7% melhor que a referência), e é possível estimar quais imóveis têm
mais chance de baixar o preço (AUC 0,73).

### 6. A valorização medida nos anúncios ficou muito abaixo da inflação

Entre março e agosto de 2026, os mesmos imóveis se valorizaram, em média, 0,89% ao ano
(apartamentos) e 0,38% ao ano (casas), contra 4,89% de IPCA médio. As explicações mais
prováveis são a rigidez do preço anunciado, o viés de sobrevivência (os imóveis vendidos
saem da base) e a janela curta de observação. Por isso a projeção mostra a tendência
observada **e** o cenário da inflação.

### 7. A localização importa, mas o dado de localização é pobre

A diferença de preço/m² entre bairros chega a cerca de 70% (Maria Goretti R$ 9.465/m²
contra Pinheirinho R$ 5.490/m², em apartamentos). Sem endereço, o modelo trata todo o
bairro como igual, o que é uma das principais fontes do erro que resta.

## 7.3 Respostas às perguntas do trabalho

| Pergunta | Resposta |
|---|---|
| É possível estimar o valor de um imóvel com dados públicos de anúncios? | Sim, com erro típico de 15% (apartamentos) a 20% (casas). |
| Qual técnica funciona melhor? | Ensemble de *gradient boosting* (CatBoost/XGBoost), com tratamento de dados por tipo de imóvel. O TabPFN é equivalente em precisão, porém lento em CPU. |
| É possível prever a valorização? | Não no horizonte de 1 mês; parcialmente em 3 meses; no longo prazo, apenas por cenários (tendência observada x inflação). |
| Quanto um imóvel pode valer ao longo dos anos? | A interface responde com o valor de hoje e cenários ano a ano, sempre com faixa de incerteza e a comparação com o IPCA. |

## 7.4 Limitações

1. Preço anunciado ≠ preço de venda.
2. Apenas 5 meses de histórico de preços (julho sem coleta).
3. Viés de sobrevivência: imóveis vendidos saem da base.
4. Sem endereço, idade, padrão construtivo ou estado de conservação.
5. Só Chapecó e só casas e apartamentos (terrenos e chácaras ficaram de fora).
6. Erros de cadastro das imobiliárias, que precisam de regras de limpeza.

## 7.5 Trabalhos futuros

1. **Continuar as coletas mensais**: com 12 meses ou mais, a taxa de valorização e os
   modelos de curto prazo ficam muito mais confiáveis.
2. **Corrigir a escala dos preços na coleta** (valores em milhões) e padronizar cidades.
3. **Enriquecer a localização**: geocodificação, distância ao centro e a serviços.
4. **Extrair variáveis do texto e das fotos do anúncio** (padrão de acabamento, piscina,
   condomínio, estado de conservação).
5. **Obter preços de venda reais** (ITBI da prefeitura, cartórios) para comparar com os
   preços anunciados.
6. **Modelar terrenos** separadamente, com preço por m² de terreno.
7. **Conferir a deduplicação à mão** numa amostra, para medir quantos anúncios foram
   agrupados certo ou errado (unidades iguais no mesmo prédio podem virar um só imóvel).
8. **Ajustar os hiperparâmetros** do ensemble final e repetir a avaliação com outras
   divisões treino/teste, para medir a variação do resultado.
9. **Estimar a idade do prédio pelas fotos.** Avaliado e deixado como trabalho futuro
   (capítulo 8): os anúncios coletados não informam o ano de construção, então não há
   gabarito para treinar nem para medir a precisão, e as coletas não guardaram fotos.
   Exige coletar o ano de construção, onde a imobiliária o informar, e várias fotos por
   imóvel.
