# Benchmark de modelos

- treino: 3104
- teste: 777
- seed: 42

## Métricas

- **MAE**: erro médio absoluto. Quanto menor, melhor.
- **RMSE**: raiz do erro quadrático médio. Penaliza erros grandes. Quanto menor, melhor.
- **R²**: capacidade explicativa do modelo. Quanto mais perto de 1, melhor.
- **MAPE**: erro percentual médio absoluto. Quanto menor, melhor.

## Resultado

| Modelo | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|
| Random Forest | R$ 423.618,02 | R$ 1.140.397,75 | 0.3459 | 412754.27% |
| Gradient Boosting | R$ 434.355,43 | R$ 1.138.352,74 | 0.3482 | 319651.99% |
| Regressão Linear | R$ 24.191.159.639.830.876.657.811.456,00 | R$ 674.321.794.892.669.081.406.668.800,00 | -228712560338843000519846432893043795296256.0000 | 7559737387447148871680.00% |
| Ridge | R$ 144.986.120.670.819.961.960.136.704,00 | R$ 4.041.447.478.371.315.020.906.627.072,00 | -8215414882162035840383020595762482884116480.0000 | 45308162709631247843328.00% |
| Lasso | R$ 261.266.969.464.742.901.691.973.632,00 | R$ 7.282.743.548.414.088.958.982.488.064,00 | -26677532417699902632307819112020511729647616.0000 | 81645927957732150738944.00% |

## Leitura prática

- Melhor modelo por **MAE**: **Random Forest**
- `R²` muito baixo ou negativo indica que o conjunto atual ainda explica pouco da variação de preços.
- Os erros altos sugerem ruído, outliers e categorias pouco estáveis no banco.

## Próximos testes

- remoção de outliers;
- agrupamento mais forte de bairros e tipos;
- validação cruzada;
- ajuste de hiperparâmetros;
- modelos em escala log.