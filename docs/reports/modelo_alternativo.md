# Relatório alternativo de modelos

Modelos testados:
- Regressão Linear Múltipla
- Random Forest
- Gradient Boosting
- Redes Neurais Artificiais (MLP)
- Máquinas de Vetores de Suporte (SVM)

- treino: 3104
- teste: 777
- seed: 42

## Métricas

- **MAE**: erro médio absoluto (menor é melhor).
- **RMSE**: penaliza mais erros grandes (menor é melhor).
- **R²**: capacidade explicativa (maior é melhor).
- **MAPE**: erro percentual médio (menor é melhor).

## Resultado

| Modelo | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|
| Máquinas de Vetores de Suporte (SVM) | R$ 672.439,71 | R$ 1.462.408,41 | -0.0757 | 257251.11% |
| Gradient Boosting | R$ 1.098.028,66 | R$ 4.432.856,53 | -8.8838 | 468067.70% |
| Redes Neurais Artificiais (MLP) | R$ 1.159.268,74 | R$ 1.825.374,74 | -0.6759 | 163.25% |
| Random Forest | R$ 1.778.585,26 | R$ 16.626.939,64 | -138.0530 | 563673.56% |
| Regressão Linear Múltipla | R$ 2.058.126,37 | R$ 3.773.429,78 | -6.1619 | 1309703.43% |

## Melhor candidato

- Melhor por MAE: **Máquinas de Vetores de Suporte (SVM)**
- O comando `train-alt` salva esse melhor modelo como padrão alternativo.
