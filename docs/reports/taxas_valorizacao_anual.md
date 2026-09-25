# Taxas de valorização anual (projeção de longo prazo)

- Meses de coleta: 2026-03, 2026-04, 2026-05, 2026-06, 2026-08
- Método: variação do preço anunciado dos mesmos imóveis entre coletas (repeat listings), anualizada.
- Bairros puxados para a taxa do tipo por credibilidade: peso do bairro = n / (n + 30).
- Intervalo: percentis 10 e 90 de bootstrap por imóvel (cenários pessimista e otimista).

## Referência: IPCA

- Fonte: IBGE/BCB — IPCA mensal, série SGS 433 (01/09/2016 a 01/08/2026).
- Média anual dos últimos 10 anos: +4.89% (padrão da projeção).
- Acumulado dos últimos 12 meses: +4.22%.
- Não há índice de preços de imóveis (FipeZap) para Chapecó; o IPCA indica
  quanto o imóvel valeria se apenas acompanhasse a inflação.

## Por tipo

| Tipo | Imóveis | Pessimista (%/ano) | Central (%/ano) | Otimista (%/ano) |
|---|---:|---:|---:|---:|
| Apartamento | 2010 | +0.57 | +0.89 | +1.24 |
| Casa | 1458 | +0.01 | +0.38 | +0.75 |

## Por bairro

| Bairro | Tipo | Imóveis | Peso do bairro | Bruta (%/ano) | Pessimista | Central | Otimista |
|---|---|---:|---:|---:|---:|---:|---:|
| sao lucas | Apartamento | 9 | 0.23 | +18.38 | +2.21 | +4.68 | +7.58 |
| lider | Apartamento | 42 | 0.58 | +2.88 | -0.14 | +2.05 | +4.71 |
| maria goretti | Apartamento | 79 | 0.72 | +2.47 | +0.76 | +2.04 | +3.30 |
| pinheirinho | Apartamento | 42 | 0.58 | +2.70 | +0.40 | +1.94 | +3.71 |
| efapi | Apartamento | 79 | 0.72 | +2.30 | +1.04 | +1.91 | +2.84 |
| santo antonio | Apartamento | 24 | 0.44 | +2.93 | +0.54 | +1.79 | +3.14 |
| universitario | Apartamento | 32 | 0.52 | +2.53 | +0.08 | +1.73 | +3.43 |
| jardim america | Apartamento | 6 | 0.17 | +4.94 | +0.84 | +1.56 | +2.57 |
| vila real | Apartamento | 35 | 0.54 | +1.91 | -0.22 | +1.44 | +3.36 |
| palmital | Apartamento | 30 | 0.50 | +1.66 | +0.36 | +1.28 | +2.34 |
| sao cristovao | Apartamento | 78 | 0.72 | +1.42 | +0.39 | +1.27 | +2.27 |
| parque das palmeiras | Apartamento | 6 | 0.17 | +3.18 | +0.64 | +1.27 | +1.85 |
| paraiso | Apartamento | 51 | 0.63 | +1.46 | -0.48 | +1.25 | +3.19 |
| passo dos fortes | Apartamento | 80 | 0.73 | +1.12 | +0.15 | +1.06 | +2.02 |
| bom pastor | Apartamento | 12 | 0.29 | +0.94 | +0.54 | +0.91 | +1.39 |
| seminario | Apartamento | 17 | 0.36 | +0.86 | +0.47 | +0.88 | +1.36 |
| desbravador | Apartamento | 24 | 0.44 | +0.85 | -0.21 | +0.88 | +2.11 |
| jardim italia | Apartamento | 146 | 0.83 | +0.82 | +0.01 | +0.83 | +1.78 |
| engenho braun | Apartamento | 5 | 0.14 | +0.00 | +0.49 | +0.76 | +1.06 |
| jardins lunardi | Apartamento | 5 | 0.14 | +0.00 | +0.49 | +0.76 | +1.06 |
| belvedere | Apartamento | 6 | 0.17 | +0.00 | +0.48 | +0.74 | +1.03 |
| santa paulina | Apartamento | 7 | 0.19 | +0.00 | +0.47 | +0.72 | +1.01 |
| centro | Apartamento | 812 | 0.96 | +0.64 | +0.13 | +0.65 | +1.12 |
| presidente medici | Apartamento | 147 | 0.83 | +0.49 | -0.48 | +0.56 | +1.72 |
| bom retiro | Apartamento | 13 | 0.30 | -0.65 | +0.12 | +0.42 | +0.74 |
| (sem bairro) | Apartamento | 26 | 0.46 | -0.59 | -0.20 | +0.20 | +0.55 |
| bela vista | Apartamento | 25 | 0.45 | -0.66 | -0.28 | +0.18 | +0.58 |
| dom geronimo | Apartamento | 13 | 0.30 | -1.85 | -0.77 | +0.06 | +0.72 |
| esplanada | Apartamento | 38 | 0.56 | -0.68 | -0.38 | +0.01 | +0.36 |
| saic | Apartamento | 15 | 0.33 | -2.34 | -1.21 | -0.20 | +0.70 |
| santa maria | Apartamento | 77 | 0.72 | -1.01 | -1.64 | -0.48 | +0.73 |
| sao cristovao | Casa | 28 | 0.48 | +4.18 | +0.68 | +2.20 | +3.85 |
| maria goretti | Casa | 64 | 0.68 | +3.05 | +0.65 | +2.19 | +3.95 |
| seminario | Casa | 25 | 0.45 | +4.09 | +0.40 | +2.05 | +4.04 |
| desbravador | Casa | 118 | 0.80 | +2.15 | +0.61 | +1.79 | +2.95 |
| araras | Casa | 12 | 0.29 | +5.29 | +0.29 | +1.76 | +3.32 |
| santo antonio | Casa | 30 | 0.50 | +2.40 | +0.11 | +1.38 | +2.98 |
| alvorada | Casa | 6 | 0.17 | +5.69 | -0.62 | +1.25 | +4.10 |
| condominio espelho das aguas | Casa | 5 | 0.14 | +6.50 | +0.23 | +1.23 | +2.34 |
| lider | Casa | 36 | 0.55 | +1.90 | +0.22 | +1.21 | +2.27 |
| engenho braun | Casa | 13 | 0.30 | +2.19 | -0.38 | +0.92 | +2.32 |
| pinheirinho | Casa | 14 | 0.32 | +1.92 | +0.15 | +0.87 | +1.69 |
| vederti | Casa | 19 | 0.39 | +1.63 | +0.13 | +0.86 | +1.65 |
| centro | Casa | 51 | 0.63 | +0.90 | -1.39 | +0.70 | +2.76 |
| jardim italia | Casa | 39 | 0.57 | +0.82 | -0.99 | +0.63 | +2.45 |
| santos dumont | Casa | 31 | 0.51 | +0.59 | +0.12 | +0.49 | +0.92 |
| sao lucas | Casa | 29 | 0.49 | +0.60 | +0.11 | +0.49 | +0.93 |
| vila real | Casa | 24 | 0.44 | +0.58 | -0.40 | +0.47 | +1.48 |
| santa maria | Casa | 50 | 0.62 | +0.45 | -1.55 | +0.42 | +2.56 |
| jardim europa | Casa | 66 | 0.69 | +0.40 | -0.06 | +0.39 | +0.87 |
| vederti ii | Casa | 5 | 0.14 | +0.00 | +0.00 | +0.33 | +0.64 |
| universitario | Casa | 42 | 0.58 | +0.28 | -0.36 | +0.32 | +1.05 |
| boa vista | Casa | 7 | 0.19 | +0.00 | +0.00 | +0.31 | +0.61 |
| eldorado | Casa | 7 | 0.19 | +0.00 | +0.00 | +0.31 | +0.61 |
| walville | Casa | 7 | 0.19 | +0.00 | +0.00 | +0.31 | +0.61 |
| (sem bairro) | Casa | 8 | 0.21 | +0.00 | +0.00 | +0.30 | +0.59 |
| cristo rei | Casa | 8 | 0.21 | +0.00 | +0.00 | +0.30 | +0.59 |
| espelho das aguas | Casa | 13 | 0.30 | +0.00 | +0.00 | +0.26 | +0.52 |
| saic | Casa | 14 | 0.32 | +0.00 | +0.00 | +0.26 | +0.51 |
| jardim america | Casa | 17 | 0.36 | +0.00 | +0.00 | +0.24 | +0.48 |
| autodromo | Casa | 20 | 0.40 | +0.00 | +0.00 | +0.23 | +0.45 |
| bela vista | Casa | 24 | 0.44 | +0.00 | +0.00 | +0.21 | +0.42 |
| palmital | Casa | 32 | 0.52 | +0.00 | +0.00 | +0.18 | +0.36 |
| paraiso | Casa | 93 | 0.76 | +0.08 | -0.83 | +0.15 | +1.17 |
| passo dos fortes | Casa | 82 | 0.73 | -0.07 | -0.51 | +0.05 | +0.67 |
| bom retiro | Casa | 11 | 0.27 | -1.60 | -0.76 | -0.16 | +0.38 |
| parque das palmeiras | Casa | 18 | 0.38 | -1.79 | -1.84 | -0.44 | +0.76 |
| presidente medici | Casa | 101 | 0.77 | -0.83 | -1.39 | -0.56 | +0.29 |
| efapi | Casa | 146 | 0.83 | -0.80 | -1.40 | -0.60 | +0.19 |
| belvedere | Casa | 10 | 0.25 | -3.86 | -1.95 | -0.70 | +0.40 |
| esplanada | Casa | 56 | 0.65 | -3.87 | -3.74 | -2.41 | -1.14 |

## Limitações

- A taxa vem de poucos meses de coleta e é extrapolada para anos.
- Preço anunciado não é preço de venda; imóveis vendidos saem do painel.
- Anúncios raramente mudam de preço, o que tende a subestimar a valorização real.
  Por isso a projeção mostra também o cenário de referência com o IPCA.
