# Данные

## chembl_solubility.csv
| Столбец | Смысл |
|---|---|
| `mol_id` | идентификатор ChEMBL |
| `canonical_smiles` | канонический SMILES (RDKit) |
| `mol_weight`, `logp`, `tpsa` | молекулярная масса, logP, полярная площадь поверхности |
| `h_donors`, `h_acceptors` | доноры и акцепторы водородной связи |
| `rot_bonds`, `rings`, `aromatic_rings` | вращаемые связи, кольца, ароматические кольца |
| `log_solubility` | измеренная растворимость, log(моль/л) — **целевая переменная** |
| `solubility_class` | класс растворимости, **вычислен из `log_solubility`** (low: < −4, medium: от −4 до −2, high: ≥ −2); только для графиков EDA |

## pubchem_solubility.csv
Те же дескрипторы и `log_solubility`, идентификатор — `cid`. Часть молекул совпадает с ChEMBL
(повторные измерения в других лабораториях).
