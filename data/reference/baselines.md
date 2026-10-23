# Comparators

## AUC and EER on the six KNAD captures

| Capture | KitNET (reference) AUC | EER | KitNET-style dA AUC | EER | Best variant (oracle selection) | AUC | EER |
|---|---|---|---|---|---|---|---|
| KNAD:Active_Wiretap | 0.268 | 0.776 | 0.322 | 0.734 | LSTM-AR | 0.921 | 0.126 |
| KNAD:Fuzzing | 0.336 | 0.616 | 0.753 | 0.315 | Conv2D-AR | 0.774 | 0.300 |
| KNAD:OS_Scan | 0.950 | 0.088 | 0.958 | 0.087 | LSTM-AR | 0.966 | 0.073 |
| KNAD:SSDP_Flood | 0.996 | 0.008 | 0.996 | 0.008 | MLP | 0.996 | 0.008 |
| KNAD:SSL_Renegotiation | 0.749 | 0.292 | 0.754 | 0.292 | LSTM-TSR | 0.793 | 0.295 |
| KNAD:Video_Injection | 0.721 | 0.376 | 0.528 | 0.507 | MLP | 0.896 | 0.143 |
| **Mean** | 0.670 | 0.359 | 0.719 | 0.324 |  | 0.891 | 0.157 |

## Isolation Forest, ROC-AUC of the fused detector (single run)

| Stream | ROC-AUC |
|---|---|
| KNAD:Active_Wiretap | 0.933 |
| KNAD:Fuzzing | 0.801 |
| KNAD:OS_Scan | 0.963 |
| KNAD:SSDP_Flood | 0.996 |
| KNAD:SSL_Renegotiation | 0.683 |
| KNAD:Video_Injection | 0.830 |
| ACI-IOT-2023 | 0.064 |
| CIC2017 | 0.756 |
| CIC2018 | 0.653 |
| **Mean** | 0.742 |
| **Median** | 0.801 |
