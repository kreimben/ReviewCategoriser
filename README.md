# Performance Comparison: PEFT Models with and without Quantization

This repository contains two CSV files that compare the performance of various PEFT (Parameter-Efficient Fine-Tuning) models on a specific task, with and without quantization applied.

## File 1: peft_only_result.csv

This file contains performance metrics for PEFT models without quantization. The columns in the file are:

1. Model name
2. Accuracy
3. Precision
4. Recall
5. F1 score
6. Inference time (in seconds)
7. Inference speed (examples per second)
8. Inference time per example (in seconds)

The models included in this file are:

- `peft/bert-base-uncased`
- `peft/bert-large-uncased`
- `peft/roberta-base`
- `peft/roberta-large`
- `peft/distilbert-base-uncased`

## File 2: quantization_result.csv

This file contains performance metrics for PEFT models with quantization applied. The columns are the same as in the first file. However, only three models are included:

- `peft/bert-base-uncased`
- `peft/bert-large-uncased`
- `peft/distilbert-base-uncased`

## Comparison Table

| Model                        | Quantization | Accuracy | Precision | Recall   | F1 Score | Inference Time (s) | Inference Speed (examples/s) | Inference Time per Example (s) |
|------------------------------|--------------|----------|-----------|----------|----------|-------------------|-----------------------------|--------------------------------|
| peft/bert-base-uncased       | No           | 0.921    | 0.926230  | 0.913131 | 0.919634 | 263.371240        | 3.796922                    | 0.263371                       |
| peft/bert-base-uncased       | Yes          | 0.916    | 0.905738  | 0.920833 | 0.913223 | 37.729872         | 26.504198                   | 0.037730                       |
| peft/bert-large-uncased      | No           | 0.923    | 0.920082  | 0.921971 | 0.921026 | 910.154669        | 1.098714                    | 0.910155                       |
| peft/bert-large-uncased      | Yes          | 0.924    | 0.928279  | 0.917004 | 0.922607 | 66.757128         | 14.979674                   | 0.066757                       |
| peft/roberta-base            | No           | 0.937    | 0.913934  | 0.955032 | 0.934031 | 259.067528        | 3.859997                    | 0.259068                       |
| peft/roberta-large           | No           | 0.965    | 0.948770  | 0.978858 | 0.963580 | 885.460542        | 1.129356                    | 0.885461                       |
| peft/distilbert-base-uncased | No           | 0.861    | 0.844262  | 0.867368 | 0.855659 | 130.017804        | 7.691254                    | 0.130018                       |
| peft/distilbert-base-uncased | Yes          | 0.856    | 0.827869  | 0.870690 | 0.848739 | 18.674401         | 53.549241                   | 0.018674                       |

## Analysis

1. The performance metrics (accuracy, precision, recall, and F1 score) for the models with quantization are slightly lower than their counterparts without quantization. This suggests that quantization may have a small negative impact on the model's performance.

2. The inference times for the models with quantization are significantly lower than those without quantization. For example, the inference time for `peft/bert-base-uncased` is reduced from 263.371240 seconds to 37.729872 seconds after quantization. This indicates that quantization greatly improves the inference speed of the models.

3. Consequently, the inference speed (examples per second) for the quantized models is much higher than their non-quantized counterparts. For instance, `peft/bert-base-uncased` processes 3.796922 examples per second without quantization, but after quantization, it can process 26.504198 examples per second.

## Conclusion

The comparison table and analysis demonstrate that applying quantization to PEFT models can significantly improve their inference speed while having a minimal impact on their performance metrics. This repository provides a valuable resource for researchers and practitioners interested in optimizing the performance of PEFT models for various applications.