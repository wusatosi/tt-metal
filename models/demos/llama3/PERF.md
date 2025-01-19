# Llama 3 model performance and accuracy

Performance collected from [demo/demo.py](demo/demo.py) and accuracy collected from [tests/test_llama_accuracy.py](tests/test_llama_accuracy.py). You can generate this table by running these tests with the `lt` tool (tell it to run `accuracy,demo`) and pressing `m` whilst in the results section to export to markdown.

Note that `test_llama_accuracy.py` parses the below to determine expected values.

## Performance

This configuration uses bfp4 MLP FF1+FF3 for all models.

| Model | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) |
|-------|--------|-----------|-----------|---------------|
| Llama3.2-1B | N150 | 79 | 98 | 90.5 |
| Llama3.2-1B | N300 | 81 | 98 | 101.7 |
| Llama3.2-1B | T3K | 81 | 98 | 97.5 |
| Llama3.2-3B | N150 | 85 | 96 | 49.0 |
| Llama3.2-3B | N300 | 88 | 97 | 56.9 |
| Llama3.2-3B | T3K | 88 | 97 | 54.5 |
| Llama3.1-8B | N150 | 86 | 98 | 28.4 |
| Llama3.1-8B | N300 | 84 | 98 | 38.6 |
| Llama3.1-8B | T3K | 84 | 98 | 52.6 |
| Llama3.2-11B | N300 | 86 | 97 | 38.6 |
| Llama3.2-11B | T3K | 84 | 98 | 52.6 |
| Llama3.1-70B | T3K | 95 | 100 | 14.3 |
| Qwen2.5-7B | N300 | 81 | 96 | 37.9 |
| Qwen2.5-72B | T3K | 99 | 100 | 12.8 |

## Accuracy

This configuration uses bfp4 MLP FF1+FF3 only for the Llama-3.1-70B model and the Qwen-2.5-72B model.

| Model | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) |
|-------|--------|-----------|-----------|---------------|
| Llama3.2-1B | N150 | 77 | 96 | 85.8 |
| Llama3.2-1B | N300 | 80 | 98 | 98.6 |
| Llama3.2-1B | T3K | 78 | 98 | 97.2 |
| Llama3.2-3B | N150 | 88 | 98 | 44.1 |
| Llama3.2-3B | N300 | 88 | 98 | 53.9 |
| Llama3.2-3B | T3K | 88 | 98 | 54.8 |
| Llama3.1-8B | N150 | 89 | 98 | 23.5 |
| Llama3.1-8B | N300 | 90 | 98 | 34.1 |
| Llama3.1-8B | T3K | 88 | 97 | 49.9 |
| Llama3.2-11B | N300 | 90 | 97 | 33.8 |
| Llama3.2-11B | T3K | 88 | 97 | 52.6 |
| Llama3.1-70B | T3K | 95 | 100 | 14.5 |
| Qwen2.5-7B | N300 | 81 | 96 | 33.4 |
| Qwen2.5-72B | T3K | 99 | 100 | 12.8 |
