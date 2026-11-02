# Demographics-only LLM ranking (ranks 81-100), descriptions by gpt-4o

|  | users | random | popularity | socialvec | llm_baseline | llm_demographics | llm_k12 | llm_k25 | llm_k50 |
|---|---|---|---|---|---|---|---|---|---|
| Musical artists | 30.000 | 0.252 | 0.453 | 0.611 | 0.316 | 0.398 | 0.395 | 0.369 | 0.489 |
| News outlets | 30.000 | 0.253 | 0.267 | 0.396 | 0.268 | 0.318 | 0.532 | 0.546 | 0.533 |
| Comedians | 30.000 | 0.217 | 0.290 | 0.281 | 0.154 | 0.220 | 0.290 | 0.272 | 0.351 |
| Politicians | 30.000 | 0.386 | 0.328 | 0.463 | 0.540 | 0.452 | 0.484 | 0.546 | 0.519 |
| TV stations | 30.000 | 0.191 | 0.149 | 0.444 | 0.236 | 0.290 | 0.353 | 0.463 | 0.666 |
| Actors | 30.000 | 0.268 | 0.320 | 0.518 | 0.271 | 0.369 | 0.353 | 0.403 | 0.487 |
| TV shows | 30.000 | 0.213 | 0.276 | 0.407 | 0.334 | 0.436 | 0.485 | 0.456 | 0.570 |
| Sports teams | 30.000 | 0.252 | 0.269 | 0.516 | 0.222 | 0.324 | 0.302 | 0.315 | 0.426 |
| Journalists | 30.000 | 0.324 | 0.303 | 0.423 | 0.392 | 0.357 | 0.424 | 0.423 | 0.467 |
| **Overall** | 270.000 | 0.262 | 0.295 | 0.451 | 0.304 | 0.352 | 0.402 | 0.421 | 0.501 |

Paired Wilcoxon p: llm_demographics vs popularity 0.000113, llm_demographics vs llm_baseline 0.127, llm_demographics vs llm_k12 0.00942, llm_demographics vs llm_k50 1.72e-15, llm_demographics vs socialvec 9.33e-06
