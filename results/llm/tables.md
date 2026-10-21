# LLM experiments: summary tables

## Table 3 rerun (per category MAP)

|  | users | popularity | se1_legacy_k50 | llm_baseline | llm_k12 | llm_k25 | llm_k50 |
|---|---|---|---|---|---|---|---|
| Musical artists | 50 | 0.505 | 0.639 | 0.521 | 0.599 | 0.594 | 0.699 |
| News outlets | 50 | 0.645 | 0.710 | 0.677 | 0.744 | 0.756 | 0.753 |
| Comedians | 50 | 0.545 | 0.600 | 0.578 | 0.640 | 0.642 | 0.662 |
| Politicians | 50 | 0.737 | 0.736 | 0.802 | 0.823 | 0.830 | 0.832 |
| TV stations | 50 | 0.450 | 0.615 | 0.474 | 0.604 | 0.627 | 0.667 |
| Actors | 50 | 0.550 | 0.619 | 0.496 | 0.628 | 0.623 | 0.678 |
| TV shows | 50 | 0.364 | 0.603 | 0.433 | 0.610 | 0.633 | 0.624 |
| Sports teams | 50 | 0.348 | 0.478 | 0.322 | 0.371 | 0.362 | 0.554 |
| Fashion | 50 | 0.332 | 0.435 | 0.335 | 0.380 | 0.369 | 0.471 |
| Journalists | 50 | 0.628 | 0.656 | 0.684 | 0.692 | 0.695 | 0.712 |
| TV hosts | 50 | 0.336 | 0.412 | 0.299 | 0.424 | 0.453 | 0.538 |
| Films | 50 | 0.245 | 0.413 | 0.175 | 0.273 | 0.254 | 0.361 |
| Food chains | 50 | 0.441 | 0.498 | 0.426 | 0.444 | 0.434 | 0.449 |
| Car makers | 50 | 0.432 | 0.409 | 0.411 | 0.450 | 0.485 | 0.551 |
| **Overall** | 700 | 0.468 | 0.559 | 0.474 | 0.549 | 0.554 | 0.611 |

## Table 3 rerun: overall

|  | overall MAP | gain vs LLM baseline | p vs LLM baseline | complete rankings | paper (GPT-4o) |
|---|---|---|---|---|---|
| llm_baseline | 0.4737 |  |  | 0.9286 | 0.4899 |
| llm_k12 | 0.5487 | 0.1584 | 7.391e-14 | 0.9043 | 0.5529 |
| llm_k25 | 0.554 | 0.1695 | 1.071e-14 | 0.8686 | 0.5595 |
| llm_k50 | 0.6107 | 0.2892 | 5.548e-35 | 0.8814 | 0.6004 |

## Order experiment, k = 25 (per category MAP)

|  | users | popularity | original | repeat | shuffled |
|---|---|---|---|---|---|
| Musical artists | 50 | 0.505 | 0.594 | 0.570 | 0.550 |
| News outlets | 50 | 0.645 | 0.756 | 0.758 | 0.735 |
| Comedians | 50 | 0.545 | 0.642 | 0.637 | 0.640 |
| Politicians | 50 | 0.737 | 0.830 | 0.834 | 0.830 |
| TV stations | 50 | 0.450 | 0.627 | 0.634 | 0.594 |
| Actors | 50 | 0.550 | 0.623 | 0.625 | 0.612 |
| TV shows | 50 | 0.364 | 0.633 | 0.635 | 0.575 |
| Sports teams | 50 | 0.348 | 0.362 | 0.363 | 0.353 |
| Fashion | 50 | 0.332 | 0.369 | 0.369 | 0.416 |
| Journalists | 50 | 0.628 | 0.695 | 0.685 | 0.706 |
| TV hosts | 50 | 0.336 | 0.453 | 0.461 | 0.425 |
| Films | 50 | 0.245 | 0.254 | 0.233 | 0.264 |
| Food chains | 50 | 0.441 | 0.434 | 0.442 | 0.436 |
| Car makers | 50 | 0.432 | 0.485 | 0.497 | 0.440 |
| **Overall** | 700 | 0.468 | 0.554 | 0.553 | 0.541 |

## Order experiment, k = 25: agreement

|  | original | repeat | shuffled |
|---|---|---|---|
| MAP | 0.554 | 0.553 | 0.541 |
| p vs original (Wilcoxon) |  | 0.974 | 0.0003738 |
| mean |ΔAP| vs original | 0 | 0.05842 | 0.08473 |
| τ with presented order | 0.3152 | 0.3146 | 0.09086 |
| τ with popularity order | 0.3152 | 0.3146 | 0.1995 |
| τ with original-run output | 1 | 0.7757 | 0.6666 |
| output = presented order | 0 | 0 | 0 |

## Non-personalized baseline: 10 original-order and 10 shuffled calls per category

| order_type | overall MAP (mean over trials) | SD across trials | τ with presented order | τ with popularity order | distinct rankings per category (of 10) |
|---|---|---|---|---|---|
| original | 0.4745 | 0.00442 | 0.5134 | 0.4255 | 9.857 |
| shuffled | 0.4524 | 0.005398 | 0.1332 | 0.3236 | 10 |

## Ranks 81-100 (per category MAP)

|  | users | random | popularity | socialvec | llm_baseline | llm_k12 | llm_k25 | llm_k50 |
|---|---|---|---|---|---|---|---|---|
| Musical artists | 30 | 0.252 | 0.453 | 0.611 | 0.316 | 0.395 | 0.369 | 0.489 |
| News outlets | 30 | 0.253 | 0.267 | 0.396 | 0.268 | 0.532 | 0.546 | 0.533 |
| Comedians | 30 | 0.217 | 0.290 | 0.281 | 0.154 | 0.290 | 0.272 | 0.351 |
| Politicians | 30 | 0.386 | 0.328 | 0.463 | 0.540 | 0.484 | 0.546 | 0.519 |
| TV stations | 30 | 0.191 | 0.149 | 0.444 | 0.236 | 0.353 | 0.463 | 0.666 |
| Actors | 30 | 0.268 | 0.320 | 0.518 | 0.271 | 0.353 | 0.403 | 0.487 |
| TV shows | 30 | 0.213 | 0.276 | 0.407 | 0.334 | 0.485 | 0.456 | 0.570 |
| Sports teams | 30 | 0.252 | 0.269 | 0.516 | 0.222 | 0.302 | 0.315 | 0.426 |
| Journalists | 30 | 0.324 | 0.303 | 0.423 | 0.392 | 0.424 | 0.423 | 0.467 |
| **Overall** | 270 | 0.262 | 0.295 | 0.451 | 0.304 | 0.402 | 0.421 | 0.501 |

Paired Wilcoxon p vs popularity: socialvec 1e-12, llm_baseline 0.567, llm_k12 9.33e-10, llm_k25 2.03e-12, llm_k50 2.57e-23
