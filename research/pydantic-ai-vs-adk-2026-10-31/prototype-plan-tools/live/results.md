## smoke-\* (1 runs)

| Task               | Accepted | Rounds   | Refusal codes before acceptance | Right (kinds, years) | Run finished | Tool calls | Input tokens | Output + thinking |
| ------------------ | -------- | -------- | ------------------------------- | -------------------- | ------------ | ---------- | ------------ | ----------------- |
| `series-gdp-ghana` | 1/1      | 1        | none                            | 1/1 (1, 1)           | 1/1          | 4          | -            | -                 |
| all                | 1/1      | mean 1.0 |                                 | 1/1 (1, 1)           | 1/1          | 4          | -            | -                 |

## gemini-\* (20 runs)

| Task                                                | Accepted | Rounds   | Refusal codes before acceptance                                                                                       | Right (kinds, years) | Run finished | Tool calls | Input tokens | Output + thinking |
| --------------------------------------------------- | -------- | -------- | --------------------------------------------------------------------------------------------------------------------- | -------------------- | ------------ | ---------- | ------------ | ----------------- |
| `aggregate-health-population-oda-germany-by-sector` | 2/2      | 1 1      | none                                                                                                                  | 2/2 (2, 2)           | 2/2          | 17         | 891,424      | 21,132            |
| `aggregate-total-oda-kenya`                         | 2/2      | 3 5      | `referral_pending`, `source_not_offered`, `translation:pinned_source_not_found`, `translation:requirement_unmeetable` | 2/2 (2, 2)           | 2/2          | 14         | 829,578      | 33,263            |
| `growth-total-oda-ghana`                            | 2/2      | 4 4      | `referral_pending`, `translation:pinned_source_not_found`, `translation:requirement_unmeetable`                       | 2/2 (2, 2)           | 2/2          | 10         | 563,306      | 27,460            |
| `join-gdp-debt-service-ghana`                       | 2/2      | 1 1      | none                                                                                                                  | 1/2 (1, 2)           | 2/2          | 9          | 469,257      | 14,408            |
| `quotient-oda-per-person-ethiopia`                  | 2/2      | 1 1      | none                                                                                                                  | 0/2 (0, 2)           | 2/2          | 9          | 449,297      | 10,280            |
| `rank-top-donors-malawi`                            | 2/2      | 2 1      | `invalid_plan`                                                                                                        | 2/2 (2, 2)           | 2/2          | 8          | 484,736      | 24,071            |
| `series-gdp-ghana`                                  | 2/2      | 1 1      | none                                                                                                                  | 2/2 (2, 2)           | 2/2          | 8          | 374,648      | 7,188             |
| `series-oda-germany-kenya`                          | 2/2      | 1 1      | none                                                                                                                  | 2/2 (2, 2)           | 2/2          | 8          | 401,700      | 9,845             |
| `share-two-donors-of-kenya-oda`                     | 2/2      | 1 1      | none                                                                                                                  | 2/2 (2, 2)           | 2/2          | 8          | 427,016      | 24,157            |
| `sum-oda-oof-ethiopia`                              | 2/2      | 1 1      | none                                                                                                                  | 0/2 (0, 2)           | 2/2          | 10         | 499,102      | 20,068            |
| all                                                 | 20/20    | mean 1.6 |                                                                                                                       | 15/20 (15, 20)       | 20/20        | 10         | 539,006      | 19,187            |

## baseline (30 runs)

| Task                                                | Accepted | Rounds   | Refusal codes before acceptance        | Right (kinds, years) | Run finished | Tool calls | Input tokens | Output + thinking |
| --------------------------------------------------- | -------- | -------- | -------------------------------------- | -------------------- | ------------ | ---------- | ------------ | ----------------- |
| `series-gdp-ghana`                                  | 3/3      | 1 1 1    | none                                   | 3/3 (3, 3)           | -            | 2          | 32,883       | 226               |
| `aggregate-total-oda-kenya`                         | 3/3      | 1 2 1    | `measure_fixed_by_pin`                 | 3/3 (3, 3)           | -            | 2          | 38,981       | 328               |
| `series-oda-germany-kenya`                          | 3/3      | 1 1 1    | none                                   | 3/3 (3, 3)           | -            | 2          | 32,949       | 210               |
| `aggregate-health-population-oda-germany-by-sector` | 2/3      | - 3 5    | `combination_unasked`, `invalid_plan`  | 0/3 (0, 2)           | -            | 5          | 86,559       | 1,933             |
| `growth-total-oda-ghana`                            | 3/3      | 2 1 2    | `invalid_plan`, `measure_fixed_by_pin` | 3/3 (3, 3)           | -            | 3          | 44,767       | 561               |
| `rank-top-donors-malawi`                            | 3/3      | 2 2 2    | `invalid_plan`                         | 3/3 (3, 3)           | -            | 3          | 50,995       | 925               |
| `quotient-oda-per-person-ethiopia`                  | 3/3      | 1 2 1    | `invalid_plan`                         | 3/3 (3, 3)           | -            | 2          | 39,043       | 535               |
| `join-gdp-debt-service-ghana`                       | 3/3      | 1 1 1    | none                                   | 3/3 (3, 3)           | -            | 2          | 33,555       | 351               |
| `sum-oda-oof-ethiopia`                              | 3/3      | 1 1 1    | none                                   | 0/3 (0, 3)           | -            | 3          | 45,797       | 533               |
| `share-two-donors-of-kenya-oda`                     | 3/3      | 2 2 2    | `invalid_plan`                         | 3/3 (3, 3)           | -            | 3          | 51,271       | 933               |
| all                                                 | 29/30    | mean 1.6 |                                        | 24/30 (24, 29)       | -            | 3          | 45,680       | 654               |

## claude-sonnet-\* (10 runs)

| Task                                                | Accepted | Rounds   | Refusal codes before acceptance                                                     | Right (kinds, years) | Run finished | Tool calls | Input tokens | Output + thinking |
| --------------------------------------------------- | -------- | -------- | ----------------------------------------------------------------------------------- | -------------------- | ------------ | ---------- | ------------ | ----------------- |
| `aggregate-health-population-oda-germany-by-sector` | 1/1      | 1        | none                                                                                | 0/1 (1, 0)           | 1/1          | 9          | -            | -                 |
| `aggregate-total-oda-kenya`                         | 1/1      | 4        | `asks_unaccounted`, `invalid_plan`, `translation:pinned_source_not_found`           | 1/1 (1, 1)           | 1/1          | 9          | -            | -                 |
| `growth-total-oda-ghana`                            | 1/1      | 3        | `referral_pending`, `translation:requirement_unmeetable`                            | 1/1 (1, 1)           | 1/1          | 7          | -            | -                 |
| `join-gdp-debt-service-ghana`                       | 1/1      | 1        | none                                                                                | 1/1 (1, 1)           | 1/1          | 8          | -            | -                 |
| `quotient-oda-per-person-ethiopia`                  | 1/1      | 2        | `asks_unaccounted`                                                                  | 0/1 (0, 1)           | 1/1          | 9          | -            | -                 |
| `rank-top-donors-malawi`                            | 1/1      | 4        | `invalid_plan`, `period_reduction_unstated`, `translation:requirement_needs_repair` | 1/1 (1, 1)           | 1/1          | 9          | -            | -                 |
| `series-gdp-ghana`                                  | 1/1      | 3        | `translation:no_data`, `translation:pinned_source_not_found`                        | 1/1 (1, 1)           | 1/1          | 9          | -            | -                 |
| `series-oda-germany-kenya`                          | 1/1      | 1        | none                                                                                | 1/1 (1, 1)           | 1/1          | 10         | -            | -                 |
| `share-two-donors-of-kenya-oda`                     | 1/1      | 3        | `invalid_plan`, `translation:requirement_needs_repair`                              | 0/1 (0, 1)           | 1/1          | 7          | -            | -                 |
| `sum-oda-oof-ethiopia`                              | 1/1      | 5        | `combination_unasked`, `measure_fixed_by_pin`                                       | 0/1 (0, 1)           | 1/1          | 14         | -            | -                 |
| all                                                 | 10/10    | mean 2.7 |                                                                                     | 6/10 (7, 9)          | 10/10        | 9          | -            | -                 |

## claude-haiku-\*-v2 (10 runs)

| Task                                                | Accepted | Rounds   | Refusal codes before acceptance                                                                                            | Right (kinds, years) | Run finished | Tool calls | Input tokens | Output + thinking |
| --------------------------------------------------- | -------- | -------- | -------------------------------------------------------------------------------------------------------------------------- | -------------------- | ------------ | ---------- | ------------ | ----------------- |
| `aggregate-health-population-oda-germany-by-sector` | 1/1      | 2        | `invalid_plan`                                                                                                             | 1/1 (1, 1)           | 1/1          | 7          | -            | -                 |
| `aggregate-total-oda-kenya`                         | 1/1      | 2        | `invalid_plan`                                                                                                             | 1/1 (1, 1)           | 1/1          | 7          | -            | -                 |
| `growth-total-oda-ghana`                            | 1/1      | 2        | `translation:pinned_source_not_found`                                                                                      | 1/1 (1, 1)           | 1/1          | 7          | -            | -                 |
| `join-gdp-debt-service-ghana`                       | 1/1      | 1        | none                                                                                                                       | 0/1 (0, 1)           | 1/1          | 14         | -            | -                 |
| `quotient-oda-per-person-ethiopia`                  | 1/1      | 1        | none                                                                                                                       | 1/1 (1, 1)           | 1/1          | 8          | -            | -                 |
| `rank-top-donors-malawi`                            | 1/1      | 1        | none                                                                                                                       | 1/1 (1, 1)           | 1/1          | 7          | -            | -                 |
| `series-gdp-ghana`                                  | 1/1      | 4        | `invalid_plan`, `referral_pending`, `translation:requirement_unmeetable`                                                   | 1/1 (1, 1)           | 1/1          | 10         | -            | -                 |
| `series-oda-germany-kenya`                          | 1/1      | 1        | none                                                                                                                       | 1/1 (1, 1)           | 1/1          | 10         | -            | -                 |
| `share-two-donors-of-kenya-oda`                     | 0/1      | -        | `invalid_plan`, `source_not_offered`, `translation:pinned_source_not_found`                                                | 0/1 (0, 0)           | 0/1          | 10         | -            | -                 |
| `sum-oda-oof-ethiopia`                              | 0/1      | -        | `combination_unasked`, `invalid_plan`, `measure_fixed_by_pin`, `source_not_offered`, `translation:pinned_source_not_found` | 0/1 (0, 0)           | 0/1          | 14         | -            | -                 |
| all                                                 | 8/10     | mean 1.8 |                                                                                                                            | 7/10 (7, 8)          | 8/10         | 9          | -            | -                 |

## claude-haiku-\*-r1 (6 runs)

| Task                          | Accepted | Rounds   | Refusal codes before acceptance                                                                    | Right (kinds, years) | Run finished | Tool calls | Input tokens | Output + thinking |
| ----------------------------- | -------- | -------- | -------------------------------------------------------------------------------------------------- | -------------------- | ------------ | ---------- | ------------ | ----------------- |
| `aggregate-total-oda-kenya`   | 1/1      | 3        | `invalid_plan`                                                                                     | 1/1 (1, 1)           | 1/1          | 8          | -            | -                 |
| `growth-total-oda-ghana`      | 0/1      | -        | `invalid_plan`                                                                                     | 0/1 (0, 0)           | 0/1          | 10         | -            | -                 |
| `join-gdp-debt-service-ghana` | 1/1      | 2        | `invalid_plan`                                                                                     | 0/1 (0, 1)           | 1/1          | 9          | -            | -                 |
| `rank-top-donors-malawi`      | 0/1      | -        | `invalid_plan`                                                                                     | 0/1 (0, 0)           | 0/1          | 12         | -            | -                 |
| `series-gdp-ghana`            | 1/1      | 5        | `invalid_plan`, `source_not_offered`, `translation:no_data`, `translation:pinned_source_not_found` | 1/1 (1, 1)           | 1/1          | 12         | -            | -                 |
| `series-oda-germany-kenya`    | 0/1      | -        | `invalid_plan`                                                                                     | 0/1 (0, 0)           | 0/1          | 9          | -            | -                 |
| all                           | 3/6      | mean 3.3 |                                                                                                    | 2/6 (2, 3)           | 3/6          | 10         | -            | -                 |

## luna-\* (10 runs)

| Task                                                | Accepted | Rounds | Refusal codes before acceptance | Right (kinds, years) | Run finished | Tool calls | Input tokens | Output + thinking |
| --------------------------------------------------- | -------- | ------ | ------------------------------- | -------------------- | ------------ | ---------- | ------------ | ----------------- |
| `aggregate-health-population-oda-germany-by-sector` | 0/1      | -      | none                            | 0/1 (0, 0)           | 0/1          | 22         | 714,301      | 6,163             |
| `aggregate-total-oda-kenya`                         | 0/1      | -      | none                            | 0/1 (0, 0)           | 0/1          | 18         | 579,748      | 4,582             |
| `growth-total-oda-ghana`                            | 0/1      | -      | none                            | 0/1 (0, 0)           | 0/1          | 24         | 805,202      | 5,876             |
| `join-gdp-debt-service-ghana`                       | 0/1      | -      | none                            | 0/1 (0, 0)           | 0/1          | 31         | 935,689      | 6,315             |
| `quotient-oda-per-person-ethiopia`                  | 0/1      | -      | none                            | 0/1 (0, 0)           | 0/1          | 30         | 1,027,066    | 6,977             |
| `rank-top-donors-malawi`                            | 0/1      | -      | none                            | 0/1 (0, 0)           | 0/1          | 17         | 519,045      | 3,400             |
| `series-gdp-ghana`                                  | 0/1      | -      | none                            | 0/1 (0, 0)           | 0/1          | 23         | 694,564      | 5,038             |
| `series-oda-germany-kenya`                          | 0/1      | -      | none                            | 0/1 (0, 0)           | 0/1          | 19         | 620,013      | 5,410             |
| `share-two-donors-of-kenya-oda`                     | 0/1      | -      | none                            | 0/1 (0, 0)           | 0/1          | 37         | 1,981,339    | 19,894            |
| `sum-oda-oof-ethiopia`                              | 0/1      | -      | none                            | 0/1 (0, 0)           | 0/1          | 21         | 665,840      | 4,796             |
| all                                                 | 0/10     | mean - |                                 | 0/10 (0, 0)          | 0/10         | 24         | 854,281      | 6,845             |
