# Planner entry eval: pai against baseline

372 runs over 62 tasks on gemini-3.8-flash; baseline at afbf84ff; pai at afbf84ff. Rates count runs; tasks with a first-proposal refusal are scored only on whether they ended in that refusal and an ask. Runs that errored or missed their recording are left out of every figure; attempts count every entry started, retries included. Unasked units count dataset requirements stating a unit in first proposals of tasks whose question and brief name none. Tokens are means over runs whose every model call finished, input including cached.

## Per arm

| Arm      | Runs scored (attempts) | Errored | Missed the recording | First proposal accepted | Kind pass, runs | Kind pass, tasks | Refusal tasks ending right | Unasked units | Model calls | Input tokens (cached)           | Output tokens | Thinking tokens | Wall s: mean, p90, max | Closed by the turn budget |
| -------- | ---------------------- | ------- | -------------------- | ----------------------- | --------------- | ---------------- | -------------------------- | ------------- | ----------- | ------------------------------- | ------------- | --------------- | ---------------------- | ------------------------- |
| baseline | 186 (186)              | 0       | 0                    | 129/177 (73%)           | 169/177 (95%)   | 56/59 (95%)      | 6/9 (67%)                  | 0 of 217      | 2.56        | 39,547 (35,092); 2 runs unknown | 614           | 99              | 11.4, 19.7, 90.1       | 2                         |
| pai      | 186 (186)              | 0       | 0                    | 127/177 (72%)           | 173/177 (98%)   | 58/59 (98%)      | 8/9 (89%)                  | 0 of 217      | 2.40        | 36,868 (32,965)                 | 562           | 254             | 11.1, 19.6, 70.1       | 0                         |

## Paired against the baseline

Per task, the arm's rate over its repetitions minus the baseline's.

| Outcome                 | Tasks | Better / worse / same | Mean difference [95%] | Sign test p |
| ----------------------- | ----- | --------------------- | --------------------- | ----------- |
| First proposal accepted | 59    | 8 / 11 / 40           | -1.1% [-6.8%, +5.1%]  | 0.65        |
| Kind pass               | 59    | 4 / 1 / 54            | +2.3% [-0.6%, +5.6%]  | 0.38        |

| Cost per run    | Tasks | Mean difference [95%]       |
| --------------- | ----- | --------------------------- |
| Model calls     | 62    | -0.2 [-0.3, -0.0]           |
| Input tokens    | 62    | -2,617.7 [-5,800.9, +147.2] |
| Output tokens   | 62    | -54.2 [-143.4, +13.3]       |
| Thinking tokens | 62    | +156.2 [-69.2, +434.4]      |
| Wall s          | 62    | -0.4 [-2.5, +2.0]           |

## Switch triggers

| Trigger                                                        | Fired | Numbers                          |
| -------------------------------------------------------------- | ----- | -------------------------------- |
| First-proposal acceptance down 5% or more, interval below zero | no    | -1.1% [-6.8%, +5.1%]             |
| Kind pass down 2 or more tasks                                 | no    | 58 against 56 of 59 shared tasks |
| p90 wall time up 10 s or more                                  | no    | 19.6 s against 19.7 s            |
| More runs closed by the turn budget than the baseline          | no    | 0 of 186 runs against 2 of 186   |

Budget alarm: the turn budget closed 0 of pai's runs and 2 of baseline's, by turn_model_time_spent 2.

## How entries ended

| Arm      | accepted | asked_user | turn_model_time_spent |
| -------- | -------- | ---------- | --------------------- |
| baseline | 167      | 17         | 2                     |
| pai      | 171      | 15         | 0                     |

## Why first proposals were refused

| Arm      | Codes                                                                                                 |
| -------- | ----------------------------------------------------------------------------------------------------- |
| baseline | invalid_plan rule 30, invalid_plan shape 11, combination_unasked 5, asks_unaccounted 1, no proposal 1 |
| pai      | invalid_plan rule 31, invalid_plan shape 10, combination_unasked 6, asks_unaccounted 3                |

## Per task

| Task                                              | Kind        | baseline: first ok, kind pass | pai: first ok, kind pass |
| ------------------------------------------------- | ----------- | ----------------------------- | ------------------------ |
| series-debt-stock-kenya-ghana                     | dataset     | 3/3, 3/3                      | 3/3, 3/3                 |
| series-france-oda-africa                          | dataset     | 3/3, 3/3                      | 3/3, 3/3                 |
| series-gdp-ghana                                  | dataset     | 3/3, 3/3                      | 3/3, 3/3                 |
| series-oda-by-sector-kenya                        | dataset     | 3/3, 3/3                      | 3/3, 3/3                 |
| series-oda-germany-kenya                          | dataset     | 3/3, 3/3                      | 3/3, 3/3                 |
| series-population-three                           | dataset     | 3/3, 3/3                      | 3/3, 3/3                 |
| series-zambia-debt-to-china                       | dataset     | 3/3, 3/3                      | 3/3, 3/3                 |
| aggregate-average-gdp-ghana                       | aggregate   | 3/3, 3/3                      | 3/3, 3/3                 |
| aggregate-average-oda-kenya                       | aggregate   | 1/3, 3/3                      | 0/3, 3/3                 |
| aggregate-combined-oda-kenya-uganda               | aggregate   | 3/3, 3/3                      | 3/3, 3/3                 |
| aggregate-cumulative-debt-flows-china-kenya       | aggregate   | 3/3, 3/3                      | 3/3, 3/3                 |
| aggregate-cumulative-oda-g7-africa                | aggregate   | 3/3, 3/3                      | 2/3, 3/3                 |
| aggregate-health-oda-kenya-ghana                  | aggregate   | 2/3, 3/3                      | 3/3, 3/3                 |
| aggregate-health-population-oda-all-donors        | aggregate   | 3/3, 3/3                      | 3/3, 3/3                 |
| aggregate-health-population-oda-germany-by-sector | aggregate   | 0/3, 1/3                      | 1/3, 0/3                 |
| aggregate-health-population-oda-kenya-by-sector   | aggregate   | 0/3, 3/3                      | 1/3, 3/3                 |
| aggregate-malaria-oda-all-recipients              | aggregate   | 3/3, 3/3                      | 3/3, 3/3                 |
| aggregate-total-debt-service-zambia               | aggregate   | 3/3, 3/3                      | 3/3, 3/3                 |
| aggregate-total-oda-kenya                         | aggregate   | 3/3, 3/3                      | 2/3, 3/3                 |
| aggregate-two-donors-yearly                       | aggregate   | 3/3, 3/3                      | 3/3, 3/3                 |
| growth-absolute-debt-zambia                       | growth      | 2/3, 3/3                      | 3/3, 3/3                 |
| growth-decline-german-oda                         | growth      | 2/3, 2/3                      | 2/3, 3/3                 |
| growth-gdp-kenya                                  | growth      | 3/3, 3/3                      | 3/3, 3/3                 |
| growth-population-nigeria                         | growth      | 3/3, 3/3                      | 3/3, 3/3                 |
| growth-total-oda-ghana                            | growth      | 3/3, 3/3                      | 3/3, 3/3                 |
| growth-uk-oda-kenya                               | growth      | 3/3, 3/3                      | 3/3, 3/3                 |
| comparison-debt-service-climate-finance-kenya     | comparison  | 0/3, 3/3                      | 0/3, 3/3                 |
| comparison-debt-stock-gdp-ghana                   | comparison  | 3/3, 3/3                      | 3/3, 3/3                 |
| comparison-oda-kenya-uganda                       | comparison  | 3/3, 3/3                      | 3/3, 3/3                 |
| comparison-us-uk-oda-kenya                        | comparison  | 3/3, 3/3                      | 3/3, 3/3                 |
| sum-bilateral-plus-imputed-france-guatemala       | sum_of      | 0/3, 3/3                      | 2/3, 3/3                 |
| sum-bilateral-plus-imputed-japan-indonesia        | sum_of      | 0/3, 3/3                      | 0/3, 3/3                 |
| sum-net-flows-aid-debt-ghana                      | sum_of      | ending 1/3                    | ending 3/3               |
| sum-oda-debt-disbursements-kenya                  | sum_of      | ending 2/3                    | ending 3/3               |
| sum-oda-fdi-ethiopia                              | sum_of      | ending 3/3                    | ending 2/3               |
| sum-then-quotient-germany-tanzania                | sum_of      | 0/3, 3/3                      | 0/3, 3/3                 |
| rank-bottom-five-african-oda-2021                 | rank        | 0/3, 3/3                      | 0/3, 3/3                 |
| rank-donor-each-year-kenya                        | rank        | 0/3, 3/3                      | 0/3, 3/3                 |
| rank-external-debt-african-countries              | rank        | 0/3, 3/3                      | 0/3, 3/3                 |
| rank-fastest-gdp-growth                           | rank        | 3/3, 3/3                      | 3/3, 3/3                 |
| rank-sectors-kenya                                | rank        | 0/3, 3/3                      | 1/3, 3/3                 |
| rank-top-donors-guatemala                         | rank        | 1/3, 1/3                      | 3/3, 3/3                 |
| rank-top-donors-guatemala-since-2020              | rank        | 2/3, 3/3                      | 2/3, 3/3                 |
| rank-top-gdp-2022                                 | rank        | 3/3, 3/3                      | 2/3, 3/3                 |
| share-germany-of-oda-ethiopia                     | share_of    | 3/3, 3/3                      | 3/3, 3/3                 |
| share-health-oda-kenya                            | share_of    | 3/3, 3/3                      | 2/3, 3/3                 |
| share-nigeria-world-population                    | share_of    | 3/3, 3/3                      | 3/3, 3/3                 |
| share-two-donors-of-kenya-oda                     | share_of    | 0/3, 3/3                      | 1/3, 3/3                 |
| quotient-debt-service-per-gdp-ghana               | quotient_of | 2/3, 3/3                      | 2/3, 3/3                 |
| quotient-external-debt-gdp-kenya-nigeria          | quotient_of | 3/3, 3/3                      | 2/3, 3/3                 |
| quotient-gdp-per-person-guatemala                 | quotient_of | 3/3, 3/3                      | 3/3, 3/3                 |
| quotient-keyed-oda-per-person-three               | quotient_of | 3/3, 3/3                      | 3/3, 3/3                 |
| quotient-oda-per-person-ethiopia                  | quotient_of | 3/3, 3/3                      | 2/3, 3/3                 |
| quotient-oda-per-person-france                    | quotient_of | 3/3, 3/3                      | 3/3, 3/3                 |
| quotient-oda-per-person-malawi                    | quotient_of | 3/3, 3/3                      | 2/3, 3/3                 |
| join-gdp-debt-service-ghana                       | join        | 3/3, 3/3                      | 3/3, 3/3                 |
| join-malaria-domestic-and-aid-kenya               | join        | 3/3, 1/3                      | 1/3, 2/3                 |
| join-oda-debt-service-zambia                      | join        | 3/3, 3/3                      | 3/3, 3/3                 |
| join-population-gdp-kenya                         | join        | 3/3, 3/3                      | 3/3, 3/3                 |
| filter-donors-above-10b-2021                      | filter      | 1/3, 2/3                      | 0/3, 3/3                 |
| filter-donors-above-200m-kenya                    | filter      | 1/3, 3/3                      | 1/3, 3/3                 |
| filter-gdp-above-trillion-2022                    | filter      | 1/3, 3/3                      | 0/3, 3/3                 |
