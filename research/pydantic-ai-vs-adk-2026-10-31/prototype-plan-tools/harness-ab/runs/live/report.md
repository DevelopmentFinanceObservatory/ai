## Decision rule, B against A

| Criterion                      | A        | B arm    | B        | B/A   | 95% interval   | Limit | Verdict |
| ------------------------------ | -------- | -------- | -------- | ----- | -------------- | ----- | ------- |
| mean input tokens              | 45,740.2 | B-http   | 36,020.4 | 0.787 | 0.684 to 0.898 | 1.02  | pass    |
| mean input tokens              | 45,740.2 | B-inproc | 42,265.4 | 0.924 | 0.788 to 1.068 | 1.02  | pass    |
| mean output + thinking tokens  | 644.8    | B-http   | 574.9    | 0.891 | 0.680 to 1.274 | 1.02  | pass    |
| mean output + thinking tokens  | 644.8    | B-inproc | 663.4    | 1.029 | 0.790 to 1.299 | 1.02  | fail    |
| median wall s to accepted plan | 14.2     | B-http   | 14.1     | 0.989 | 0.731 to 1.514 | 1.05  | pass    |
| median wall s to accepted plan | 14.2     | B-inproc | 13.4     | 0.946 | 0.770 to 1.347 | 1.05  | pass    |
| p90 wall s to accepted plan    | 36.1     | B-http   | 34.4     | 0.953 | 0.637 to 1.614 | 1.10  | pass    |
| p90 wall s to accepted plan    | 36.1     | B-inproc | 37.2     | 1.030 | 0.521 to 1.427 | 1.10  | pass    |
| accepted                       | 29/30    | B-http   | 27/30    |       |                | >= A  | fail    |
| accepted                       | 29/30    | B-inproc | 29/30    |       |                | >= A  | pass    |
| right                          | 23/30    | B-http   | 20/30    |       |                | >= A  | fail    |
| right                          | 23/30    | B-inproc | 21/30    |       |                | >= A  | fail    |

## Per arm

| Arm      | Runs | Accepted | Right | Kind pass | Years | Model calls | Input tokens       | Cached             | Output + thinking | Wall s median | Wall s p90 | Budget closed |
| -------- | ---- | -------- | ----- | --------- | ----- | ----------- | ------------------ | ------------------ | ----------------- | ------------- | ---------- | ------------- |
| A        | 30   | 29       | 23    | 23        | 29    | 2.67        | 45,740 (sd 20,773) | 41,093 (sd 16,355) | 645 (sd 611)      | 14.56         | 37.12      | 0             |
| B-http   | 30   | 27       | 20    | 20        | 27    | 2.37        | 36,020 (sd 15,394) | 32,543 (sd 13,024) | 575 (sd 418)      | 16.19         | 64.35      | 3             |
| B-inproc | 30   | 29       | 21    | 21        | 29    | 2.67        | 42,265 (sd 16,401) | 37,807 (sd 13,321) | 663 (sd 567)      | 13.95         | 37.40      | 1             |

## Per task

| Task                                                | Arm      | Endings                                              | Right | Model calls | Input tokens | Output + thinking | Wall s         |
| --------------------------------------------------- | -------- | ---------------------------------------------------- | ----- | ----------- | ------------ | ----------------- | -------------- |
| `series-gdp-ghana`                                  | A        | accepted accepted accepted                           | 3/3   | 2 2 2       | 32,883       | 220               | 8.2 26.0 12.9  |
| `series-gdp-ghana`                                  | B-http   | accepted accepted accepted                           | 3/3   | 2 2 2       | 30,843       | 640               | 14.1 16.3 9.2  |
| `series-gdp-ghana`                                  | B-inproc | accepted accepted accepted                           | 3/3   | 2 2 2       | 30,777       | 226               | 11.4 27.3 5.4  |
| `series-oda-germany-kenya`                          | A        | accepted accepted accepted                           | 3/3   | 2 2 2       | 32,949       | 222               | 7.5 15.6 5.9   |
| `series-oda-germany-kenya`                          | B-http   | accepted accepted accepted                           | 3/3   | 2 2 2       | 30,843       | 222               | 7.5 13.2 6.0   |
| `series-oda-germany-kenya`                          | B-inproc | accepted accepted accepted                           | 3/3   | 2 2 2       | 30,843       | 229               | 10.7 11.5 6.2  |
| `aggregate-health-population-oda-germany-by-sector` | A        | accepted asked_user accepted                         | 0/3   | 3 6 3       | 74,033       | 1,715             | 58.9 37.1 36.1 |
| `aggregate-health-population-oda-germany-by-sector` | B-http   | accepted accepted accepted                           | 0/3   | 2 2 3       | 38,114       | 804               | 64.3 17.9 38.5 |
| `aggregate-health-population-oda-germany-by-sector` | B-inproc | accepted accepted accepted                           | 0/3   | 2 4 2       | 44,477       | 1,087             | 24.0 38.5 30.7 |
| `aggregate-total-oda-kenya`                         | A        | accepted accepted accepted                           | 3/3   | 2 2 3       | 38,845       | 326               | 7.2 20.3 11.6  |
| `aggregate-total-oda-kenya`                         | B-http   | accepted turn_model_time_spent accepted              | 2/3   | 2 1 3       | 26,120       | 345               | 7.2 90.0 32.7  |
| `aggregate-total-oda-kenya`                         | B-inproc | accepted accepted accepted                           | 3/3   | 3 3 3       | 47,759       | 566               | 9.8 19.3 10.1  |
| `growth-total-oda-ghana`                            | A        | accepted accepted accepted                           | 3/3   | 2 2 3       | 38,927       | 441               | 5.6 79.2 14.2  |
| `growth-total-oda-ghana`                            | B-http   | turn_model_time_spent accepted turn_model_time_spent | 0/3   | 2 3 2       | 25,793       | 337               | 90.1 58.2 90.1 |
| `growth-total-oda-ghana`                            | B-inproc | turn_model_time_spent accepted accepted              | 0/3   | 2 2 2       | 25,506       | 228               | 90.1 9.8 5.9   |
| `sum-oda-oof-ethiopia`                              | A        | accepted accepted accepted                           | 0/3   | 3 3 3       | 51,554       | 534               | 10.9 15.2 11.3 |
| `sum-oda-oof-ethiopia`                              | B-http   | accepted accepted accepted                           | 0/3   | 3 3 3       | 48,880       | 653               | 34.4 18.2 16.1 |
| `sum-oda-oof-ethiopia`                              | B-inproc | accepted accepted accepted                           | 0/3   | 2 3 3       | 42,989       | 527               | 13.4 10.9 12.5 |
| `rank-top-donors-malawi`                            | A        | accepted accepted accepted                           | 3/3   | 3 3 3       | 50,978       | 889               | 15.6 22.2 14.9 |
| `rank-top-donors-malawi`                            | B-http   | accepted accepted accepted                           | 3/3   | 3 3 2       | 42,244       | 757               | 10.9 12.2 7.7  |
| `rank-top-donors-malawi`                            | B-inproc | accepted accepted accepted                           | 3/3   | 3 3 3       | 47,873       | 963               | 37.4 15.9 21.3 |
| `share-two-donors-of-kenya-oda`                     | A        | accepted accepted accepted                           | 3/3   | 2 6 4       | 70,547       | 1,406             | 7.5 44.6 27.9  |
| `share-two-donors-of-kenya-oda`                     | B-http   | accepted accepted accepted                           | 3/3   | 5 3 2       | 54,742       | 1,107             | 27.2 13.6 8.8  |
| `share-two-donors-of-kenya-oda`                     | B-inproc | accepted accepted accepted                           | 3/3   | 5 4 5       | 78,453       | 1,862             | 69.5 16.7 29.2 |
| `quotient-oda-per-person-ethiopia`                  | A        | accepted accepted accepted                           | 3/3   | 2 2 2       | 33,131       | 354               | 28.0 9.2 13.0  |
| `quotient-oda-per-person-ethiopia`                  | B-http   | accepted accepted accepted                           | 3/3   | 2 3 2       | 36,699       | 527               | 20.6 20.3 10.0 |
| `quotient-oda-per-person-ethiopia`                  | B-inproc | accepted accepted accepted                           | 3/3   | 2 2 3       | 36,688       | 498               | 18.1 8.7 14.5  |
| `join-gdp-debt-service-ghana`                       | A        | accepted accepted accepted                           | 2/3   | 2 2 2       | 33,555       | 340               | 7.4 26.7 6.9   |
| `join-gdp-debt-service-ghana`                       | B-http   | accepted accepted accepted                           | 3/3   | 2 2 1       | 25,926       | 356               | 7.2 9.8 28.4   |
| `join-gdp-debt-service-ghana`                       | B-inproc | accepted accepted accepted                           | 3/3   | 2 3 2       | 37,288       | 449               | 8.3 37.2 5.9   |

## Tool calls

| Arm      | Tool         | Calls | Median s, call to response | Median MCP overhead ms |
| -------- | ------------ | ----- | -------------------------- | ---------------------- |
| A        | get_skill    | 33    | 0.0036                     |                        |
| A        | propose_plan | 46    | 0.0097                     |                        |
| B-http   | get_skill    | 31    | 0.0015                     |                        |
| B-http   | propose_plan | 38    | 0.0190                     | 10.95                  |
| B-http   | start_plan   | 30    | harness call               | 5.05                   |
| B-inproc | get_skill    | 33    | 0.0015                     |                        |
| B-inproc | propose_plan | 47    | 0.0121                     | 2.10                   |
| B-inproc | start_plan   | 30    | harness call               | 1.00                   |
