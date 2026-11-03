## Decision rule, B against A

| Criterion                      | A        | B arm    | B        | B/A   | 95% interval   | Limit | Verdict |
| ------------------------------ | -------- | -------- | -------- | ----- | -------------- | ----- | ------- |
| mean input tokens              | 45,561.6 | B-served | 48,560.3 | 1.066 | 0.960 to 1.176 | 1.02  | fail    |
| mean output + thinking tokens  | 648.6    | B-served | 735.1    | 1.133 | 0.959 to 1.261 | 1.02  | fail    |
| median wall s to accepted plan | 12.2     | B-served | 9.8      | 0.808 | 0.661 to 1.311 | 1.05  | pass    |
| p90 wall s to accepted plan    | 27.6     | B-served | 30.8     | 1.118 | 0.464 to 3.234 | 1.10  | fail    |
| accepted                       | 29/30    | B-served | 29/30    |       |                | >= A  | pass    |
| right                          | 21/30    | B-served | 21/30    |       |                | >= A  | pass    |

## Per arm

| Arm      | Runs | Accepted | Right | Kind pass | Years | Model calls | Input tokens       | Cached             | Output + thinking | Wall s median | Wall s p90 | Budget closed |
| -------- | ---- | -------- | ----- | --------- | ----- | ----------- | ------------------ | ------------------ | ----------------- | ------------- | ---------- | ------------- |
| A        | 30   | 29       | 21    | 21        | 29    | 2.67        | 45,562 (sd 15,384) | 41,093 (sd 12,363) | 649 (sd 471)      | 12.26         | 28.04      | 0             |
| B-served | 30   | 29       | 21    | 21        | 29    | 2.87        | 48,560 (sd 23,276) | 43,000 (sd 17,937) | 735 (sd 685)      | 9.87          | 37.49      | 0             |

## Per task

| Task                                                | Arm      | Endings                      | Right | Model calls | Input tokens | Output + thinking | Wall s         |
| --------------------------------------------------- | -------- | ---------------------------- | ----- | ----------- | ------------ | ----------------- | -------------- |
| `series-gdp-ghana`                                  | A        | accepted accepted accepted   | 3/3   | 2 2 2       | 32,883       | 222               | 14.4 5.8 5.9   |
| `series-gdp-ghana`                                  | B-served | accepted accepted accepted   | 3/3   | 2 2 2       | 32,063       | 223               | 7.3 8.4 6.1    |
| `series-oda-germany-kenya`                          | A        | accepted accepted accepted   | 3/3   | 2 2 2       | 32,949       | 221               | 5.9 9.1 5.9    |
| `series-oda-germany-kenya`                          | B-served | accepted accepted accepted   | 3/3   | 2 2 2       | 32,129       | 213               | 5.7 5.2 6.5    |
| `aggregate-health-population-oda-germany-by-sector` | A        | asked_user accepted accepted | 0/3   | 5 3 3       | 66,565       | 1,441             | 28.0 14.0 43.7 |
| `aggregate-health-population-oda-germany-by-sector` | B-served | accepted asked_user accepted | 0/3   | 2 6 5       | 80,563       | 1,771             | 8.2 43.5 24.1  |
| `aggregate-total-oda-kenya`                         | A        | accepted accepted accepted   | 3/3   | 3 3 2       | 44,827       | 467               | 10.2 27.6 6.4  |
| `aggregate-total-oda-kenya`                         | B-served | accepted accepted accepted   | 3/3   | 3 3 2       | 43,875       | 479               | 9.9 14.6 27.6  |
| `growth-total-oda-ghana`                            | A        | accepted accepted accepted   | 0/3   | 2 3 2       | 38,906       | 394               | 5.9 11.7 13.8  |
| `growth-total-oda-ghana`                            | B-served | accepted accepted accepted   | 0/3   | 3 2 2       | 37,968       | 392               | 9.8 4.6 7.4    |
| `sum-oda-oof-ethiopia`                              | A        | accepted accepted accepted   | 0/3   | 3 3 3       | 52,103       | 660               | 11.2 12.2 18.6 |
| `sum-oda-oof-ethiopia`                              | B-served | accepted accepted accepted   | 0/3   | 3 3 2       | 44,704       | 505               | 9.2 16.2 9.8   |
| `rank-top-donors-malawi`                            | A        | accepted accepted accepted   | 3/3   | 3 3 3       | 50,984       | 876               | 14.7 12.6 13.8 |
| `rank-top-donors-malawi`                            | B-served | accepted accepted accepted   | 3/3   | 3 3 3       | 49,768       | 903               | 12.1 12.7 11.3 |
| `share-two-donors-of-kenya-oda`                     | A        | accepted accepted accepted   | 3/3   | 3 5 3       | 63,799       | 1,341             | 20.9 44.1 12.3 |
| `share-two-donors-of-kenya-oda`                     | B-served | accepted accepted accepted   | 3/3   | 6 4 5       | 87,856       | 1,901             | 37.5 16.5 67.7 |
| `quotient-oda-per-person-ethiopia`                  | A        | accepted accepted accepted   | 3/3   | 2 2 3       | 39,044       | 513               | 7.3 8.3 19.9   |
| `quotient-oda-per-person-ethiopia`                  | B-served | accepted accepted accepted   | 3/3   | 2 3 3       | 43,942       | 618               | 7.0 30.8 79.4  |
| `join-gdp-debt-service-ghana`                       | A        | accepted accepted accepted   | 3/3   | 2 2 2       | 33,555       | 351               | 7.9 44.2 5.3   |
| `join-gdp-debt-service-ghana`                       | B-served | accepted accepted accepted   | 3/3   | 2 2 2       | 32,735       | 345               | 6.0 5.4 20.3   |

## Tool calls

| Arm      | Tool         | Calls | Median s, call to response | Median MCP overhead ms |
| -------- | ------------ | ----- | -------------------------- | ---------------------- |
| A        | get_skill    | 33    | 0.0032                     |                        |
| A        | propose_plan | 47    | 0.0099                     |                        |
| B-served | get_skill    | 33    | 0.0032                     |                        |
| B-served | propose_plan | 53    | 0.0109                     | 2.10                   |
| B-served | start_plan   | 30    | harness call               | 0.95                   |
