# Gemini comparison: results

558 runs over 62 tasks (baseline 186, pai 186, pai-validated 186), 0 of them errors; 1405 model calls and 24 cache creates tapped, 0 tap bookkeeping errors.

## Baseline alone

| Line                                          | Numbers                                                                                                        |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| W2 cache requests carry no prefix             | 0 violations in 485 cached of 485 requests                                                                     |
| W3 signatures kept, thought text not replayed | 0 signatures lost of 547 signed originals replayed (0 changed, 0 added, 0 unmatched), 0 thought parts replayed |
| W4 cached share, later calls                  | 84.7% of 4,774,636 prompt tokens over 295 calls                                                                |
| W5 seconds to first chunk                     | p50 3.30 s, p90 5.77 s, first-call p50 2.09 s, 476 streams                                                     |

## Pass lines

| Line                                          | Arm           | Result | Arm numbers                                                                                                                                                                                       | Baseline numbers                                            |
| --------------------------------------------- | ------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| W2 cache requests carry no prefix             | pai           | pass   | 0 violations in 452 cached of 452 requests                                                                                                                                                        | 0 in 485 of 485                                             |
| W3 signatures kept, thought text not replayed | pai           | pass   | 0 signatures lost of 389 signed originals replayed (0 changed, 0 added, 0 unmatched), 0 thought parts replayed                                                                                    | 0 of 547 (0 changed, 0 added, 0 unmatched), 0 thought parts |
| W4 cached share, later calls                  | pai           | pass   | 85.3% of 4,260,817 prompt tokens over 265 calls (+0.6% against the baseline)                                                                                                                      | 84.7% of 4,774,636 over 295 calls                           |
| W5 seconds to first chunk                     | pai           | pass   | p50 3.42 s (1.04 times), p90 5.74 s, first-call p50 2.16 s, 447 streams                                                                                                                           | p50 3.30 s, p90 5.77 s, first-call p50 2.09 s, 476 streams  |
| Contents parity, first request                | pai           | pass   | 62 identical of 62 tasks                                                                                                                                                                          | n/a                                                         |
| O1 outcomes against the baseline              | pai           | pass   | first proposal 127/177, kind pass 173/177 (58/59 tasks); paired first proposal 8 better, 11 worse, kind 4 better, 1 worse; triggers fired: none; 0 tasks fall from passing                        | first proposal 129/177, kind pass 169/177 (56/59 tasks)     |
| W2 cache requests carry no prefix             | pai-validated | pass   | 0 violations in 468 cached of 468 requests                                                                                                                                                        | 0 in 485 of 485                                             |
| W3 signatures kept, thought text not replayed | pai-validated | pass   | 0 signatures lost of 449 signed originals replayed (0 changed, 0 added, 0 unmatched), 0 thought parts replayed                                                                                    | 0 of 547 (0 changed, 0 added, 0 unmatched), 0 thought parts |
| W4 cached share, later calls                  | pai-validated | pass   | 85.1% of 4,528,223 prompt tokens over 281 calls (+0.4% against the baseline)                                                                                                                      | 84.7% of 4,774,636 over 295 calls                           |
| W5 seconds to first chunk                     | pai-validated | FAIL   | p50 3.95 s (1.20 times), p90 6.12 s, first-call p50 2.66 s, 461 streams                                                                                                                           | p50 3.30 s, p90 5.77 s, first-call p50 2.09 s, 476 streams  |
| Contents parity, first request                | pai-validated | pass   | 62 identical of 62 tasks                                                                                                                                                                          | n/a                                                         |
| O1 outcomes against the baseline              | pai-validated | FAIL   | first proposal 125/177, kind pass 168/177 (57/59 tasks); paired first proposal 8 better, 10 worse, kind 2 better, 4 worse; triggers fired: none; 1 tasks fall from passing (sum-oda-fdi-ethiopia) | first proposal 129/177, kind pass 169/177 (56/59 tasks)     |

W2 passes only when at least one request carried a cache. W3 passes only when at least one replayed call had a signed original. W4 passes when the arm's share is not more than 5 points below the baseline's. O1 passes when no switch trigger fires and no task the baseline passes in a majority of runs falls below that.

## W1

From `parity.json`:

```json
{
 "tasks": {
  "series-debt-stock-kenya-ghana": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "series-france-oda-africa": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "series-gdp-ghana": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "series-oda-by-sector-kenya": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "series-oda-germany-kenya": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "series-population-three": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "series-zambia-debt-to-china": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "aggregate-average-gdp-ghana": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "aggregate-average-oda-kenya": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "aggregate-combined-oda-kenya-uganda": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    },
    "alternative 0": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "aggregate-cumulative-debt-flows-china-kenya": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    },
    "alternative 0": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "aggregate-cumulative-oda-g7-africa": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "aggregate-health-oda-kenya-ghana": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "aggregate-health-population-oda-all-donors": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "aggregate-health-population-oda-germany-by-sector": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    },
    "alternative 0": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "aggregate-health-population-oda-kenya-by-sector": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "aggregate-malaria-oda-all-recipients": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "aggregate-total-debt-service-zambia": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "aggregate-total-oda-kenya": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "aggregate-two-donors-yearly": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "growth-absolute-debt-zambia": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "growth-decline-german-oda": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "growth-gdp-kenya": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "growth-population-nigeria": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "growth-total-oda-ghana": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "growth-uk-oda-kenya": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "comparison-debt-service-climate-finance-kenya": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "comparison-debt-stock-gdp-ghana": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "comparison-oda-kenya-uganda": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    },
    "alternative 0": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "comparison-us-uk-oda-kenya": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    },
    "alternative 0": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "sum-bilateral-plus-imputed-france-guatemala": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "sum-bilateral-plus-imputed-japan-indonesia": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "sum-net-flows-aid-debt-ghana": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "sum-oda-debt-disbursements-kenya": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "sum-oda-fdi-ethiopia": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "sum-then-quotient-germany-tanzania": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "rank-bottom-five-african-oda-2021": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    },
    "alternative 0": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "rank-donor-each-year-kenya": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "rank-external-debt-african-countries": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    },
    "alternative 0": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "rank-fastest-gdp-growth": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "rank-sectors-kenya": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    },
    "alternative 0": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "rank-top-donors-guatemala": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "rank-top-donors-guatemala-since-2020": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "rank-top-gdp-2022": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "share-germany-of-oda-ethiopia": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    },
    "alternative 0": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "share-health-oda-kenya": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    },
    "alternative 0": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    },
    "alternative 1": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "share-nigeria-world-population": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "share-two-donors-of-kenya-oda": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    },
    "alternative 0": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "quotient-debt-service-per-gdp-ghana": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    },
    "alternative 0": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "quotient-external-debt-gdp-kenya-nigeria": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "quotient-gdp-per-person-guatemala": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "quotient-keyed-oda-per-person-three": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    },
    "alternative 0": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "quotient-oda-per-person-ethiopia": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    },
    "alternative 0": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "quotient-oda-per-person-france": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "quotient-oda-per-person-malawi": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    },
    "alternative 0": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "join-gdp-debt-service-ghana": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "join-malaria-domestic-and-aid-kenya": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "join-oda-debt-service-zambia": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "join-population-gdp-kenya": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "filter-donors-above-10b-2021": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "filter-donors-above-200m-kenya": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  },
  "filter-gdp-above-trillion-2022": {
   "equal": false,
   "explained": true,
   "first_request": {
    "fields": {
     "declarations": [
      {
       "path": "reject_plan",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "tools_grouping": [
      {
       "path": "",
       "adk": [
        5
       ],
       "pai": [
        1,
        1,
        1,
        1,
        1
       ]
      }
     ],
     "tool_config": [
      {
       "path": "",
       "adk": null,
       "pai": {
        "function_calling_config": {
         "mode": "AUTO"
        }
       }
      }
     ],
     "other_config": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "declarations": [
      "reject_plan_parameters"
     ],
     "tools_grouping": [
      "tools_grouping"
     ],
     "tool_config": [
      "tool_config"
     ],
     "other_config": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "prefix": {
    "fields": {
     "prefix_key": [
      {
       "path": "",
       "adk": "44e54fd13af82c3366fd1343c22a9c8872f682ff53e6ffddb9842af759367a97",
       "pai": "8b68dbb872a092bc4d774ad1cf999fbfc8731ecae34d50d65648253036c137a2"
      }
     ],
     "prefix_content": [
      {
       "path": "declarations[4]",
       "adk": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\"}",
       "pai": "{\"description\":\"Record that the user rejected the current plan.\\n\\nNothing in the plan may run afterwards. Call `propose_plan` again if\\nthe user wants a revision rather than an outright stop.\",\"name\":\"reject_plan\",\"parameters_json_schema\":{\"properties\":{},\"type\":\"object\"}}"
      }
     ],
     "cached_request": [
      {
       "path": "http_options.headers.Content-Type",
       "adk": "<absent>",
       "pai": "application/json"
      },
      {
       "path": "http_options.headers.User-Agent",
       "adk": "<absent>",
       "pai": "pydantic-ai/2.64.0"
      },
      {
       "path": "http_options.headers.user-agent",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "http_options.headers.x-goog-api-client",
       "adk": "google-adk/2.10.0 gl-python/3.13.14",
       "pai": "<absent>"
      },
      {
       "path": "response_modalities",
       "adk": "<absent>",
       "pai": [
        "TEXT"
       ]
      }
     ]
    },
    "equal": false,
    "explained": {
     "prefix_key": [
      "prefix_key"
     ],
     "prefix_content": [
      "reject_plan_parameters"
     ],
     "cached_request": [
      "headers",
      "response_modalities"
     ]
    },
    "unexplained": []
   },
   "outcomes": {
    "reference": {
     "equal": true,
     "pai_faults": [],
     "adk_faults": []
    }
   }
  }
 },
 "thoughts": {
  "adk": {
   "thought_text_replayed": false,
   "signature_kept": true
  },
  "pai": {
   "thought_text_replayed": false,
   "signature_kept": true
  },
  "equal": true,
  "second_request_contents": [
   {
    "path": "[2].parts[0].function_call.id",
    "adk": "<absent>",
    "pai": "pyd_ai_20b32cbd16ae408b9aaae77c903f3bf1"
   },
   {
    "path": "[3].parts[0].function_response.id",
    "adk": "<absent>",
    "pai": "pyd_ai_20b32cbd16ae408b9aaae77c903f3bf1"
   }
  ],
  "explained": {
   "second_request_contents": [
    "call_ids"
   ]
  },
  "unexplained": []
 },
 "summary": {
  "tasks": 62,
  "plans_proposed": 78,
  "plans_same_outcome": 78,
  "first_request_equal": 0,
  "prefix_equal": 0,
  "fields_differing": {
   "first_request.declarations": 62,
   "first_request.other_config": 62,
   "first_request.tool_config": 62,
   "first_request.tools_grouping": 62,
   "prefix.cached_request": 62,
   "prefix.prefix_content": 62,
   "prefix.prefix_key": 62
  },
  "tasks_every_difference_explained": 62,
  "unexplained": [],
  "thoughts_equal": true,
  "thoughts_unexplained": [],
  "pai_behaviour": {
   "reject_plan_parameters": "pydantic_ai_slim/pydantic_ai/models/google.py:2523 always sends `parameters_json_schema`, so `reject_plan`, which ADK declares with no parameters, goes with an empty object schema",
   "tools_grouping": "pydantic_ai_slim/pydantic_ai/models/google.py:1094 puts each function declaration in a `Tool` of its own",
   "tool_config": "pydantic_ai_slim/pydantic_ai/models/google.py:1053-1074 always sends a function calling config with function tools (AUTO in like-adk mode, VALIDATED in validated mode); ADK sends none",
   "headers": "pydantic_ai_slim/pydantic_ai/models/google.py:1256 sends its own headers; ADK sends its tracking headers",
   "response_modalities": "pydantic_ai_slim/pydantic_ai/models/google.py:1250 always sends `response_modalities` ['TEXT']",
   "prefix_key": "follows from the three above and from the system instruction going as a `Content` (pydantic_ai_slim/pydantic_ai/models/google.py:1510) where ADK's goes as a string: `prefix_key` hashes each framework's own form, so the two arms never share a cache",
   "call_ids": "pydantic_ai_slim/pydantic_ai/models/google.py:2141 and :1557 send Pydantic AI's tool call id on each replayed function call and response; ADK strips its own client ids"
  },
  "pass": true
 },
 "broken": null
}
```

## Detail

### Contents parity, pai

| Task      | Status    | First difference |
| --------- | --------- | ---------------- |
| all tasks | identical |                  |

### Contents parity, pai-validated

| Task      | Status    | First difference |
| --------- | --------- | ---------------- |
| all tasks | identical |                  |
