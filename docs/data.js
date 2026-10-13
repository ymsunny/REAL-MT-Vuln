// Tables 1, 3 and 4 from the supplied AAAI 2026 paper.
window.REAL_MT = {
  "directions": [
    "Hi→En",
    "Fa→En",
    "Fi→En",
    "Ja→En",
    "Fr→En",
    "Ko→En",
    "Ru→En",
    "De→En",
    "En→Fa",
    "En→De"
  ],
  "fullDirections": [
    "Hindi → English",
    "Persian → English",
    "Finnish → English",
    "Japanese → English",
    "French → English",
    "Korean → English",
    "Russian → English",
    "German → English",
    "English → Persian",
    "English → German"
  ],
  "conditions": [
    {
      "key": "none",
      "symbol": "Cnone",
      "label": "No context",
      "hint": "No retrieved meaning is supplied.",
      "text": "—",
      "note": "Direct translation provides the reference point for context dependence."
    },
    {
      "key": "gold",
      "symbol": "G",
      "label": "Gold meaning",
      "hint": "Correct idiomatic meaning",
      "text": "down the drain",
      "note": "The oracle context preserves the intended figurative meaning."
    },
    {
      "key": "struct",
      "symbol": "Nstruct",
      "label": "Word-order noise",
      "hint": "Structure-perturbed gold meaning",
      "text": "drain the down",
      "note": "The word order changes while the core meaning is preserved."
    },
    {
      "key": "literal",
      "symbol": "Nliteral",
      "label": "Literal meaning",
      "hint": "Word-for-word translation",
      "text": "to Kankkula’s well",
      "note": "The literal words resemble the idiom but miss its figurative meaning."
    },
    {
      "key": "semantic",
      "symbol": "Nsemantic",
      "label": "Semantic distortion",
      "hint": "Semantic-perturbed literal meaning",
      "text": "to Kankkula’s house",
      "note": "Surface overlap remains, but the retrieved meaning drifts further away."
    },
    {
      "key": "opposite",
      "symbol": "Nopposite",
      "label": "Opposite meaning",
      "hint": "Meaning that contradicts the idiom",
      "text": "to good use",
      "note": "The hint reverses the intended meaning of wasting something."
    }
  ],
  "models": [
    {
      "name": "Qwen2.5-7B-Instruct",
      "values": [
        [
          0.8,
          0.6,
          0.4,
          1.1,
          1.5,
          1.6,
          1.8,
          1.7,
          0.8,
          1.5
        ],
        [
          2.1,
          2.5,
          2.2,
          2.4,
          2.5,
          2.6,
          2.7,
          2.7,
          1.1,
          2.2
        ],
        [
          1.9,
          2.2,
          2.0,
          2.2,
          2.3,
          2.5,
          2.5,
          2.5,
          1.1,
          2.2
        ],
        [
          1.3,
          1.1,
          0.7,
          1.5,
          1.5,
          1.9,
          1.9,
          1.6,
          0.9,
          2.0
        ],
        [
          0.8,
          0.8,
          0.5,
          1.2,
          1.4,
          1.4,
          1.6,
          1.3,
          1.4,
          1.4
        ],
        [
          0.3,
          0.5,
          0.4,
          0.9,
          0.7,
          1.4,
          1.2,
          0.8,
          0.7,
          1.2
        ]
      ],
      "averages": [
        1.2,
        2.3,
        2.1,
        1.4,
        1.2,
        0.8
      ]
    },
    {
      "name": "Qwen2.5-14B-Instruct",
      "values": [
        [
          1.3,
          1.2,
          0.6,
          1.7,
          2.1,
          2.0,
          2.1,
          1.9,
          1.2,
          2.0
        ],
        [
          2.3,
          2.6,
          2.4,
          2.4,
          2.6,
          2.7,
          2.7,
          2.8,
          1.5,
          2.4
        ],
        [
          2.1,
          2.3,
          2.1,
          2.2,
          2.4,
          2.6,
          2.7,
          2.7,
          1.5,
          2.4
        ],
        [
          1.5,
          1.2,
          0.7,
          1.6,
          1.7,
          2.1,
          2.1,
          1.8,
          1.1,
          2.1
        ],
        [
          0.9,
          1.0,
          0.5,
          1.3,
          1.2,
          1.7,
          1.6,
          1.2,
          0.8,
          1.5
        ],
        [
          0.5,
          0.5,
          0.5,
          0.9,
          0.6,
          1.3,
          1.1,
          0.7,
          0.7,
          0.9
        ]
      ],
      "averages": [
        1.6,
        2.3,
        2.3,
        1.6,
        1.2,
        0.8
      ]
    },
    {
      "name": "Qwen3-8B (non-thinking)",
      "values": [
        [
          1.3,
          1.0,
          0.6,
          1.4,
          1.7,
          1.8,
          1.8,
          1.7,
          1.0,
          1.8
        ],
        [
          2.1,
          2.5,
          2.4,
          2.3,
          2.5,
          2.6,
          2.7,
          2.7,
          1.3,
          2.2
        ],
        [
          2.0,
          2.2,
          2.0,
          2.1,
          2.3,
          2.5,
          2.6,
          2.4,
          1.5,
          2.2
        ],
        [
          1.4,
          1.1,
          0.8,
          1.5,
          1.2,
          2.0,
          1.8,
          1.6,
          1.2,
          2.0
        ],
        [
          0.9,
          0.9,
          0.5,
          1.0,
          1.0,
          1.5,
          1.3,
          1.0,
          1.0,
          1.5
        ],
        [
          0.6,
          0.5,
          0.3,
          0.8,
          0.4,
          1.5,
          1.0,
          0.7,
          1.0,
          1.5
        ]
      ],
      "averages": [
        1.4,
        2.3,
        2.2,
        1.5,
        1.1,
        0.8
      ]
    },
    {
      "name": "Qwen3-8B (thinking)",
      "values": [
        [
          1.3,
          1.0,
          0.6,
          1.4,
          1.8,
          1.9,
          1.8,
          1.7,
          1.2,
          2.0
        ],
        [
          2.2,
          2.5,
          2.4,
          2.5,
          1.2,
          2.7,
          2.7,
          2.8,
          1.5,
          2.3
        ],
        [
          2.0,
          2.2,
          2.1,
          2.2,
          2.3,
          1.9,
          2.5,
          2.5,
          1.5,
          2.3
        ],
        [
          1.4,
          1.1,
          0.8,
          1.4,
          1.5,
          0.7,
          1.8,
          1.5,
          1.1,
          2.0
        ],
        [
          0.6,
          0.8,
          0.5,
          0.8,
          0.9,
          1.2,
          1.2,
          1.0,
          0.9,
          1.2
        ],
        [
          0.1,
          0.5,
          0.2,
          0.3,
          0.1,
          0.6,
          0.5,
          0.2,
          0.5,
          0.6
        ]
      ],
      "averages": [
        1.5,
        2.3,
        2.2,
        1.3,
        0.9,
        0.4
      ]
    }
  ],
  "mitigation": [
    {
      "direction": "Fr→En",
      "scores": [
        {
          "name": "Baseline",
          "none": 1.5,
          "gold": 2.5,
          "goldCar": 67.0,
          "opposite": 0.7,
          "oppositeCar": 68.0
        },
        {
          "name": "Vanilla",
          "none": 1.9,
          "gold": 2.4,
          "goldCar": 44.0,
          "opposite": 1.1,
          "oppositeCar": 49.0
        },
        {
          "name": "CDA",
          "none": 1.9,
          "gold": 2.3,
          "goldCar": 32.0,
          "opposite": 1.7,
          "oppositeCar": 21.0
        },
        {
          "name": "ALI",
          "none": 1.9,
          "gold": 2.2,
          "goldCar": 30.0,
          "opposite": 1.8,
          "oppositeCar": 11.0
        },
        {
          "name": "CK-PLUG",
          "none": 1.5,
          "gold": 2.4,
          "goldCar": 56.0,
          "opposite": 1.3,
          "oppositeCar": 41.0
        }
      ]
    },
    {
      "direction": "Ja→En",
      "scores": [
        {
          "name": "Baseline",
          "none": 1.1,
          "gold": 2.4,
          "goldCar": 75.0,
          "opposite": 0.9,
          "oppositeCar": 64.0
        },
        {
          "name": "Vanilla",
          "none": 1.4,
          "gold": 2.4,
          "goldCar": 69.0,
          "opposite": 0.9,
          "oppositeCar": 62.0
        },
        {
          "name": "CDA",
          "none": 1.4,
          "gold": 2.0,
          "goldCar": 51.0,
          "opposite": 1.4,
          "oppositeCar": 23.0
        },
        {
          "name": "ALI",
          "none": 1.3,
          "gold": 1.9,
          "goldCar": 48.0,
          "opposite": 1.4,
          "oppositeCar": 16.0
        },
        {
          "name": "CK-PLUG",
          "none": 1.1,
          "gold": 2.1,
          "goldCar": 65.0,
          "opposite": 1.3,
          "oppositeCar": 32.0
        }
      ]
    },
    {
      "direction": "Hi→En",
      "scores": [
        {
          "name": "Baseline",
          "none": 0.8,
          "gold": 2.1,
          "goldCar": 85.0,
          "opposite": 0.3,
          "oppositeCar": 74.0
        },
        {
          "name": "Vanilla",
          "none": 0.8,
          "gold": 2.2,
          "goldCar": 81.0,
          "opposite": 0.4,
          "oppositeCar": 65.0
        },
        {
          "name": "CDA",
          "none": 0.8,
          "gold": 2.0,
          "goldCar": 70.0,
          "opposite": 0.7,
          "oppositeCar": 42.0
        },
        {
          "name": "ALI",
          "none": 0.8,
          "gold": 1.7,
          "goldCar": 54.0,
          "opposite": 0.8,
          "oppositeCar": 26.0
        },
        {
          "name": "CK-PLUG",
          "none": 0.8,
          "gold": 1.5,
          "goldCar": 64.0,
          "opposite": 0.6,
          "oppositeCar": 36.0
        }
      ]
    }
  ]
};
