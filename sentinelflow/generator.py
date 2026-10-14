from pathlib import Path
import numpy as np
import pandas as pd


def generate_events(n_per_scenario: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    rows = []
    start = pd.Timestamp("2026-01-01 08:00:00")

    configs = {
        "Retail": {
            "events": ["purchase", "inventory_update", "clickstream"],
            "locations": ["Store-A", "Store-B", "Online"],
        },
        "Manufacturing": {
            "events": ["machine_reading", "maintenance", "production"],
            "locations": ["Plant-A", "Plant-B"],
        },
        "Cybersecurity": {
            "events": ["login", "network_activity", "authentication"],
            "locations": ["Office-A", "Office-B", "Remote"],
        },
    }

    for scenario, cfg in configs.items():

        for i in range(n_per_scenario):

            timestamp = start + pd.Timedelta(
                minutes=int(rng.integers(0, 60 * 24 * 30))
            )

            entity_id = (
                f"{scenario[:3].upper()}-{rng.integers(1, 31):03d}"
            )

            event_type = rng.choice(cfg["events"])
            location = rng.choice(cfg["locations"])

            # -------------------------------
            # Retail
            # -------------------------------
            if scenario == "Retail":

                value_1 = max(
                    10,
                    rng.normal(450, 140)
                )

                value_2 = max(
                    1,
                    rng.normal(18, 7)
                )

                value_3 = max(
                    5,
                    rng.normal(55, 18)
                )

                risk_score = (
                    (value_1 > 750) * 0.35
                    + (value_2 > 30) * 0.30
                    + (value_3 < 20) * 0.35
                )

            # -------------------------------
            # Manufacturing
            # -------------------------------
            elif scenario == "Manufacturing":

                value_1 = max(
                    30,
                    rng.normal(68, 12)
                )

                value_2 = max(
                    0.2,
                    rng.normal(4.2, 1.5)
                )

                value_3 = max(
                    20,
                    rng.normal(68, 15)
                )

                risk_score = (
                    (value_1 > 85) * 0.40
                    + (value_2 > 7) * 0.35
                    + (value_3 > 90) * 0.25
                )

            # -------------------------------
            # Cybersecurity
            # -------------------------------
            else:

                value_1 = max(
                    1,
                    rng.normal(12, 6)
                )

                value_2 = min(
                    1,
                    max(0, rng.normal(0.08, 0.05))
                )

                value_3 = max(
                    500,
                    rng.normal(4500, 1500)
                )

                risk_score = (
                    (value_1 > 25) * 0.30
                    + (value_2 > 0.20) * 0.45
                    + (value_3 > 7500) * 0.25
                )

            # Inject abnormal events
            if i % 10 == 0:

                if scenario == "Retail":
                    value_1 *= 2.1
                    value_2 *= 2.0
                    value_3 *= 0.25

                elif scenario == "Manufacturing":
                    value_1 += 35
                    value_2 += 5
                    value_3 += 25

                else:
                    value_1 *= 3
                    value_2 = min(1, value_2 + 0.35)
                    value_3 *= 2.2

                risk_score = min(
                    1.0,
                    risk_score + 0.65
                )

            risk_label = int(risk_score >= 0.5)

            rows.append({
                "timestamp": timestamp,
                "entity_id": entity_id,
                "scenario": scenario,
                "event_type": event_type,
                "location": location,
                "value_1": round(float(value_1), 3),
                "value_2": round(float(value_2), 5),
                "value_3": round(float(value_3), 3),
                "risk_label": risk_label,
            })

    return (
        pd.DataFrame(rows)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def save_sample_data(
    path: str = "data/sample_events.csv",
    n_per_scenario: int = 300
) -> str:

    output = Path(path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df = generate_events(
        n_per_scenario=n_per_scenario
    )

    df.to_csv(
        output,
        index=False
    )

    return str(output)


if __name__ == "__main__":

    print(
        f"Created: {save_sample_data()}"
    )