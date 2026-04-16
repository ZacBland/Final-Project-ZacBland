#!/usr/bin/env python3
"""Look up nutritional statistics for a Nutrition5k dish by ID.

Usage:
    python lookup_dish.py dish_1561662216
    python lookup_dish.py dish_1572974428 --data_dir ./data
"""

import argparse
import os
import sys

METADATA_CSVS = [
    "metadata/dish_metadata_cafe1.csv",
    "metadata/dish_metadata_cafe2.csv",
]


def load_dish(data_dir, dish_id):
    for csv_rel in METADATA_CSVS:
        csv_path = os.path.join(data_dir, csv_rel)
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, "r") as f:
            for line in f:
                fields = line.strip().split(",")
                if fields[0] == dish_id:
                    return fields
    return None


def main():
    parser = argparse.ArgumentParser(description="Look up nutrition stats for a dish.")
    parser.add_argument("dish_id", help="Dish ID (e.g. dish_1561662216)")
    parser.add_argument("--data_dir", default="./data", help="Path to data directory (default: ./data)")
    args = parser.parse_args()

    fields = load_dish(args.data_dir, args.dish_id)
    if fields is None:
        print(f"Dish '{args.dish_id}' not found in metadata CSVs.")
        sys.exit(1)

    # Dish-level totals (indices 0-5)
    dish_id = fields[0]
    total_cal = float(fields[1])
    total_mass = float(fields[2])
    total_fat = float(fields[3])
    total_carb = float(fields[4])
    total_protein = float(fields[5])

    print(f"\n{'=' * 60}")
    print(f"  Dish: {dish_id}")
    print(f"{'=' * 60}")
    print(f"  Calories:  {total_cal:>10.2f} kcal")
    print(f"  Mass:      {total_mass:>10.2f} g")
    print(f"  Fat:       {total_fat:>10.2f} g")
    print(f"  Carbs:     {total_carb:>10.2f} g")
    print(f"  Protein:   {total_protein:>10.2f} g")

    # Per-ingredient breakdown (groups of 7 starting at index 6)
    ingr_fields = fields[6:]
    if len(ingr_fields) >= 7:
        print(f"\n  {'Ingredient':<25} {'Grams':>8} {'Cal':>8} {'Fat':>8} {'Carb':>8} {'Protein':>8}")
        print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for i in range(0, len(ingr_fields) - 6, 7):
            name = ingr_fields[i + 1]
            grams = float(ingr_fields[i + 2])
            cal = float(ingr_fields[i + 3])
            fat = float(ingr_fields[i + 4])
            carb = float(ingr_fields[i + 5])
            protein = float(ingr_fields[i + 6])
            print(f"  {name:<25} {grams:>8.2f} {cal:>8.2f} {fat:>8.2f} {carb:>8.2f} {protein:>8.2f}")

    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
