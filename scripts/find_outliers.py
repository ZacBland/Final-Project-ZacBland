#!/usr/bin/env python3
"""Find dishes with extreme nutritional values (outliers) in the Nutrition5k dataset.

Usage:
    python3 scripts/find_outliers.py
    python3 scripts/find_outliers.py --sort mass --top 30
    python3 scripts/find_outliers.py --sort calories --bottom 20
    python3 scripts/find_outliers.py --sort fat --data_dir ./data
"""

import argparse
import os

METADATA_CSVS = [
    "metadata/dish_metadata_cafe1.csv",
    "metadata/dish_metadata_cafe2.csv",
]

LABEL_NAMES = ["calories", "mass", "fat", "carbs", "protein"]


def load_all_dishes(data_dir):
    """Load all dishes with nutrition and ingredient info."""
    dishes = []
    for csv_rel in METADATA_CSVS:
        csv_path = os.path.join(data_dir, csv_rel)
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, "r") as f:
            for line in f:
                fields = line.strip().split(",")
                if not fields or not fields[0].startswith("dish_"):
                    continue
                try:
                    dish = {
                        "dish_id": fields[0],
                        "calories": float(fields[1]),
                        "mass": float(fields[2]),
                        "fat": float(fields[3]),
                        "carbs": float(fields[4]),
                        "protein": float(fields[5]),
                    }
                    # Extract ingredient names
                    ingr_fields = fields[6:]
                    ingredients = []
                    for i in range(0, len(ingr_fields) - 6, 7):
                        ingredients.append({
                            "name": ingr_fields[i + 1],
                            "grams": float(ingr_fields[i + 2]),
                        })
                    dish["ingredients"] = ingredients
                    dish["cal_density"] = dish["calories"] / max(dish["mass"], 0.1)
                    dishes.append(dish)
                except (ValueError, IndexError):
                    continue
    return dishes


def top_ingredients(dish, n=3):
    """Return the top N ingredients by mass."""
    sorted_ingr = sorted(dish["ingredients"], key=lambda x: x["grams"], reverse=True)
    return ", ".join(f"{ing['name']} ({ing['grams']:.0f}g)" for ing in sorted_ingr[:n])


def print_table(dishes, sort_key, label):
    """Print a formatted table of dishes."""
    print(f"\n{'=' * 110}")
    print(f"  {label}")
    print(f"{'=' * 110}")
    print(f"  {'#':<4} {'Dish ID':<22} {'Cal':>8} {'Mass':>8} {'Fat':>6} {'Carb':>6} {'Prot':>6} {'Cal/g':>6}  Top Ingredients")
    print(f"  {'-'*4} {'-'*22} {'-'*8} {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*6}  {'-'*30}")
    for i, d in enumerate(dishes, 1):
        ingr_str = top_ingredients(d)
        print(f"  {i:<4} {d['dish_id']:<22} {d['calories']:>8.1f} {d['mass']:>8.1f} {d['fat']:>6.1f} {d['carbs']:>6.1f} "
              f"{d['protein']:>6.1f} {d['cal_density']:>6.2f}  {ingr_str}")
    print(f"{'=' * 110}")


def print_stats(dishes):
    """Print basic statistics for all nutrients."""
    import statistics
    print(f"\n{'=' * 70}")
    print(f"  DATASET STATISTICS ({len(dishes)} dishes)")
    print(f"{'=' * 70}")
    print(f"  {'Nutrient':<12} {'Mean':>10} {'Median':>10} {'StdDev':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for key in LABEL_NAMES + ["cal_density"]:
        vals = [d[key] for d in dishes]
        label = key if key != "cal_density" else "cal/gram"
        print(f"  {label:<12} {statistics.mean(vals):>10.2f} {statistics.median(vals):>10.2f} "
              f"{statistics.stdev(vals):>10.2f} {min(vals):>10.2f} {max(vals):>10.2f}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Find outlier dishes in Nutrition5k by nutritional values.")
    parser.add_argument("--data_dir", default="./data", help="Path to data directory (default: ./data)")
    parser.add_argument("--sort", default="mass", choices=LABEL_NAMES + ["cal_density"],
                        help="Sort by this nutrient (default: mass)")
    parser.add_argument("--top", type=int, default=20, help="Show top N highest (default: 20)")
    parser.add_argument("--bottom", type=int, default=10, help="Show bottom N lowest (default: 10)")
    parser.add_argument("--stats", action="store_true", default=True,
                        help="Show dataset statistics (default: True)")
    args = parser.parse_args()

    dishes = load_all_dishes(args.data_dir)
    if not dishes:
        print("No dishes found. Check --data_dir path.")
        return

    print(f"Loaded {len(dishes)} dishes")

    if args.stats:
        print_stats(dishes)

    # Sort and show extremes
    sorted_dishes = sorted(dishes, key=lambda d: d[args.sort], reverse=True)

    sort_label = args.sort if args.sort != "cal_density" else "cal/gram"
    print_table(sorted_dishes[:args.top], args.sort,
                f"TOP {args.top} HIGHEST {sort_label.upper()}")

    print_table(sorted_dishes[-args.bottom:], args.sort,
                f"BOTTOM {args.bottom} LOWEST {sort_label.upper()}")


if __name__ == "__main__":
    main()
