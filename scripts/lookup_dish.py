#!/usr/bin/env python3
"""Look up nutritional statistics for a Nutrition5k dish by ID or search by ingredient.

Usage:
    python3 lookup_dish.py dish_1561662216
    python3 lookup_dish.py --search "hot dog"
    python3 lookup_dish.py --search "chicken" --data_dir ./data
"""

import argparse
import glob
import os
import sys

METADATA_CSVS = [
    "metadata/dish_metadata_cafe1.csv",
    "metadata/dish_metadata_cafe2.csv",
]


def load_all_dishes(data_dir):
    """Load all dishes from metadata CSVs. Returns list of (fields, csv_name)."""
    dishes = []
    for csv_rel in METADATA_CSVS:
        csv_path = os.path.join(data_dir, csv_rel)
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, "r") as f:
            for line in f:
                fields = line.strip().split(",")
                if fields and fields[0].startswith("dish_"):
                    dishes.append(fields)
    return dishes


def load_dish(data_dir, dish_id):
    for fields in load_all_dishes(data_dir):
        if fields[0] == dish_id:
            return fields
    return None


def get_ingredients(fields):
    """Extract ingredient names from a dish's fields."""
    ingr_fields = fields[6:]
    names = []
    for i in range(0, len(ingr_fields) - 6, 7):
        names.append(ingr_fields[i + 1])
    return names


def check_images(data_dir, dish_id):
    """Check which images exist for a dish."""
    images = []
    overhead = os.path.join(data_dir, "imagery", "realsense_overhead", dish_id, "rgb.png")
    if os.path.exists(overhead):
        images.append(("overhead", overhead))

    side_dir = os.path.join(data_dir, "imagery", "side_angles", dish_id)
    if os.path.isdir(side_dir):
        for camera in sorted(os.listdir(side_dir)):
            cam_dir = os.path.join(side_dir, camera)
            if os.path.isdir(cam_dir):
                frames = sorted(glob.glob(os.path.join(cam_dir, "*.png")))
                if frames:
                    images.append((camera, f"{len(frames)} frames"))
    return images


def print_dish(fields, data_dir):
    """Print a dish's full nutritional breakdown and image availability."""
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

    # Per-ingredient breakdown
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

    # Image availability
    images = check_images(data_dir, dish_id)
    print(f"\n  Images:")
    if images:
        for source, detail in images:
            print(f"    {source}: {detail}")
    else:
        print(f"    (no images downloaded)")

    print(f"{'=' * 60}\n")


def search_dishes(data_dir, query):
    """Search for dishes containing an ingredient matching the query."""
    query_lower = query.lower()
    dishes = load_all_dishes(data_dir)
    matches = []
    for fields in dishes:
        ingredients = get_ingredients(fields)
        for ingr in ingredients:
            if query_lower in ingr.lower():
                matches.append((fields, ingr))
                break
    return matches


def main():
    parser = argparse.ArgumentParser(description="Look up nutrition stats for a dish.")
    parser.add_argument("dish_id", nargs="?", help="Dish ID (e.g. dish_1561662216)")
    parser.add_argument("--search", "-s", type=str, help="Search for dishes by ingredient name (e.g. 'hot dog')")
    parser.add_argument("--data_dir", default="./data", help="Path to data directory (default: ./data)")
    parser.add_argument("--limit", type=int, default=10, help="Max results for search (default: 10)")
    args = parser.parse_args()

    if not args.dish_id and not args.search:
        parser.print_help()
        sys.exit(1)

    if args.search:
        matches = search_dishes(args.data_dir, args.search)
        if not matches:
            print(f"No dishes found containing '{args.search}'.")
            sys.exit(1)

        print(f"\nFound {len(matches)} dish(es) containing '{args.search}'")
        print(f"Showing first {min(len(matches), args.limit)}:\n")

        for fields, matched_ingr in matches[:args.limit]:
            dish_id = fields[0]
            cal = float(fields[1])
            mass = float(fields[2])
            images = check_images(args.data_dir, dish_id)
            has_img = "yes" if images else "no"
            print(f"  {dish_id}  {cal:>8.1f} kcal  {mass:>6.0f}g  matched: {matched_ingr:<20}  images: {has_img}")

        print(f"\nUse 'python3 lookup_dish.py <dish_id>' for full details.")
    else:
        fields = load_dish(args.data_dir, args.dish_id)
        if fields is None:
            print(f"Dish '{args.dish_id}' not found in metadata CSVs.")
            sys.exit(1)
        print_dish(fields, args.data_dir)


if __name__ == "__main__":
    main()
