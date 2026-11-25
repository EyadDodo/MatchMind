
import pandas as pd
import math

# ===== TRAINING DATA =====

files = [
    "data/season-2021.csv",
    "data/season-2122.csv",
    "data/season-2223.csv",
    "data/season-2324.csv"
]

# Load and combine the datasets
matches = pd.concat(
    [pd.read_csv(file) for file in files],
    ignore_index=True
)

# Remove matches without a final result
matches = matches.dropna(subset=["FTR"])

# Count each result
results = matches["FTR"].value_counts()

# Calculate probabilities
total_matches = len(matches)

home_probability = results.get("H", 0) / total_matches
draw_probability = results.get("D", 0) / total_matches
away_probability = results.get("A", 0) / total_matches

# Display training results
print("===== MATCHMIND =====")
print("Training matches:", total_matches)

print("\nPredicted probabilities:")
print(f"Home Win: {home_probability:.2%}")
print(f"Draw: {draw_probability:.2%}")
print(f"Away Win: {away_probability:.2%}")


# ===== TESTING DATA =====

test = pd.read_csv("data/season-2425.csv")

# Remove matches without results
test = test.dropna(subset=["FTR"])

# Display actual results
actual_results = test["FTR"].value_counts()

print("\n===== MATCHMIND TESTING =====")
print("Testing matches:", len(test))

print("\nActual results:")
print("Home wins:", actual_results.get("H", 0))
print("Draws:", actual_results.get("D", 0))
print("Away wins:", actual_results.get("A", 0))


# ===== LOG LOSS =====

probabilities = {
    "H": home_probability,
    "D": draw_probability,
    "A": away_probability
}

total_loss = 0

for result in test["FTR"]:
    total_loss -= math.log(probabilities[result])

log_loss = total_loss / len(test)

print("\n===== MODEL PERFORMANCE =====")
print("Log Loss:", round(log_loss, 4))
