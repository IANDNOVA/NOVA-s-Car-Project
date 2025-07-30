from collections import Counter

# Read all logged observations
with open("NOVA_Vision_Log.txt", "r") as log:
    lines = log.readlines()

# Extract object names (everything after "I see a ")
objects = [line.strip().replace("I see a ", "") for line in lines if line.startswith("I see a")]

# Count each object
summary = Counter(objects)

# Print summary to screen
print("📊 NOVA's Vision Summary")
for obj, count in summary.items():
    print(f"{obj}: {count}")
