import csv

filename = "human.csv"

# Read CSV
with open(filename, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    rows = list(reader)

# Modify the header
header = rows[0]
new_header = [col.replace(' ', '_').replace('(', '_').replace(')', '_').replace('-', '_') for col in header]
rows[0] = new_header

# Write back to the same file
with open(filename, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(rows)