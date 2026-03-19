import csv

filename = "user_data.csv"

# Read the original CSV and modify cell values
rows = []
with open(filename, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        # Replace commas inside each cell with a space
        new_row = [cell.replace(',', ' ') for cell in row]
        rows.append(new_row)

# Write back to the same file
with open(filename, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(rows)