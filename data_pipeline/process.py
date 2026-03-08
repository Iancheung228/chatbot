import os
import csv
import argparse
from data_pipeline.ocr import screenshotsToText
from api.llm import ocr_to_smart_text

def multiline_to_singleline(messages):
    # Sort by vertical position just in case
    messages.sort(key=lambda m: m[0])

    merged = []
    for _, sender, text in messages:
        if not merged:
            merged.append([sender, text])
        else:
            last_sender, last_text = merged[-1]
            if sender == last_sender:
                merged[-1][1] = last_text + " " + text  # merge consecutive
            else:
                merged.append([sender, text])

    # Convert back to tuple if you prefer
    merged = [(s, t.strip()) for s, t in merged]

    # for s, t in merged:
    #     print(f"{s}: {t}")
    return merged

def process_screenshot_to_messages(screenshot_paths):
    """Extract and merge messages from multiple screenshots."""
    all_messages = []
    for img_path in screenshot_paths:
        messages = screenshotsToText(img_path)
        all_messages.extend(messages)
    # Merge multiline messages
    merged = multiline_to_singleline(all_messages)
    return merged

def main_func(data_folder, csv_path):
    """
    data_folder: path to the folder containing example folders
    csv_path: output CSV file path
    """
    all_rows = []
    # Loop through each example folder
    for example_name in sorted(os.listdir(data_folder)):
        
        example_path = os.path.join(data_folder, example_name)
        print(example_path)
        if not os.path.isdir(example_path):
            continue
        # Collect all screenshot image paths in this example folder
        screenshot_paths = []
        
        for fname in sorted(os.listdir(example_path)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                screenshot_paths.append(os.path.join(example_path, fname))
        # Extract and merge messages from all screenshots in this example
        merged_messages = process_screenshot_to_messages(screenshot_paths)
        # For each message, get smart text and add to all_rows
        for sender, original_text in merged_messages:
            smart_text = ocr_to_smart_text(original_text)
            all_rows.append([example_name, sender, original_text, smart_text])

    # Write all rows to CSV
    with open(csv_path, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["example", "sender", "original_text", "smart_text"])
        writer.writerows(all_rows)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process screenshots into a conversation CSV.")
    parser.add_argument("--data_folder", required=True, help="Path to folder containing example subfolders")
    parser.add_argument("--output", default="output.csv", help="Output CSV file path")
    args = parser.parse_args()
    main_func(args.data_folder, args.output)
    print("Done! Output saved to", args.output)
