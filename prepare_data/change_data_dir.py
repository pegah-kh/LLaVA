import json



original_path = '/home/khayatan/llava/prepare_data/llava_coco_actions.json'
dest_path = '/home/khayatan/llava/prepare_data/llava_coco_action_correct.json'



# Function to update the image paths for both train and val sets
def update_image_paths(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Define the old base path and the possible subdirectories to replace
    old_base_path = '/data/mshukor/data/COCO/images'
    new_base_path = '/data/mshukor/data/coco'

    # Update image paths in the JSON data
    for entry in data:
        if 'image' in entry:
            # Handle both train2014 and val2014 cases
            if 'train2014' in entry['image']:
                entry['image'] = entry['image'].replace(f"{old_base_path}/train2014", f"{new_base_path}/train2014")
                # print(f"Updated train image path: {entry['image']}")
            elif 'val2014' in entry['image']:
                entry['image'] = entry['image'].replace(f"{old_base_path}/val2014", f"{new_base_path}/val2014")
                # print(f"Updated val image path: {entry['image']}")
            else:
                print(f"Image path not updated (no match found): {entry['image']}")

    # Save the updated data to a new file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)


# Call the function
update_image_paths(original_path, dest_path)

