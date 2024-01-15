import utils.image_cutting_support as ics 
import os


folder = "/Users/charlieturner/Documents/CountingBoats/TestGBR"

# Walk through each subdir, and for each image 'composite.tif' create the padded png for it
# as root_png.png
for root, dirs, files in os.walk(folder):
    for file in files:
        if file == 'composite.tif':
            # find the json file:
            date_file = [f for f in files if f.endswith('xml')][0]
            date = date_file.split('_')[0]
            aoi = root.split('_')[-2].split('/')[-1]
            name = f"{date}_{aoi}.tif"
            print(name)
            os.rename(os.path.join(root, file), 
                      os.path.join(root, name))
            # want to create a png for this
            new_name = os.path.join(folder, f"{name.split('.')[0]}.png")
            if not os.path.exists(os.path.join(root, f"{name.split('.')[0]}.png")):
                ics.create_padded_png(root, folder, name)
        # if the file is a tif and first part is a date, don't need to rename
        elif file.endswith('tif') and file.split('_')[0].isdigit():
            # check if we have already created a png for this
            new_name = os.path.join(folder, f"{file.split('.')[0]}.png")
            if not os.path.exists(os.path.join(folder, f"{file.split('.')[0]}.png")):
                ics.create_padded_png(root, folder, file)
