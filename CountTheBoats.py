"""
The semi-automated boat counter.
1. Asks: New order or existing?
NEW:
    1. Which AOI/AOI's?
    2. Which date?
        Latest is default, or specify
    3. Acceptable Cloud Coverage?
        Default is 10%
    4. Acceptable Area Coverage?
        Default is 90%
EXISTING:
    1. Order ID?
    2. Download the zip
    3. Unzip the zip
    4. Move the TIF to the correct folder
    5. Delete the zip
    6. Count them boats
    7. Add boat counts to the database
    8. save a labelled image for review
"""
import os
import PlanetUtils
import NNclassifier

def main():
    choice = input("New order or existing? (N/E): ")
    if choice == "N":
        new_order()
    elif choice == "E":
        existing_order()
    else:
        print("Invalid choice. Exiting.")
        exit()

def new_order():
    """
    Prompts the user and orders the image from Planet
    """
    print(PlanetUtils.get_aois())
    aoi = option_select(PlanetUtils.get_aois(), prompt="Select an AOI:")
    polygon = PlanetUtils.get_polygon(aoi)
    min_date = input("Minimum date (YYYY-MM-DD): ")
    max_date = input("Maximum date (YYYY-MM-DD): ")
    cloud_cover = float(input("Maximum cloud cover (0-1): "))
    area_cover = float(input("Minimum area cover (0-1): "))
    try:
        options = PlanetUtils.PlanetSearch(polygon_file=polygon, min_date=min_date, max_date=max_date, cloud_cover=cloud_cover, area_cover=area_cover)
        items = PlanetUtils.PlanetSelect(options)
        order = PlanetUtils.PlanetOrder(polygon_file=polygon, items=items, name="PeelIsland")
        order_id = order["id"]
        print("Order ID:", order_id)
        return order_id
    except Exception as e:
        print(e)
        exit()



def existing_order():
    """
    Checks an order and downloads the files
    """
    order_id = input("Order ID: ")
    # Download 
    try:
        PlanetUtils.PlanetDownload(order_id)
    except Exception as e:
        print(e)
        exit()

def option_select(options:list, prompt:str="Select an option:"):
    print(prompt)
    for i in range(len(options)):
        print(i+1, options[i])
    choice = input("Choice: ")
    if choice.isdigit():
        choice = int(choice)
        if choice > 0 and choice <= len(options):
            return options[choice-1]
        else:
            print("Invalid choice. Exiting.")
            exit()
        

if __name__ == "__main__":
    main()

