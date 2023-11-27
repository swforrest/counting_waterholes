import requests
import os
import yaml
import json
from dotenv import load_dotenv
import zipfile
load_dotenv()

# api key is either set in the environment variable or config.yml
config = yaml.load(open('config.yml', 'r'), Loader=yaml.FullLoader)
API_KEY = os.environ.get('PLANET_API_KEY', config['planet']['api_key'])
if API_KEY is None or API_KEY == 'ENV':
    raise Exception('Planet API key not found in environment variable or config.yml')

def PlanetSearch(search_path):
    """
    Search a given area of interest for Planet imagery
    :param search_path: path to a json file containing search body
    :requires: Planet API key be set in environment variable
    :return: a list of Planet items (json result from API)
    """
    # 1. Get the json file as text
    with open(search_path, 'r') as f:
        search_body = f.read()
    # 2. Get the api key from the environment variable
    api_key = os.environ.get('PLANET_API_KEY')
    # 3. Set the headers
    headers = {'content-type': 'application/json'}
    auth = (API_KEY, '')
    # 4. Send the request
    response = requests.post('https://api.planet.com/data/v1/quick-search',
                                data=search_body, headers=headers, auth=auth)
    # 5. Return the features in the response
    # NOTE: there is pagination, so this will only return the first page (a lot of items)
    # If required, can add request pagination but hopefully not nessecary
    # write the response to a file
    with open('response.json', 'w') as f:
        f.write(response.text)
    return response.json()['features']

def PlanetSelect(items:list):
    """
    Select a subset of items from a search result
    :param items: the list of items to select from
    :return: list of selected items
    """
    #NOTE: currently selects the most recent item from the list
    # sort by aquired
    items.sort(key=lambda x: x['properties']['acquired'])
    selected = [items[-1]]
    # pretty print
    print(json.dumps(selected, indent=4))
    return selected



def PlanetOrder(items:list, polygon_file:str, name:str):
    """
    Order a given search result.
    :param itemIDs: a list of item IDs to order
    :param polygon_file: a geojson file containing the area of interest to clip to
        must be of format: {"type": "Polygon", "coordinates": [[[lon, lat], ...]]}
    :param name: the name of the order
    :requires: Planet API key be set in environment variable
    :return: a list of order IDs
    """
    pids = [item['id'] for item in items]
    products = [{
        "item_ids": pids,
        "item_type": "PSScene",
        "product_bundle": "analytic_sr_udm2"
        }]
    # get the polygon
    polygon = None
    with open(polygon_file, 'r') as f:
        polygon = json.load(f)
    # create the order body
    order_body = {
        "name": name,
        "products": products,
        "tools": [
            {
                "clip": {
                    "aoi": polygon
                },
            },
            {
                "harmonize": {
                    "target_sensor": "Sentinel-2"
                },
            },
            {
                "composite": {}
            }
        ],
        "notifications": {
            "email": True
            },
        "metadata": {
            "stac": {}
        },
        "order_type": "partial",
        "delivery": {
            "archive_filename": "planet_image_api",
            "archive_type": "zip",
            "single_archive": True,
        }
    }
    # set the headers
    headers = {'content-type': 'application/json'}
    print(json.dumps(order_body, indent=4))
    # make the request
    response = requests.post('https://api.planet.com/compute/ops/orders/v2',
                                data=json.dumps(order_body), headers=headers, auth=(API_KEY, ''))

    #TODO: Handle response codes

    # write the response to a file
    with open('order_response.json', 'w') as f:
        f.write(response.text)
    # return that
    return response.json()



def PlanetCheckOrder(orderID:str):
    """
    Check the status of a given order
    :param orderID: the order ID to check
    :requires: Planet API key be set in environment variable
    :return: the status of the order
    """
    uri = f"https://api.planet.com/compute/ops/orders/v2/{orderID}"
    response = requests.get(uri, auth=(API_KEY, ''))
    # write the response to a file
    with open('order_status.json', 'w') as f:
        f.write(response.text)
    return response.json()['state']


def PlanetDownload(orderID:str, downloadPath:str='./temp/download.zip'):
    """
    Download a given order and unzip it to a given path
    """
    uri = f"https://api.planet.com/compute/ops/orders/v2/{orderID}"
    response = requests.get(uri, auth=(API_KEY, ''))
    # grab the location uri
    links = response.json()['_links']['results']
    download_link = [link['location'] for link in links if "image" in link['name']][0]
    # make the directory if it doesn't exist
    os.makedirs(os.path.dirname(downloadPath), exist_ok=True)
    # download the file
    progress = 0
    with requests.get(download_link, stream=True) as r:
        with open(downloadPath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                if chunk:
                    f.write(chunk)
                    progress += len(chunk)
                    print(f"Downloaded {progress} bytes", end='\r') 
    # unzip the file
    with zipfile.ZipFile(downloadPath, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(downloadPath))
    # delete the zip file
    os.remove(downloadPath)
    return downloadPath

if __name__ == "__main__":
    # prompt for what to do
    print('What would you like to do?')
    print('1. Search')
    print('2. Select')
    print('3. Order')
    print('4. Check Order')
    print('5. Download')
    choice = input('Enter a number: ')
    if choice == '1':
        # prompt for search path
        search_path = input('Enter the path to the search json: ')
        # search
        result = PlanetSearch(search_path)
        # print the result
        print(result)
    elif choice == '2':
        search_path = input('Enter the path to the search json: ')
        items = PlanetSearch(search_path)
        if items is not None:
            PlanetSelect(items)
    elif choice == '3':
        search_path = input('Enter the path to the search json: ')
        items = PlanetSearch(search_path)
        if items is not None:
            items = PlanetSelect(items)
        polygon_file = input('Enter the path to the polygon file: ')
        name = input('Enter the name of the order: ')
        PlanetOrder(items, polygon_file, name)
    elif choice == '4':
        orderID = input('Enter the order ID: ')
        PlanetCheckOrder(orderID)
    elif choice == '5':
        orderID = input('Enter the order ID: ')
        downloadPath = input('Enter the path to download to: ')
        PlanetDownload(orderID, downloadPath)
    else:
        print('Not implemented yet')
        
