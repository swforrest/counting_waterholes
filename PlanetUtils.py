import requests
import os
import yaml
import json
from dotenv import load_dotenv
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
    selected = items[-1]
    # pretty print
    print(json.dumps(selected, indent=4))
    return selected



def PlanetOrder(items:list, polygon_file:str, name:str):
    """
    Order a given search result.
    :param itemIDs: a list of item IDs to order
    :param polygon_file: a geojson file containing the area of interest to clip to
    :param name: the name of the order
    :requires: Planet API key be set in environment variable
    :return: a list of order IDs
    """
    pass

def PlanetCheckOrder(orderID:str):
    """
    Check the status of a given order
    :param orderID: the order ID to check
    :requires: Planet API key be set in environment variable
    :return: the status of the order
    """
    pass

def PlanetDownload(orderID:str, downloadPath:str):
    pass


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
        # prompt for search path or response path
        items = None
        c2 = input('1. Search path\n2. Response path\nEnter a number: ')
        if c2 == '1':
            search_path = input('Enter the path to the search json: ')
            items = PlanetSearch(search_path)
        elif c2 == '2':
            response_path = input('Enter the path to the response json: ')
            with open(response_path, 'r') as f:
                items = json.load(f)['features']
        if items is not None:
            PlanetSelect(items)
    else:
        print('Not implemented yet')
        
