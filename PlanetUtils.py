import requests
import os
import yaml

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
    return response.json()['features']

def PlanetOrder(itemIDs:list):
    """
    Order a given search result.
    :param itemIDs: a list of item IDs to order
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
