import pickle
import json  , os
import numpy as np 

locations_list = [] 
data_columns = None 
model = None

def get_predicted_price( bhk,sqft , bath,location ):
    # loc_index
    try:
        loc_index = data_columns.index(location) ;
    except:
        print("wrong location name")
        return 0

    row = np.zeros(len(data_columns) )
    
    row[0] = bhk ;
    row[1] = sqft ;
    row[2] = bath ;
    row[loc_index] = 1 ;

    price =  model.predict( [ row]  )[0]
    return price

def get_location_names():
    return locations_list

def load_saved_columns():
    global locations_list , data_columns , model
    
    with open("./BHP Server/server/artifacts/columns.json", "r") as f :
      data_columns =   json.load(f)['data_columns']

      locations_list = data_columns[3:]

    with open("./BHP Server/server/artifacts/benglore_house_prices_model.pickle" , 'rb') as f:
       model = pickle.load(f)
    
    print("loading saved artifacts ... done")               


if __name__ == '__main__':
    print("starting Python Flask Server for House Prediction......")
    load_saved_columns()

    price  = get_predicted_price(4 ,2850.0 , 4.0  ,  "indira nagar" )
    print("price : %.2f" % price)
    price  = get_predicted_price(4 ,2850.0 , 4.0  ,  "1st block jayanagar" )
    print("price : %.2f" % price)


