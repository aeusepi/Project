import pandas as pd
import numpy as np
from collections import defaultdict
#import AllTogetherSolns as s
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def clean_data(df,var2predict,col2exclude):
    '''
    INPUT
    df - pandas dataframe 
    var2predict - the variable to be estimated 
    col2exclude - columns that aren't useful for the anlysis

    OUTPUT
    X - A matrix holding all of the variables you want to consider when predicting the response
    y - the corresponding response vector
    
    This function cleans df using the following steps to produce X and y:
    1. Drop all the rows with no salaries
    2. Create X as all the columns that are not the Salary column
    3. Create y as the Salary column
    4. Drop the Salary, Respondent, and the ExpectedSalary columns from X
    5. For each numeric variable in X, fill the column with the mean value of the column.
    6. Create dummy columns for all the categorical variables in X, drop the original columns
    '''
    # Drop rows with missing salary values the data we want to predict
    df = df.dropna(subset=[var2predict], axis=0)
    y = df[var2predict]
    
    #Drop respondent and expected salary columns not needed for the analysis
    df = df.drop(col2exclude, axis=1)
    
    # Fill numeric columns with the mean - select the numeric variables 
    num_vars = df.select_dtypes(include=['float', 'int']).columns
    
    for col in num_vars:
        # loop throughout the columns and replace the NAs with the mean/mode/meadian 
        df[col].fillna((df[col].median()), inplace=True)
        
    # Dummy the categorical variables those are listed like objects
    cat_vars = df.select_dtypes(include=['object']).copy().columns
    for var in  cat_vars:
        # for each cat add dummy var, drop original column
        df = pd.concat([df.drop(var, axis=1), pd.get_dummies(df[var], prefix=var, prefix_sep='_', drop_first=True)], axis=1)
    
    X = df
    return X, y

def find_optimal_lm_mod(X, y, cutoffs, test_size = .30, random_state=42, plot=True):
    '''
    #cutoffs here pertains to the number of missing values allowed in the used columns.
    #Therefore, lower values for the cutoff provides more predictors in the model.
    INPUT
    X - pandas dataframe, X matrix
    y - pandas dataframe, response variable
    cutoffs - list of ints, cutoff for number of non-zero values in dummy categorical vars
    test_size - float between 0 and 1, default 0.3, determines the proportion of data as test data
    random_state - int, default 42, controls random state for train_test_split
    plot - boolean, default 0.3, True to plot result

    OUTPUT
    r2_scores_test - list of floats of r2 scores on the test data
    r2_scores_train - list of floats of r2 scores on the train data
    lm_model - model object from sklearn
    X_train, X_test, y_train, y_test - output from sklearn train test split used for optimal model
    '''
    r2_scores_test, r2_scores_train, num_feats, results = [], [], [], dict()
    for cutoff in cutoffs:

        #reduce X matrix selecting only the colums cointaing enough non zeros values in dummy cat var
        #selecting all those 
        reduce_X = X.iloc[:, np.where((X.sum() > cutoff) == True)[0]]
        
        #how many colums we are selecting
        num_feats.append(reduce_X.shape[1])
        
        #split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(reduce_X, y, test_size = test_size, random_state=random_state)

        #fit the model and obtain pred response
        lm_model = LinearRegression(normalize=True)
        lm_model.fit(X_train, y_train)
        y_test_preds = lm_model.predict(X_test)
        y_train_preds = lm_model.predict(X_train)

        #append the r2 value from the test set
        r2_scores_test.append(r2_score(y_test, y_test_preds))
        r2_scores_train.append(r2_score(y_train, y_train_preds))
        results[str(cutoff)] = r2_score(y_test, y_test_preds)

    if plot:
        plt.plot(num_feats, r2_scores_test, label="Test", alpha=.5)
        plt.plot(num_feats, r2_scores_train, label="Train", alpha=.5)
        plt.xlabel('Number of Features')
        plt.ylabel('Rsquared')
        plt.title('Rsquared by Number of Features')
        plt.legend(loc=1)
        plt.show()

    best_cutoff = max(results, key=results.get)

    #reduce X matrix
    reduce_X = X.iloc[:, np.where((X.sum() > int(best_cutoff)) == True)[0]]
    # how many columns have been selected
    num_feats.append(reduce_X.shape[1])

    #split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(reduce_X, y, test_size = test_size, random_state=random_state)

    #fit the model
    lm_model = LinearRegression(normalize=True)
    lm_model.fit(X_train, y_train)

    return r2_scores_test, r2_scores_train, lm_model, X_train, X_test, y_train, y_test

def coef_weights(lm_model, X_train):
    '''
    INPUT:
    coefficients - the coefficients of the linear model 
    X_train - the training data, so the column names can be used
    OUTPUT:
    coefs_df - a dataframe holding the coefficient, estimate, and abs(estimate)
    
    Provides a dataframe that can be used to understand the most influential coefficients
    in a linear model by providing the coefficient estimates along with the name of the 
    variable attached to the coefficient.
    '''
    coefs_df = pd.DataFrame()
    coefs_df['est_int'] = X_train.columns
    coefs_df['coefs'] = lm_model.coef_
    coefs_df['abs_coefs'] = np.abs(lm_model.coef_)
    coefs_df = coefs_df.sort_values('abs_coefs', ascending=False)
    
    return coefs_df

### Let's see what be the best number of features to use based on the test set performance
def find_optimal_rf_mod(X, y, cutoffs, test_size = .30, random_state=42, plot=True, param_grid=None):
    '''
    INPUT
    X - pandas dataframe, X matrix
    y - pandas dataframe, response variable
    cutoffs - list of ints, cutoff for number of non-zero values in dummy categorical vars
    test_size - float between 0 and 1, default 0.3, determines the proportion of data as test data
    random_state - int, default 42, controls random state for train_test_split
    plot - boolean, default 0.3, True to plot result
    kwargs - include the arguments you want to pass to the rf model
    param_grid are the parameters that neeed to be selected otherwise normal random forest
    params = {'n_estimators': [10, 100, 1000], 'max_depth': [1, 5, 10, 100]}

    OUTPUT
    r2_scores_test - list of floats of r2 scores on the test data
    r2_scores_train - list of floats of r2 scores on the train data
    rf_model - model object from sklearn
    X_train, X_test, y_train, y_test - output from sklearn train test split used for optimal model
    '''

    r2_scores_test, r2_scores_train, num_feats, results = [], [], [], dict()
    for cutoff in cutoffs:

        #reduce X matrix
        reduce_X = X.iloc[:, np.where((X.sum() > cutoff) == True)[0]]
        num_feats.append(reduce_X.shape[1])

        #split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(reduce_X, y, test_size = test_size, random_state=random_state)

        #fit the model and obtain pred response
        if param_grid==None:
            rf_model = RandomForestRegressor()  #no normalizing here, but could tune other hyperparameters

        else:
            rf_inst = RandomForestRegressor(n_jobs=-1, verbose=1)
            rf_model = GridSearchCV(rf_inst, param_grid, n_jobs=-1) 
            
        rf_model.fit(X_train, y_train)
        y_test_preds = rf_model.predict(X_test)
        y_train_preds = rf_model.predict(X_train)

        #append the r2 value from the test set
        r2_scores_test.append(r2_score(y_test, y_test_preds))
        r2_scores_train.append(r2_score(y_train, y_train_preds))
        results[str(cutoff)] = r2_score(y_test, y_test_preds)

    if plot:
        plt.plot(num_feats, r2_scores_test, label="Test", alpha=.5)
        plt.plot(num_feats, r2_scores_train, label="Train", alpha=.5)
        plt.xlabel('Number of Features')
        plt.ylabel('Rsquared')
        plt.title('Rsquared by Number of Features')
        plt.legend(loc=1)
        plt.show()
        
    best_cutoff = max(results, key=results.get)

    #reduce X matrix
    reduce_X = X.iloc[:, np.where((X.sum() > int(best_cutoff)) == True)[0]]
    num_feats.append(reduce_X.shape[1])

    #split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(reduce_X, y, test_size = test_size, random_state=random_state)

    #fit the model
    if param_grid==None:
        rf_model = RandomForestRegressor()  #no normalizing here, but could tune other hyperparameters

    else:
        rf_inst = RandomForestRegressor(n_jobs=-1, verbose=1)
        rf_model = GridSearchCV(rf_inst, param_grid, n_jobs=-1) 
    rf_model.fit(X_train, y_train)
     
    return r2_scores_test, r2_scores_train, rf_model, X_train, X_test, y_train, y_test



def convert_property_type(df):
    """
    Applies transformations to the property_type feature of the dataset.
    
    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data
    Output: df - the modified dataset containing the transformed features
            new_data.columns - the column names of the dummy variables
    """
    
    # Map the property type to the top 3 values and an Other bucket
    df['property_type'] = df['property_type'].map(
        {'House': 'House', 
         'Apartment': 'Apartment', 
         'Townhouse': 'Townhouse'}
    ).fillna('Other')
    
    # Create the dummy columns and append them to our dataframe
    new_data = pd.get_dummies(df[['property_type']])
    df[new_data.columns] = new_data
    
    # Remove the original categorical column
    df = df.drop(['property_type'], axis=1)
    
    return df, new_data.columns


def convert_room_type(df):
    """
    Applies transformations to the room_type feature of the dataset.
    
    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data
    Output: df - the modified dataset containing the transformed features
            new_data.columns - the column names of the dummy variables
    """
    # Create the dummy columns and append them to our dataframe
    new_data = pd.get_dummies(df[['room_type']])
    df[new_data.columns] = new_data
    
    # Remove the original categorical column
    df = df.drop(['room_type'], axis=1)
    
    return df, new_data.columns


def convert_bed_type(df):
    """
    Applies transformations to the bed_type feature of the dataset.
    
    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data
    Output: df - the modified dataset containing the transformed features
    """
    
    # We just care about whether the bed is REAL
    df['real_bed'] = df['bed_type'].map({'Real bed': 1}).fillna(0)
    
    # Remove the original categorical column
    df = df.drop(['bed_type'], axis=1)
    
    return df    

def convert_amenities(df):
    """
    Applies transformations to the amenities feature of the dataset.
    
    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data
    Output: df - the modified dataset containing the transformed features
    """
    
    # Look for presence of the string within the amenities column
    df['amenities_tv'] = df['amenities'].str.contains('TV')
    df['amenities_internet'] = df['amenities'].str.contains('Internet')
    df['amenities_wireless_internet'] = df['amenities'].str.contains('Wireless Internet')
    df['amenities_cable_tv'] = df['amenities'].str.contains('Cable TV')
    df['amenities_kitchen'] = df['amenities'].str.contains('Kitchen')
    df['amenities_elevator_in_building'] = df['amenities'].str.contains('Elevator in Building')
    df['amenities_wheelchair_accessible'] = df['amenities'].str.contains('Wheelchair Accessible')
    df['amenities_smoke_detector'] = df['amenities'].str.contains('Smoke Detector')
    df['amenities_pool'] = df['amenities'].str.contains('Pool')
    df['amenities_free_parking_on_premises'] = df['amenities'].str.contains('Free Parking on Premises')
    df['amenities_air_conditioning'] = df['amenities'].str.contains('Air Conditioning')
    df['amenities_heating'] = df['amenities'].str.contains('Heating')
    df['amenities_pets_live_on_this_property'] = df['amenities'].str.contains('Pets live on this property')
    df['amenities_washer'] = df['amenities'].str.contains('Washer')
    df['amenities_breakfast'] = df['amenities'].str.contains('Breakfast')
    df['amenities_buzzer_wireless_intercom'] = df['amenities'].str.contains('Buzzer/Wireless Intercom')
    df['amenities_pets_allowed'] = df['amenities'].str.contains('Pets Allowed')
    df['amenities_carbon_monoxide_detector'] = df['amenities'].str.contains('Carbon Monoxide Detector')
    df['amenities_gym'] = df['amenities'].str.contains('Gym')
    df['amenities_dryer'] = df['amenities'].str.contains('Dryer')
    df['amenities_indoor_fireplace'] = df['amenities'].str.contains('Indoor Fireplace')
    df['amenities_family_kid_friendly'] = df['amenities'].str.contains('Family/Kid Friendly')
    df['amenities_dogs'] = df['amenities'].str.contains('Dog(s)')
    df['amenities_essentials'] = df['amenities'].str.contains('Essentials')
    df['amenities_cats'] = df['amenities'].str.contains('Cat(s)')
    df['amenities_hot_tub'] = df['amenities'].str.contains('Hot Tub')
    df['amenities_shampoo'] = df['amenities'].str.contains('Shampoo')
    df['amenities_first_aid_kit'] = df['amenities'].str.contains('First Aid Kit')
    df['amenities_smoking_allowed'] = df['amenities'].str.contains('Smoking Allowed')
    df['amenities_fire_extinguisher'] = df['amenities'].str.contains('Fire Extinguisher')
    df['amenities_doorman'] = df['amenities'].str.contains('Doorman')
    df['amenities_washer_dryer'] = df['amenities'].str.contains('Washer / Dryer')
    df['amenities_safety_card'] = df['amenities'].str.contains('Safety Card')
    df['amenities_suitable_for_events'] = df['amenities'].str.contains('Suitable for Events')
    df['amenities_other_pets'] = df['amenities'].str.contains('Other pet(s)')
    df['amenities_24_hour_check_in'] = df['amenities'].str.contains('24-Hour Check-in')
    df['amenities_hangers'] = df['amenities'].str.contains('Hangers')
    df['amenities_laptop_friendly_workspace'] = df['amenities'].str.contains('Laptop Friendly Workspace')
    df['amenities_iron'] = df['amenities'].str.contains('Iron')
    df['amenities_hair_dryer'] = df['amenities'].str.contains('Hair Dryer')
    df['amenities_lock_on_bedroom_door'] = df['amenities'].str.contains('Lock on Bedroom Door')
    
    # Remove the original categorical column
    df = df.drop(['amenities'], axis=1)
    
    return df    

def convert_neighbourhood_cleansed(df):
    """
    Applies transformations to the neighbourhood_cleansed feature of the dataset.
    
    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data
    Output: df - the modified dataset containing the transformed features
            new_data.columns - the column names of the dummy variables
    """
    
    # Create dummies on the column
    new_data = pd.get_dummies(df[['neighbourhood_cleansed']])
    df[new_data.columns] = new_data
    
    # We will keep the neighbourhood_cleansed column for future use
    return df, new_data.columns

def convert_neighbourhood_group_cleansed(df):
    """
    Applies transformations to the neighbourhood_group_cleansed feature of the dataset.
    
    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data
    Output: df - the modified dataset containing the transformed features
            new_data.columns - the column names of the dummy variables
    """
    
    # Create dummies on the column
    new_data = pd.get_dummies(df[['neighbourhood_group_cleansed']])
    df[new_data.columns] = new_data
    
    # We will keep the neighbourhood_cleansed column for future use
    return df, new_data.columns   


def convert_is_location_exact(df):
    """
    Input: df - a dataset containing the unprocessed column's data
    Output: df - the modified dataset containing the transformed features
    """
    
    df['is_location_exact'] = df['is_location_exact'].map({'t':1}).fillna(0)
    
    return df     


def convert_price(df):
    """
    Applies transformations to the price feature of the dataset.
    
    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data
    Output: df - the modified dataset containing the transformed features
    """
    
    # Convert the money variable into a numeric variable
    df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
    
    return df

def convert_weekly_price(df):
    """
    Applies transformations to the weekly_price feature of the dataset.
    
    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data
    Output: df - the modified dataset containing the transformed features
    """
    
    # Convert the money variable into a numeric variable
    df['weekly_price'] = df['weekly_price'].replace('[\$,]', '', regex=True).astype(float)
    
    # Note that this code is assuming that price has already been converted
    # so we will run this code after convert_price().
    df['weekly_price_ratio'] = df['weekly_price'] / df['price']
        
    # Boolean feature to indicate that a weekly price has been set
    df['has_weekly_price'] = ~df['weekly_price'].isnull()
    
    # If there is no weekly price then set the ratio to 7, since
    # this would imply the regular price
    df['weekly_price_ratio'] = df['weekly_price_ratio'].fillna(7)
    df['weekly_price'] = df['weekly_price'].fillna(7*df['price'])
    
    return df

def convert_monthly_price(df):
    """
    Applies transformations to the weekly_price feature of the dataset.
    
    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data
    Output: df - the modified dataset containing the transformed features
    """
    
    # Convert the money variable into a numeric variable
    df['monthly_price'] = df['monthly_price'].replace('[\$,]', '', regex=True).astype(float)
    
    # Note that this code is assuming that price has already been converted
    # so we will run this code after convert_price().
    df['monthly_price_ratio'] = df['monthly_price'] / df['price']
        
    # Boolean feature to indicate that a weekly price has been set
    df['has_monthly_price'] = ~df['monthly_price'].isnull()
    
    # If there is no monthly price then set the ratio to 365/12, since 
    # this would imply the regular price.
    df['monthly_price_ratio'] = df['monthly_price_ratio'].fillna(365./12.)
    df['monthly_price'] = df['monthly_price'].fillna(365./12.*df['price'])
    
    return df    

def convert_security_deposit(df):
    """
    Applies transformations to the security_deposit feature of the dataset.
    
    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data
    Output: df - the modified dataset containing the transformed features
    """
    
    # Convert the money variable into a numeric variable
    df['security_deposit'] = df['security_deposit'].replace('[\$,]', '', regex=True).astype(float)
    
    # Note that this code is assuming that price has already been converted
    # so we will run this code after convert_price().
    df['security_deposit_ratio'] = df['security_deposit'] / df['price']
        
    # Boolean feature to indicate that a weekly price has been set
    df['has_security_deposit'] = ~df['security_deposit'].isnull()
    
    # If there is no security_deposit then set the ratio to zero
    # This assumes that there is no security deposit
    df['security_deposit_ratio'] = df['security_deposit_ratio'].fillna(0)
    df['security_deposit'] = df['security_deposit'].fillna(0)
    
    return df  

def convert_cleaning_fee(df):
    """
    Applies transformations to the cleaning_fee feature of the dataset.
    
    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data
    Output: df - the modified dataset containing the transformed features
    """
    
    # Convert the money variable into a numeric variable
    df['cleaning_fee'] = df['cleaning_fee'].replace('[\$,]', '', regex=True).astype(float)
    
    # Note that this code is assuming that price has already been converted
    # so we will run this code after convert_price().
    df['cleaning_fee_ratio'] = df['cleaning_fee'] / df['price']
    
    # Convert the money variable into a numeric variable
    df['cleaning_fee'] = df['cleaning_fee'].replace('[\$,]', '', regex=True).astype(float)
    
    # If there is no cleaning_fee then set the ratio to zero
    # This assumes that there is no cleaning fee
    df['cleaning_fee_ratio'] = df['cleaning_fee_ratio'].fillna(0)
    df['cleaning_fee'] = df['cleaning_fee'].fillna(0)
    
    return df


def convert_extra_people(df):
    """
    Applies transformations to the extra_people feature of the dataset.
    
    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data
    Output: df - the modified dataset containing the transformed features
    """
    
    # Convert the money variable into a numeric variable
    df['extra_people'] = df['extra_people'].replace('[\$,]', '', regex=True).astype(float)
    
    # Note that this code is assuming that price has already been converted
    # so we will run this code after convert_price().
    df['extra_people_ratio'] = df['extra_people'] / df['price']
    
    # Convert the money variable into a numeric variable
    df['extra_people'] = df['extra_people'].replace('[\$,]', '', regex=True).astype(float)
    
    # If there is no extra_people then set the ratio to zero
    # This assumes that there is no extra people fee
    df['extra_people_ratio'] = df['extra_people_ratio'].fillna(0)
    df['extra_people'] = df['extra_people'].fillna(0)
    
    return df 

def convert_instant_bookable(df):
    """
    Applies transformations to the instant_bookable feature of the dataset.
    
    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data
    Output: df - the modified dataset containing the transformed features
    """
    
    df['instant_bookable'] = df['instant_bookable'].map({'t': 1}).fillna(0)
    
    return df

def convert_require_guest_profile_picture(df):
    """
    Applies transformations to the require_guest_profile_picture feature of the dataset.
    
    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data
    Output: df - the modified dataset containing the transformed features
    """
    
    df['require_guest_profile_picture'] = df['require_guest_profile_picture'].map({'t': 1}).fillna(0)
    
    return df

def convert_require_guest_phone_verification(df):
    """
    Applies transformations to the require_guest_phone_verification feature of the dataset.
    
    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data
    Output: df - the modified dataset containing the transformed features
    """
    
    df['require_guest_phone_verification'] = df['require_guest_phone_verification'].map({'t': 1}).fillna(0)
    
    return df


def convert_cancellation_policy(df):
    """
    Applies transformations to the cancellation_policy feature of the dataset.
    
    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data
    Output: df - the modified dataset containing the transformed features
            new_data.columns - the column names of the dummy variables
    """
    
    # Create dummies on the column
    new_data = pd.get_dummies(df[['cancellation_policy']])
    df[new_data.columns] = new_data
    
    # We will keep the cancellation_policy column for future use
    df = df.drop(['cancellation_policy'], axis=1)
    
    return df, new_data.columns 

def convert_host_about(df):
    """
    Applies transformations to the host_about feature of the dataset.
    
    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data
    Output: df - the modified dataset containing the transformed features
    """
    
    # Ascertain whether the value is null
    df['has_host_about'] = ~df['host_about'].isnull()
    
    # Drop the original column
    df = df.drop(['host_about'], axis=1)
    
    return df       


def convert_host_since(df):
    """
    Applies transformations to the host_since feature of the dataset.
    
    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data
    Output: df - the modified dataset containing the transformed features
    """
    
    # Subtract the dates to get the number of days
    df['days_as_host'] = (pd.to_datetime(df['last_scraped']) - pd.to_datetime(df['host_since'])) / np.timedelta64(1, 'D')
    
    # Drop the original column
    df = df.drop([ 'last_scraped'], axis=1)
    
    return df 

def convert_host_response_time(df):
    """
    Applies transformations to the host_response_time feature of the dataset.
    
    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data
    Output: df - the modified dataset containing the transformed features
    """

    # Map the values
    df['host_response_time'] = df['host_response_time'].map(
        {'within an hour': 1, 'within a few hours': 2, 'within a day': 3}
    ).fillna(4)
    
    return df  

def convert_host_location(df):
    """
    Applies transformations to the host_location feature of the dataset.
    
    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data
    Output: df - the modified dataset containing the transformed features
    """

    # Search for 'Seattle' in the host_location field
    df['host_in_seattle'] = df['host_location'].str.contains('Seattle')
    
    # Drop the original column
    df = df.drop(['host_location'], axis=1)
    
    return df 

def convert_host_response_time(df):
    """
    Applies transformations to the host_response_time feature of the dataset.
    
    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data
    Output: df - the modified dataset containing the transformed features
    """

    # Map the values
    df['host_response_time'] = df['host_response_time'].map(
        {'within an hour': 1, 'within a few hours': 2, 'within a day': 3}
    ).fillna(4)
    
    return df   

def convert_host_response_rate(df):
    """
    Applies transformations to the host_response_rate feature of the dataset.
    
    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data
    Output: df - the modified dataset containing the transformed features
    """
    
    # Convert to float
    df['host_response_rate'] = df['host_response_rate'].str.replace(r'%', r'.0').astype('float') / 100.0
    
    # Fill missing values with zero
    df['host_response_rate'] = df['host_response_rate'].fillna(0)
    
    return df

def convert_host_neighbourhood(df):
    """
    Applies transformations to the host_neighbourhood feature of the dataset.
    
    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data
    Output: df - the modified dataset containing the transformed features
    """
    
    # Lookup against all 3 neighbourhood columns
    df['host_in_neighbourhood'] = np.where(
        df['host_neighbourhood'] == df['neighbourhood'], True, 
        np.where(
            df['host_neighbourhood'] == df['neighbourhood_cleansed'], True,
            np.where(
                df['host_neighbourhood'] == df['neighbourhood_group_cleansed'], True, False
            )
        )
    )
    
    # Remove the original columns
    df = df.drop(['host_neighbourhood', 'neighbourhood', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed'], axis=1)
    
    return df  

def convert_host_verifications(df):
    """
    Applies transformations to the host_verifications feature of the dataset.
    
    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data
    Output: df - the modified dataset containing the transformed features
    """
    
    # Lookup the substring and set boolean value as column
    df['host_verif_email'] = df['host_verifications'].str.contains('email')
    df['host_verif_kba'] = df['host_verifications'].str.contains('kba')
    df['host_verif_phone'] = df['host_verifications'].str.contains('phone')
    df['host_verif_reviews'] = df['host_verifications'].str.contains('reviews')
    df['host_verif_jumio'] = df['host_verifications'].str.contains('jumio')
    df['host_verif_facebook'] = df['host_verifications'].str.contains('facebook')
    df['host_verif_linkedin'] = df['host_verifications'].str.contains('linkedin')
    df['host_verif_google'] = df['host_verifications'].str.contains('google')
    df['host_verif_manual_online'] = df['host_verifications'].str.contains('manual_online')
    df['host_verif_manual_offline'] = df['host_verifications'].str.contains('manual_offline')
    df['host_verif_sent_id'] = df['host_verifications'].str.contains('sent_id')
    df['host_verif_amex'] = df['host_verifications'].str.contains('amex')
    df['host_verif_weibo'] = df['host_verifications'].str.contains('weibo')
    df['host_verif_photographer'] = df['host_verifications'].str.contains('photographer')
    
    # Drop the original column
    df = df.drop(['host_verifications'], axis=1)
    
    return df

def convert_host_is_superhost(df):
    """
    Applies transformations to the host_is_superhost feature of the dataset.
    
    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data
    Output: df - the modified dataset containing the transformed features
    """
    
    df['host_is_superhost'] = df['host_is_superhost'].map({'t': 1}).fillna(0)
    
    return df  

def convert_host_has_profile_pic(df):
    """
    Applies transformations to the host_has_profile_pic feature of the dataset.
    
    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data
    Output: df - the modified dataset containing the transformed features
    """
    
    df['host_has_profile_pic'] = df['host_has_profile_pic'].map({'t': 1}).fillna(0)
    
    return df

def convert_host_identity_verified(df):
    """
    Applies transformations to the host_is_superhost feature of the dataset.
    
    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data
    Output: df - the modified dataset containing the transformed features
    """
    
    df['host_identity_verified'] = df['host_identity_verified'].map({'t': 1}).fillna(0)
    
    return df