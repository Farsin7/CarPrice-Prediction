{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.linear_model import Lasso\n",
    "from sklearn import metrics"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Data Collection and Processing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Car_Name</th>\n",
       "      <th>Year</th>\n",
       "      <th>Selling_Price</th>\n",
       "      <th>Present_Price</th>\n",
       "      <th>Kms_Driven</th>\n",
       "      <th>Fuel_Type</th>\n",
       "      <th>Seller_Type</th>\n",
       "      <th>Transmission</th>\n",
       "      <th>Owner</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>ritz</td>\n",
       "      <td>2014</td>\n",
       "      <td>3.35</td>\n",
       "      <td>5.59</td>\n",
       "      <td>27000</td>\n",
       "      <td>Petrol</td>\n",
       "      <td>Dealer</td>\n",
       "      <td>Manual</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>sx4</td>\n",
       "      <td>2013</td>\n",
       "      <td>4.75</td>\n",
       "      <td>9.54</td>\n",
       "      <td>43000</td>\n",
       "      <td>Diesel</td>\n",
       "      <td>Dealer</td>\n",
       "      <td>Manual</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>ciaz</td>\n",
       "      <td>2017</td>\n",
       "      <td>7.25</td>\n",
       "      <td>9.85</td>\n",
       "      <td>6900</td>\n",
       "      <td>Petrol</td>\n",
       "      <td>Dealer</td>\n",
       "      <td>Manual</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>wagon r</td>\n",
       "      <td>2011</td>\n",
       "      <td>2.85</td>\n",
       "      <td>4.15</td>\n",
       "      <td>5200</td>\n",
       "      <td>Petrol</td>\n",
       "      <td>Dealer</td>\n",
       "      <td>Manual</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>swift</td>\n",
       "      <td>2014</td>\n",
       "      <td>4.60</td>\n",
       "      <td>6.87</td>\n",
       "      <td>42450</td>\n",
       "      <td>Diesel</td>\n",
       "      <td>Dealer</td>\n",
       "      <td>Manual</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Car_Name  Year  Selling_Price  Present_Price  Kms_Driven Fuel_Type  \\\n",
       "0     ritz  2014           3.35           5.59       27000    Petrol   \n",
       "1      sx4  2013           4.75           9.54       43000    Diesel   \n",
       "2     ciaz  2017           7.25           9.85        6900    Petrol   \n",
       "3  wagon r  2011           2.85           4.15        5200    Petrol   \n",
       "4    swift  2014           4.60           6.87       42450    Diesel   \n",
       "\n",
       "  Seller_Type Transmission  Owner  \n",
       "0      Dealer       Manual      0  \n",
       "1      Dealer       Manual      0  \n",
       "2      Dealer       Manual      0  \n",
       "3      Dealer       Manual      0  \n",
       "4      Dealer       Manual      0  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "car_dataset = pd.read_csv('car data.csv')\n",
    "car_dataset.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Car_Name</th>\n",
       "      <th>Year</th>\n",
       "      <th>Selling_Price</th>\n",
       "      <th>Present_Price</th>\n",
       "      <th>Kms_Driven</th>\n",
       "      <th>Fuel_Type</th>\n",
       "      <th>Seller_Type</th>\n",
       "      <th>Transmission</th>\n",
       "      <th>Owner</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>296</th>\n",
       "      <td>city</td>\n",
       "      <td>2016</td>\n",
       "      <td>9.50</td>\n",
       "      <td>11.6</td>\n",
       "      <td>33988</td>\n",
       "      <td>Diesel</td>\n",
       "      <td>Dealer</td>\n",
       "      <td>Manual</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>297</th>\n",
       "      <td>brio</td>\n",
       "      <td>2015</td>\n",
       "      <td>4.00</td>\n",
       "      <td>5.9</td>\n",
       "      <td>60000</td>\n",
       "      <td>Petrol</td>\n",
       "      <td>Dealer</td>\n",
       "      <td>Manual</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>298</th>\n",
       "      <td>city</td>\n",
       "      <td>2009</td>\n",
       "      <td>3.35</td>\n",
       "      <td>11.0</td>\n",
       "      <td>87934</td>\n",
       "      <td>Petrol</td>\n",
       "      <td>Dealer</td>\n",
       "      <td>Manual</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>299</th>\n",
       "      <td>city</td>\n",
       "      <td>2017</td>\n",
       "      <td>11.50</td>\n",
       "      <td>12.5</td>\n",
       "      <td>9000</td>\n",
       "      <td>Diesel</td>\n",
       "      <td>Dealer</td>\n",
       "      <td>Manual</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>300</th>\n",
       "      <td>brio</td>\n",
       "      <td>2016</td>\n",
       "      <td>5.30</td>\n",
       "      <td>5.9</td>\n",
       "      <td>5464</td>\n",
       "      <td>Petrol</td>\n",
       "      <td>Dealer</td>\n",
       "      <td>Manual</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    Car_Name  Year  Selling_Price  Present_Price  Kms_Driven Fuel_Type  \\\n",
       "296     city  2016           9.50           11.6       33988    Diesel   \n",
       "297     brio  2015           4.00            5.9       60000    Petrol   \n",
       "298     city  2009           3.35           11.0       87934    Petrol   \n",
       "299     city  2017          11.50           12.5        9000    Diesel   \n",
       "300     brio  2016           5.30            5.9        5464    Petrol   \n",
       "\n",
       "    Seller_Type Transmission  Owner  \n",
       "296      Dealer       Manual      0  \n",
       "297      Dealer       Manual      0  \n",
       "298      Dealer       Manual      0  \n",
       "299      Dealer       Manual      0  \n",
       "300      Dealer       Manual      0  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "car_dataset.tail()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(301, 9)"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "car_dataset.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 301 entries, 0 to 300\n",
      "Data columns (total 9 columns):\n",
      " #   Column         Non-Null Count  Dtype  \n",
      "---  ------         --------------  -----  \n",
      " 0   Car_Name       301 non-null    object \n",
      " 1   Year           301 non-null    int64  \n",
      " 2   Selling_Price  301 non-null    float64\n",
      " 3   Present_Price  301 non-null    float64\n",
      " 4   Kms_Driven     301 non-null    int64  \n",
      " 5   Fuel_Type      301 non-null    object \n",
      " 6   Seller_Type    301 non-null    object \n",
      " 7   Transmission   301 non-null    object \n",
      " 8   Owner          301 non-null    int64  \n",
      "dtypes: float64(2), int64(3), object(4)\n",
      "memory usage: 21.3+ KB\n"
     ]
    }
   ],
   "source": [
    "car_dataset.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Car_Name         0\n",
       "Year             0\n",
       "Selling_Price    0\n",
       "Present_Price    0\n",
       "Kms_Driven       0\n",
       "Fuel_Type        0\n",
       "Seller_Type      0\n",
       "Transmission     0\n",
       "Owner            0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "car_dataset.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Petrol    239\n",
      "Diesel     60\n",
      "CNG         2\n",
      "Name: Fuel_Type, dtype: int64\n",
      "Dealer        195\n",
      "Individual    106\n",
      "Name: Seller_Type, dtype: int64\n",
      "Manual       261\n",
      "Automatic     40\n",
      "Name: Transmission, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "#Distribution of categorical data\n",
    "\n",
    "print (car_dataset.Fuel_Type.value_counts())\n",
    "print (car_dataset.Seller_Type.value_counts())\n",
    "print (car_dataset.Transmission.value_counts())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Encoding the Categorical Data\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "car_dataset.replace({'Fuel_Type':{'Petrol':0, 'Diesel':1, 'CNG':2}}, inplace=True)\n",
    "car_dataset.replace({'Seller_Type':{'Dealer':0, 'Individual':1}}, inplace=True)\n",
    "car_dataset.replace({'Transmission':{'Manual':0, 'Automatic':1,}}, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Car_Name</th>\n",
       "      <th>Year</th>\n",
       "      <th>Selling_Price</th>\n",
       "      <th>Present_Price</th>\n",
       "      <th>Kms_Driven</th>\n",
       "      <th>Fuel_Type</th>\n",
       "      <th>Seller_Type</th>\n",
       "      <th>Transmission</th>\n",
       "      <th>Owner</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>ritz</td>\n",
       "      <td>2014</td>\n",
       "      <td>3.35</td>\n",
       "      <td>5.59</td>\n",
       "      <td>27000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>sx4</td>\n",
       "      <td>2013</td>\n",
       "      <td>4.75</td>\n",
       "      <td>9.54</td>\n",
       "      <td>43000</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>ciaz</td>\n",
       "      <td>2017</td>\n",
       "      <td>7.25</td>\n",
       "      <td>9.85</td>\n",
       "      <td>6900</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>wagon r</td>\n",
       "      <td>2011</td>\n",
       "      <td>2.85</td>\n",
       "      <td>4.15</td>\n",
       "      <td>5200</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>swift</td>\n",
       "      <td>2014</td>\n",
       "      <td>4.60</td>\n",
       "      <td>6.87</td>\n",
       "      <td>42450</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Car_Name  Year  Selling_Price  Present_Price  Kms_Driven  Fuel_Type  \\\n",
       "0     ritz  2014           3.35           5.59       27000          0   \n",
       "1      sx4  2013           4.75           9.54       43000          1   \n",
       "2     ciaz  2017           7.25           9.85        6900          0   \n",
       "3  wagon r  2011           2.85           4.15        5200          0   \n",
       "4    swift  2014           4.60           6.87       42450          1   \n",
       "\n",
       "   Seller_Type  Transmission  Owner  \n",
       "0            0             0      0  \n",
       "1            0             0      0  \n",
       "2            0             0      0  \n",
       "3            0             0      0  \n",
       "4            0             0      0  "
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "car_dataset.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Splitting into Training and Test Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = car_dataset.drop(['Car_Name', 'Selling_Price'], axis=1)\n",
    "Y = car_dataset['Selling_Price']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "     Year  Present_Price  Kms_Driven  Fuel_Type  Seller_Type  Transmission  \\\n",
      "0    2014           5.59       27000          0            0             0   \n",
      "1    2013           9.54       43000          1            0             0   \n",
      "2    2017           9.85        6900          0            0             0   \n",
      "3    2011           4.15        5200          0            0             0   \n",
      "4    2014           6.87       42450          1            0             0   \n",
      "..    ...            ...         ...        ...          ...           ...   \n",
      "296  2016          11.60       33988          1            0             0   \n",
      "297  2015           5.90       60000          0            0             0   \n",
      "298  2009          11.00       87934          0            0             0   \n",
      "299  2017          12.50        9000          1            0             0   \n",
      "300  2016           5.90        5464          0            0             0   \n",
      "\n",
      "     Owner  \n",
      "0        0  \n",
      "1        0  \n",
      "2        0  \n",
      "3        0  \n",
      "4        0  \n",
      "..     ...  \n",
      "296      0  \n",
      "297      0  \n",
      "298      0  \n",
      "299      0  \n",
      "300      0  \n",
      "\n",
      "[301 rows x 7 columns]\n"
     ]
    }
   ],
   "source": [
    "print(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0       3.35\n",
      "1       4.75\n",
      "2       7.25\n",
      "3       2.85\n",
      "4       4.60\n",
      "       ...  \n",
      "296     9.50\n",
      "297     4.00\n",
      "298     3.35\n",
      "299    11.50\n",
      "300     5.30\n",
      "Name: Selling_Price, Length: 301, dtype: float64\n"
     ]
    }
   ],
   "source": [
    "print(Y)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Spilitting Training and Test Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Model Training\n",
    "    Linear Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "lin_reg_model = LinearRegression()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LinearRegression()"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "lin_reg_model.fit(X_train, Y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Model Evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "training_data_prediction = lin_reg_model.predict(X_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "R Squared Error -  0.8799451660493698\n"
     ]
    }
   ],
   "source": [
    "#R squared error\n",
    "error_score = metrics.r2_score(Y_train, training_data_prediction)\n",
    "print(\"R Squared Error - \",error_score)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Visualizing Actual & Predicted Prices"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAX4AAAEWCAYAAABhffzLAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8vihELAAAACXBIWXMAAAsTAAALEwEAmpwYAAAfbUlEQVR4nO3dfZRddX3v8fcnkwEmggYkIAwJwUqjCJJwp0BNtYAPwQdgSuUhym2814petZWWpgbLUujSmjYqumrvVaxWKoqg4oiIRgVR4QqaECAiRrgVA0OEIBkBGWUy+d4/9j7hZHIe9pk5+zztz2utrDlnn4f9PXvgc37z27/9+ykiMDOz4pjV7gLMzKy1HPxmZgXj4DczKxgHv5lZwTj4zcwKxsFvZlYwDn4rBEkXSbq83XVMl6TPSHpfevslkja1aL8h6Xmt2Je1joPfWkLSjZK2Sdoz4/PfKOmmvOtqJkn3SRqX9ISkhyT9h6S9m72fiPhBRCzKUE/XHUNrDQe/5U7SQuAlQACntrea3J0SEXsDxwB/BFw49QmSZre8KrMyDn5rhb8AbgE+A6wof0DSfElXS9oq6deSPibpBcDHgT9OW89j6XNvlPSXZa/dpUUr6aOS7pf0mKT1kl6SpThJd0t6bdn92ZIekXSMpL0kXZ7WNibpx5IOrPeeETEKfAM4Mn3PkPR2SfcA96TbXivp9vR9/6+kF5XVsETSbZIel3QlsFfZYydIemCax3BPSR+UtDn9q+TjkgbK3mulpC2SHpT0P7McP+s+Dn5rhb8APpf+W1YKTkl9wLXAL4GFwCDwhYi4G3gr8MOI2Dsi5mbcz4+BxcB+wOeBL0raq+YrElcAy8vuLwMeiYjbSL6ongXMB56d1jVe7w0lzQdeDWwo2zwMHAccIekY4NPAW9L3/QRwTRrMewAjwGfTz/JF4M+r7KfRY/jPwB+SHKfnpc9/T/peJwN/B7wCOBx4eb3Pad3JwW+5kvQnwKHAVRGxHvh/wOvTh48FDgZWRsRvI+J3ETHtPumIuDwifh0R2yPiQ8CeQN2+cJIviVMlzUnvvz7dBjBBEszPi4jJiFgfEY/VeK+RtHV9E/A94J/KHvtARDwaEePAm4FPRMSt6fteBvweOD791w98JCImIuJLJF9qlWQ+hpKU7vdv0joeT+s7O33KmcB/RMRPIuK3wEU1Pqd1MQe/5W0F8K2IeCS9/3me7u6ZD/wyIrY3Y0eSzk+7bX6Thu+zgP3rvS4i7gXuBk5Jw/9Ung7+zwJrgS+k3R//Iqm/xtsNR8TciDg0It6WhnzJ/WW3DwXOT7t5xtJ655OE+MHAaOw6g+Ivq+yvkWM4D5gDrC/b5zfT7aT7La+x2j6ty/kkk+Um7Ts+E+iT9Kt0857AXElHk4TMAkmzKwRXpWljf0sSXCXPKdvXS4B3AS8D7oqIHZK2AcpYbqm7Zxbw0/TLgIiYAC4GLk5PUl8HbAI+lfF9y5V/pvuB90fE+6c+SdKfAoOSVBb+C0j+WpqqkWP4CEk31QvTcxBTbSH5IilZUP2jWDdzi9/yNAxMAkeQ9CkvBl4A/ICk3/9HJGGzWtIz0hOpS9PXPgQckvZ3l9wOnC5pTjq2/E1lj+0DbAe2ArMlvQd4ZgO1fgF4JfC/eLq1j6QTJR2V9qU/RtL1M9nA+1bzSeCtko5T4hmSXiNpH+CH6Wf56/RE8+kkXTqVZD6GEbEj3e8lkg5IP9+gpGXp868C3ijpiPQvn/c24XNaB3LwW55WkPQZb46IX5X+AR8D3kDSGj+F5CTjZuAB4Kz0tTcAdwG/klTqJroEeIok0C4jOVlcspZkFM3PSboofseu3RY1RcQWksB9MXBl2UPPAb5EEvp3k/Tbz/hCsIhYR9Lf/jFgG3Av8Mb0saeA09P720iOydVV3meSxo7hu9J93SLpMeA7pOdBIuIbwEfS192b/rQeJC/EYmZWLG7xm5kVjIPfzKxgHPxmZgXj4DczK5iuGMe///77x8KFC9tdhplZV1m/fv0jETFv6vauCP6FCxeybt26dpdhZtZVJFW8+tpdPWZmBePgNzMrGAe/mVnBOPjNzArGwW9mVjBdMarHzKxoRjaMsmbtJh4cG+fguQOsXLaI4SWDTXlvB7+ZWYcZ2TDKBVdvZHwimQF8dGycC67eCNCU8HdXj5lZh1mzdtPO0C8Zn5hkzdpNTXl/B7+ZWYd5cGy8oe2NcvCbmXWYg+cONLS9UQ5+M7MOs3LZIgb6+3bZNtDfx8pli5ry/j65a2bWYUoncD2qx8ysQIaXDDYt6KfKvatHUp+kDZKuTe/vJ+nbku5Jf+6bdw1mZva0VvTxvxO4u+z+KuD6iDgcuD69b2ZmLZJr8Es6BHgN8O9lm08DLktvXwYM51mDmZntKu8W/0eAvwd2lG07MCK2AKQ/D6j0QknnSlonad3WrVtzLtPMrDhyC35JrwUejoj103l9RFwaEUMRMTRv3m4rh5mZ2TTlOapnKXCqpFcDewHPlHQ58JCkgyJii6SDgIdzrMHMzKbIrcUfERdExCERsRA4G7ghIs4BrgFWpE9bAXw1rxrMzGx37bhydzXwCkn3AK9I75uZWYu05AKuiLgRuDG9/WvgZa3Yr5mZ7c5z9ZiZFYyD38ysYBz8ZmYF4+A3MysYB7+ZWcE4+M3MCsbBb2ZWMA5+M7OCcfCbmRWMg9/MrGAc/GZmBePgNzMrGAe/mVnBOPjNzArGwW9mVjAOfjOzgnHwm5kVjIPfzKxgHPxmZgXj4DczKxgHv5lZwTj4zcwKxsFvZlYwDn4zs4Jx8JuZFYyD38ysYBz8ZmYF4+A3MysYB7+ZWcE4+M3MCsbBb2ZWMA5+M7OCcfCbmRWMg9/MrGByC35Je0n6kaQ7JN0l6eJ0+36Svi3pnvTnvnnVYGZmu8uzxf974KSIOBpYDJws6XhgFXB9RBwOXJ/eNzOzFskt+CPxRHq3P/0XwGnAZen2y4DhvGowM7Pd5drHL6lP0u3Aw8C3I+JW4MCI2AKQ/jygymvPlbRO0rqtW7fmWaaZWaHkGvwRMRkRi4FDgGMlHdnAay+NiKGIGJo3b15uNZqZFU1LRvVExBhwI3Ay8JCkgwDSnw+3ogYzM0vkOapnnqS56e0B4OXAz4BrgBXp01YAX82rBjMz293sHN/7IOAySX0kXzBXRcS1kn4IXCXpTcBm4IwcazAzsylyC/6IuBNYUmH7r4GX5bVfMzOrzVfumpkVjIPfzKxgHPxmZgXj4DczKxgHv5lZwTj4zcwKxsFvZlYwDn4zs4Jx8JuZFYyD38ysYBz8ZmYF4+A3MysYB7+ZWcE4+M3MCsbBb2ZWMA5+M7OCcfCbmRWMg9/MrGAc/GZmBePgNzMrGAe/mVnBzK71oKT9aj0eEY82txwzM8tbzeAH1gMBCFgAbEtvzwU2A4flWZyZmTVfza6eiDgsIp4LrAVOiYj9I+LZwGuBq1tRoJmZNVfWPv4/iojrSnci4hvAn+ZTkpmZ5aleV0/JI5IuBC4n6fo5B/h1blWZmVlusgb/cuC9wFdIgv/76TYzMyszsmGUNWs38eDYOAfPHWDlskUMLxlsd1m7yBT86eidd0raOyKeyLkmM7OuNLJhlAuu3sj4xCQAo2PjXHD1RoCOCv9MffySXizpp8BP0/tHS/rfuVZmZtZl1qzdtDP0S8YnJlmzdlObKqos68ndS4BlpP36EXEH8NK8ijIz60YPjo03tL1dMl+5GxH3T9k0WfGJZmYFdfDcgYa2t0vW4L9f0ouBkLSHpL8D7s6xLjOzrrNy2SIG+vt22TbQ38fKZYvaVFFlWUf1vBX4KDAIPAB8C3hbXkWZWWt1w0iUblA6Zp1+LLMG/6KIeEP5BklLgZubX5JZcbUjgLtlJEq3GF4y2PHHLWtXz79m3GZmU4xsGGXp6hs4bNXXWbr6BkY2jFZ93gVXb2R0bJzg6QCu9vxm6ZaRKNY89Wbn/GPgxcA8SX9b9tAzgb7Kr9r52vnAfwLPAXYAl0bER9MZP68EFgL3AWdGxLbpfgCzTtZIa7pWAOfZguyWkSjWPPVa/HsAe5N8QexT9u8x4HV1XrsdOD8iXgAcD7xd0hHAKuD6iDgcuD69b9aTGmlNtyuAu2UkijVPzRZ/RHwP+J6kz0TELxt544jYAmxJbz8u6W6Sk8OnASekT7sMuBF4V2Nlm3WHRsL84LkDjFbZnqeVyxbt8lcJdOZIFGuerH38/y5pbumOpH0lrc26E0kLgSXArcCB6ZdC6cvhgCqvOVfSOknrtm7dmnVXZh2lkdZ0u4YCDi8Z5AOnH8Xg3AEEDM4d4AOnH9XxJyht+rKO6tk/IsZKdyJim6SKgT2VpL2BLwPnRcRjkjLtMCIuBS4FGBoaiox1mnWURlrT7RwK2A0jUax5sgb/DkkLImIzgKRDSWbprElSP0nofy4iSgu3PCTpoIjYIukg4OHpFG7WDRoNcwewtULW4P8H4CZJ30vvvxQ4t9YLlDTtPwXcHREfLnvoGmAFsDr9+dWGKjbrMg5z6zRZp2X+pqRjSEbnCPibiHikzsuWAv8d2Cjp9nTbu0kC/ypJbyJZt/eM6RRuZtN34chGrrj1fiYj6JNYftx83jd8VLvLshapN47/+RHxszT0AR5Mfy5Iu35uq/baiLiJ5Euikpc1XqqZNcOFIxu5/JbNO+9PRuy87/Avhnot/vOBNwMfqvBYACc1vSIzy9UVt06daPfp7Q7+Yqg3jv/N6c8TW1OOmeVtMiqPy6i23XpPva6e02s9XjZSx8y6RJ9UMeT7Mg61tu5Xr6vnlPTnASRz9tyQ3j+R5IpbB79Zl1l+3Pxd+vjLt1sx1Ovq+R8Akq4FjihdcZuOv/+3/Msza4znla9v6ND9+Pytm9lR1uifpWS7FYMiQ7+epJ9ExJFl92cBd5Zvy9PQ0FCsW7euFbuyLjZ1JkxIhpUFyTQE3fwl0MwvtKWrb6g4J9Dg3AFuXuXxGr1E0vqIGJq6PesFXDemc/NcQfL/0dnAd5tYn9mMVZoJs9Ss6ebFRZq9UIqnYbZMk7RFxDuAjwNHA4tJ5tb/qxzrMmtYveDq1sVFmr1QiqdhtqwtfoDbgMcj4juS5kjaJyIez6sws0ZVm9a4XKUvh04/L9DsFrqnYbZMLX5Jbwa+BHwi3TQIjORUk9m0VJrWeKqprdp2LXfYiGa30D0Ns2Vt8b8dOJZkPn0i4p6s0zKbtUr5TJijY+M7T+yWVGrVtmu5w0bk0UL3xHHFljX4fx8RT5Xm0pc0mwzTMpu1WnmgZenC6YYTne2cp996U9bg/56kdwMDkl4BvA34Wn5lmc1cllZtu5Y7bJRb6NZMWZdefBewFdgIvAW4Drgwr6LMWqVdyx2atVPdFv+Ui7U+mX9JZq3TrG6UTh8ZZFaubvBHxA5Jd5QvvWjWS2bajdKMC6z8xWGtlLWP/yDgLkk/An5b2hgRp+ZSlVkXmenIoGZfmWtWT9bgvzjXKsy62ExHBnXDkFLrLfXm498LeCvwPJITu5+KiO2tKMysW8x0ZFA3DCm13lJvVM9lwBBJ6L+KykswmhXaTEcGee4ca7V6wX9ERJwTEZ8AXge8pAU1mXWVmU6B4CGl1mr1+vgnSjciYru8NJtZRTMZGeQrc63V6gX/0ZIeS2+L5Mrdx9LbERHPzLU6s4LwlbnWSvWWXqw91aGZmXWdrFM2mJlZj3Dwm5kVjIPfzKxgHPxmZgXTyJq7Zh3Lk5yZZefgt67nSc7MGuPg7yButU5PKyY58+/GeomDv0O41Tp9eU9y5t+N9Rqf3O0QtVqtVlu1ycxmSRy26ussXX0DIxtGp/3+/t1Yr3Hwd4hOm5p3ZMMoS1ff0JTgzFulSc4AJiMInm6hT/czdNrvxmymcuvqkfRp4LXAw+l6vUjaD7gSWAjcB5wZEdvyqqETZO0bnumc7s3UrK6N8s/+rIF+JNj25AR9EpMRDDapr3zqJGez0vcvN5M+/0763Zg1Q54t/s8AJ0/Ztgq4PiIOB65P7/esUoCOjo3XbXl20tS8zejamPrZx8Yn2PZkMtlrKZRrHY9G/+IYXjLIzatO4herX8OOKaFfMt0Weif9bsyaIbfgj4jvA49O2XwayeIupD+H89p/J2gkQGc6p3szNaNro9Jnr6TS8WjkC7OSZi9s0km/G7NmaPWongMjYgtARGyRdEC1J0o6FzgXYMGCBS0qr7kaDdBOmZq3GV0bjXxJPDg2vku30Ey7alYuW7RLVxXMvIXeKb8bs2bo2OGcEXEpcCnA0NBQ5b/dO1wr+4ZHNoxy8dfu2tmdMnegn4tOfWHdsKp0DuLE58/j8ls27/bcsSefYmTDaKYArPbZK5k7p3+XoJ4a+iVZv0y8sIlZbYoq/5M15c2lhcC1ZSd3NwEnpK39g4AbI6JuM2xoaCjWrVuXW515mXqSFJKWZ7O7CUY2jLLyS3cwMVn5d1ntJGq1+vacPYux8Ympb9NQ/ZXeu9r71drf1M9x86qT6j7PzBKS1kfE0NTtrR7OeQ2wIr29Avhqi/ffUq3qG16zdlPV0Iekj/y8K29n4aqv8wcXXMeFIxt3vq7SOYhaIZz1JO/Uzz53oJ995/QD0Jcu4Vk6Hr/JEPo+mWrWPHkO57wCOAHYX9IDwHuB1cBVkt4EbAbOyGv/nWKmfcNZhoM20p8+GbGzG2e6o1wa6XLJ8tnXrN1UsVuoT2JHhLtqzJos166eZunWrp6ZytpVtHT1DZn700tmCcTuJ1EB5vTPYnxiB9X+y9h3Tj9z9phdt/886zUMreoSMyuaTunqsQZkHQ564vPnNfzeO6L6SdQna4R+f5944nfbdxlqed6Vt7PkH7+1y3DLRoZkerikWWt17KieIqjXIs4yHHRkwyhfXt+a6RT6pKrnErY9ObHL1b2Nzpjp4ZJmreMWf5M0eqVplhZxrcnHSs/LeqFUM1T7C6Gk/K8Rz29j1rkc/E1QKcRLo2gWrvo6iy/evRvk/KvuqNuNs3LZIvr7tNv+JiM478rbWXzxtxru289bKdibffWsmTWPu3qaoF6re2x8gpVfvGPn/Quu3ljzIqVSF1C9UB8bn0BQtT++HUrBnsfVs2bWHA7+JsjSfTGxI3a25mt9ScyeRaYLn0oCOib8y4PdV8+adS4HfxNknZ4g2xcETOxorM8+SC6QynL1675z+ndO6zATA/19/Pl/G+S7P9taNdh9wtasMzn4m6BSt0YlpW6QPPrlf799R91QF8nom+n+heALqsx6g4O/Ccq7NaqFev8s7ewGqTWvznSNT0yy5+xZNUM9yn6Wnjd1Hp+Fq75edR87IvjF6tc0r2gzawsHf5OUd2tMnSlTSvr4L/7aXfx+YrLpoV/ym/EJ3nD8Aj53y+a6LfpS6E+d9GywRreVR+SY9QYP52xA1rH6w0sG2fCeV/KRsxYz0N9HaQDPticneHJiR271PWugn/cNH8UlZy3eORFaLZXOOVQbQlr+F4uZdTcHf0bTWRWqlRdXQTK888KRjQwvGay6/GC5Si344SWDrHnd0Ttn0oTkxPGaM452n75Zj3BXT0bVpiC46Jq7qg5ZbMfFVZ+7ZTNDh+5Xd6RRrTH1Ho1j1tsc/BlVG4o5Nj6xcxhl6a+AknaMrw+SL6lKI42qndA1s2Jx8GeUdax++bQL7bqo6sGxcV9AZWZVOfgzyjpWH9rTxVOu1HfvLhszq8QndzOaOmf8vnP6yTBwpuU8H46Z1ePgb8DwkkFuXnUSl5y1mCd+t512L15WGrI5dQ1bt/LNrBZ39UzDu6++k4kd7Z8WbUcE9/lKWjNrUCGC/8KRjVxx6/1MRtAnsfy4+bxv+KhdntPI+rB5XoTVCF9Ja2bT0fPBf+HIRi6/ZfPO+5MRO++Xwn/qYt/lwzKnhv/U9W5bYZbSZQ/L/spwX76ZTVfP9/Ffcev9dbdnXdQc2jNi58NnLmbNGUd7MXIza4qeb/FXW+mqfHu99WGzrojVqHoXePXP0i5TJTjozawZerrFX2/B85JqfeUB/OE/XMd5V96eS0v/krMW79KKP+f4Bbvc9/w4ZpaHnm7xZ+mPH9kwytiTT1V9/KmcplDuk3yBlZm1RU8Hf72lDmstOpK35cfNb9u+zazYerqrp93DHavNiT+nf9Zuw0nNzFqlp4N/5bJF9M9q/bwKAu5b/Ro+dObRDPT37fLYQH8f/3T6i1pek5lZSU8H//CSQfaY3fqPWD5JWvn8Ph6GaWadoKf7+AF++1TrVsAqKb+wyidwzazT9HSLP+twzmaaO9DvoDezjtbTwX/RNXe1dH8D/X1cdOoLW7pPM7NG9WxXz8iG0Z1LIrbCvnP6ee8pL3Rr38w6Xs+2+C/+Wmtb+3P2mO3QN7Ou0Jbgl3SypE2S7pW0Ko99bHtyZq39RgeB1rtYzMysU7Q8+CX1Af8GvAo4Algu6YhW11GLgDek8+Zk1e6LxczMsmpHi/9Y4N6I+K+IeAr4AnBaG+qo6g3HL+B9w0dx86qTMoW/58Y3s27SjuAfBMonyX8g3bYLSedKWidp3datW1tW3Dlp6JfU6sLxRVlm1o3aMaqnUvf5blNgRsSlwKUAQ0NDLVvgduocOgfPHag4JfPg3AFuXnVSq8oyM2uadrT4HwDKp6Y8BHiw2TupMj9aw1YuW1Rxvh137ZhZt2pHi//HwOGSDgNGgbOB1zd7J1UW3qpp3zn9u20rdeFkWYjdzKwbtDz4I2K7pHcAa4E+4NMR0dpB91W895TKV916vh0z6yVtuXI3Iq4DrmvHvmtxuJtZEfTslbuNamTMvplZN3Pw45O1ZlYshQx+kZzI9Th8MyuinpydM8s8/Bve88oWVGJm1nl6ssW/Zu2mmo97Xh0zK7KeDP5KV9qWc3++mRVZTwZ/X53Ldt2fb2ZF1pPBP1njst1KV+eamRVJTwZ/rTH5T/yudcsxmpl1op4M/lp9+BM7so36MTPrVT0Z/PX68OuN+jEz62U9GfxQuy/f6+OaWZH1bPBXm2kTPI7fzIqtZ4N/eMkg5xy/YLflvjwvj5kVXc8GPyTLKF5y1mIG5w54Xh4zs1RPztVTzouomJntqqdb/GZmtjsHv5lZwTj4zcwKxsFvZlYwDn4zs4JR1JjJslNI2gr8chov3R94pMnl5Mn15qebaoXuqrebaoVi1XtoRMyburErgn+6JK2LiKF215GV681PN9UK3VVvN9UKrhfc1WNmVjgOfjOzgun14L+03QU0yPXmp5tqhe6qt5tqBdfb2338Zma2u15v8ZuZ2RQOfjOzgunZ4Jd0sqRNku6VtKrd9dQj6T5JGyXdLmldu+spJ+nTkh6W9JOybftJ+rake9Kf+7azxnJV6r1I0mh6fG+X9Op21lgiab6k70q6W9Jdkt6Zbu/I41uj3o47vpL2kvQjSXektV6cbu/UY1ut3qYf257s45fUB/wceAXwAPBjYHlE/LSthdUg6T5gKCI67sISSS8FngD+MyKOTLf9C/BoRKxOv1j3jYh3tbPOkir1XgQ8EREfbGdtU0k6CDgoIm6TtA+wHhgG3kgHHt8a9Z5Jhx1fSQKeERFPSOoHbgLeCZxOZx7bavWeTJOPba+2+I8F7o2I/4qIp4AvAKe1uaauFRHfBx6dsvk04LL09mUk//N3hCr1dqSI2BIRt6W3HwfuBgbp0ONbo96OE4kn0rv96b+gc49ttXqbrleDfxC4v+z+A3Tof5xlAviWpPWSzm13MRkcGBFbIAkD4IA215PFOyTdmXYFdcSf9+UkLQSWALfSBcd3Sr3QgcdXUp+k24GHgW9HREcf2yr1QpOPba8G/9SldiGnb84mWhoRxwCvAt6edldY8/wf4A+AxcAW4ENtrWYKSXsDXwbOi4jH2l1PPRXq7cjjGxGTEbEYOAQ4VtKRbS6ppir1Nv3Y9mrwPwDML7t/CPBgm2rJJCIeTH8+DHyFpLuqkz2U9veW+n0fbnM9NUXEQ+n/VDuAT9JBxzftz/0y8LmIuDrd3LHHt1K9nXx8ASJiDLiRpL+8Y49tSXm9eRzbXg3+HwOHSzpM0h7A2cA1ba6pKknPSE+UIekZwCuBn9R+VdtdA6xIb68AvtrGWuoq/Y+e+jM65PimJ/Q+BdwdER8ue6gjj2+1ejvx+EqaJ2luensAeDnwMzr32FasN49j25OjegDSIU8fAfqAT0fE+9tbUXWSnkvSygeYDXy+k+qVdAVwAsn0sA8B7wVGgKuABcBm4IyI6IgTqlXqPYHkT+UA7gPeUurnbSdJfwL8ANgI7Eg3v5uk37zjjm+NepfTYcdX0otITt72kTRyr4qIf5T0bDrz2Far97M0+dj2bPCbmVllvdrVY2ZmVTj4zcwKxsFvZlYwDn4zs4Jx8JuZFYyD3ywl6c8khaTn13neeZLmzGA/b5T0sem+3mymHPxmT1tOMiPi2XWedx4w7eA3azcHvxk7555ZCryJNPjTCbM+qGSdhDsl/ZWkvwYOBr4r6bvp854oe5/XSfpMevsUSbdK2iDpO5IObPXnMqtkdrsLMOsQw8A3I+Lnkh6VdAxwHHAYsCQitkvaLyIelfS3wIkZ1k64CTg+IkLSXwJ/D5yf54cwy8LBb5ZYTjLFByTrNywHngt8PCK2A0zjsv5DgCvTuVb2AH7RnFLNZsbBb4WXzt1yEnCkpCCZKyVIVpfKMqdJ+XP2Krv9r8CHI+IaSScAFzWjXrOZch+/GbyOZJnGQyNiYUTMJ2md3wa8VdJsSNZqTZ//OLBP2esfkvQCSbNIZk8seRYwmt5egVmHcPCbJd06X5my7cskJ3E3A3dKugN4ffrYpcA3Sid3gVXAtcANJAtllFwEfFHSD4COW0vZisuzc5qZFYxb/GZmBePgNzMrGAe/mVnBOPjNzArGwW9mVjAOfjOzgnHwm5kVzP8Hbqg6qbhSUjwAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(Y_train, training_data_prediction)\n",
    "plt.xlabel(\"Actual\")\n",
    "plt.ylabel(\"Predicted\")\n",
    "plt.title(\"Actual vs Predicted\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Prediction on Test Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "test_data_prediction = lin_reg_model.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "R Squared Error -  0.8365766715026903\n"
     ]
    }
   ],
   "source": [
    "error_score = metrics.r2_score(Y_test, test_data_prediction)\n",
    "print(\"R Squared Error - \",error_score)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(Y_test, test_data_prediction)\n",
    "plt.xlabel(\"Actual\")\n",
    "plt.ylabel(\"Predicted\")\n",
    "plt.title(\"Actual vs Predicted\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Model Training \n",
    "    Lasso Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "lass_reg_model = Lasso()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Lasso()"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "lass_reg_model.fit(X_train, Y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "training_data_prediction = lass_reg_model.predict(X_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "R Squared Error -  0.8427856123435794\n"
     ]
    }
   ],
   "source": [
    "error_score = metrics.r2_score(Y_train, training_data_prediction)\n",
    "print(\"R Squared Error - \",error_score)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(Y_train, training_data_prediction)\n",
    "plt.xlabel(\"Actual\")\n",
    "plt.ylabel(\"Predicted\")\n",
    "plt.title(\"Actual vs Predicted\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "test_data_prediction = lass_reg_model.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "R Squared Error -  0.8709167941173195\n"
     ]
    }
   ],
   "source": [
    "error_score = metrics.r2_score(Y_test, test_data_prediction)\n",
    "print(\"R Squared Error - \",error_score)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(Y_test, test_data_prediction)\n",
    "plt.xlabel(\"Actual\")\n",
    "plt.ylabel(\"Predicted\")\n",
    "plt.title(\"Actual vs Predicted\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
