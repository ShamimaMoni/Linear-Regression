{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.datasets import load_boston\n",
    "boston = load_boston()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "dict_keys(['data', 'target', 'feature_names', 'DESCR', 'filename'])\n"
     ]
    }
   ],
   "source": [
    "print(boston.keys())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(506, 13)\n"
     ]
    }
   ],
   "source": [
    " print(boston.data.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'\n",
      " 'B' 'LSTAT']\n"
     ]
    }
   ],
   "source": [
    "print(boston.feature_names)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      ".. _boston_dataset:\n",
      "\n",
      "Boston house prices dataset\n",
      "---------------------------\n",
      "\n",
      "**Data Set Characteristics:**  \n",
      "\n",
      "    :Number of Instances: 506 \n",
      "\n",
      "    :Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.\n",
      "\n",
      "    :Attribute Information (in order):\n",
      "        - CRIM     per capita crime rate by town\n",
      "        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.\n",
      "        - INDUS    proportion of non-retail business acres per town\n",
      "        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)\n",
      "        - NOX      nitric oxides concentration (parts per 10 million)\n",
      "        - RM       average number of rooms per dwelling\n",
      "        - AGE      proportion of owner-occupied units built prior to 1940\n",
      "        - DIS      weighted distances to five Boston employment centres\n",
      "        - RAD      index of accessibility to radial highways\n",
      "        - TAX      full-value property-tax rate per $10,000\n",
      "        - PTRATIO  pupil-teacher ratio by town\n",
      "        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town\n",
      "        - LSTAT    % lower status of the population\n",
      "        - MEDV     Median value of owner-occupied homes in $1000's\n",
      "\n",
      "    :Missing Attribute Values: None\n",
      "\n",
      "    :Creator: Harrison, D. and Rubinfeld, D.L.\n",
      "\n",
      "This is a copy of UCI ML housing dataset.\n",
      "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/\n",
      "\n",
      "\n",
      "This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.\n",
      "\n",
      "The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic\n",
      "prices and the demand for clean air', J. Environ. Economics & Management,\n",
      "vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics\n",
      "...', Wiley, 1980.   N.B. Various transformations are used in the table on\n",
      "pages 244-261 of the latter.\n",
      "\n",
      "The Boston house-price data has been used in many machine learning papers that address regression\n",
      "problems.   \n",
      "     \n",
      ".. topic:: References\n",
      "\n",
      "   - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.\n",
      "   - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(boston.DESCR)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
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
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "      <th>4</th>\n",
       "      <th>5</th>\n",
       "      <th>6</th>\n",
       "      <th>7</th>\n",
       "      <th>8</th>\n",
       "      <th>9</th>\n",
       "      <th>10</th>\n",
       "      <th>11</th>\n",
       "      <th>12</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.00632</td>\n",
       "      <td>18.0</td>\n",
       "      <td>2.31</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.538</td>\n",
       "      <td>6.575</td>\n",
       "      <td>65.2</td>\n",
       "      <td>4.0900</td>\n",
       "      <td>1.0</td>\n",
       "      <td>296.0</td>\n",
       "      <td>15.3</td>\n",
       "      <td>396.90</td>\n",
       "      <td>4.98</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.02731</td>\n",
       "      <td>0.0</td>\n",
       "      <td>7.07</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.469</td>\n",
       "      <td>6.421</td>\n",
       "      <td>78.9</td>\n",
       "      <td>4.9671</td>\n",
       "      <td>2.0</td>\n",
       "      <td>242.0</td>\n",
       "      <td>17.8</td>\n",
       "      <td>396.90</td>\n",
       "      <td>9.14</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.02729</td>\n",
       "      <td>0.0</td>\n",
       "      <td>7.07</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.469</td>\n",
       "      <td>7.185</td>\n",
       "      <td>61.1</td>\n",
       "      <td>4.9671</td>\n",
       "      <td>2.0</td>\n",
       "      <td>242.0</td>\n",
       "      <td>17.8</td>\n",
       "      <td>392.83</td>\n",
       "      <td>4.03</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.03237</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2.18</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.458</td>\n",
       "      <td>6.998</td>\n",
       "      <td>45.8</td>\n",
       "      <td>6.0622</td>\n",
       "      <td>3.0</td>\n",
       "      <td>222.0</td>\n",
       "      <td>18.7</td>\n",
       "      <td>394.63</td>\n",
       "      <td>2.94</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.06905</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2.18</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.458</td>\n",
       "      <td>7.147</td>\n",
       "      <td>54.2</td>\n",
       "      <td>6.0622</td>\n",
       "      <td>3.0</td>\n",
       "      <td>222.0</td>\n",
       "      <td>18.7</td>\n",
       "      <td>396.90</td>\n",
       "      <td>5.33</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        0     1     2    3      4      5     6       7    8      9     10  \\\n",
       "0  0.00632  18.0  2.31  0.0  0.538  6.575  65.2  4.0900  1.0  296.0  15.3   \n",
       "1  0.02731   0.0  7.07  0.0  0.469  6.421  78.9  4.9671  2.0  242.0  17.8   \n",
       "2  0.02729   0.0  7.07  0.0  0.469  7.185  61.1  4.9671  2.0  242.0  17.8   \n",
       "3  0.03237   0.0  2.18  0.0  0.458  6.998  45.8  6.0622  3.0  222.0  18.7   \n",
       "4  0.06905   0.0  2.18  0.0  0.458  7.147  54.2  6.0622  3.0  222.0  18.7   \n",
       "\n",
       "       11    12  \n",
       "0  396.90  4.98  \n",
       "1  396.90  9.14  \n",
       "2  392.83  4.03  \n",
       "3  394.63  2.94  \n",
       "4  396.90  5.33  "
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "dat = pd.DataFrame(boston.data)\n",
    "dat.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
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
       "      <th>CRIM</th>\n",
       "      <th>ZN</th>\n",
       "      <th>INDUS</th>\n",
       "      <th>CHAS</th>\n",
       "      <th>NOX</th>\n",
       "      <th>RM</th>\n",
       "      <th>AGE</th>\n",
       "      <th>DIS</th>\n",
       "      <th>RAD</th>\n",
       "      <th>TAX</th>\n",
       "      <th>PTRATIO</th>\n",
       "      <th>B</th>\n",
       "      <th>LSTAT</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.00632</td>\n",
       "      <td>18.0</td>\n",
       "      <td>2.31</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.538</td>\n",
       "      <td>6.575</td>\n",
       "      <td>65.2</td>\n",
       "      <td>4.0900</td>\n",
       "      <td>1.0</td>\n",
       "      <td>296.0</td>\n",
       "      <td>15.3</td>\n",
       "      <td>396.90</td>\n",
       "      <td>4.98</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.02731</td>\n",
       "      <td>0.0</td>\n",
       "      <td>7.07</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.469</td>\n",
       "      <td>6.421</td>\n",
       "      <td>78.9</td>\n",
       "      <td>4.9671</td>\n",
       "      <td>2.0</td>\n",
       "      <td>242.0</td>\n",
       "      <td>17.8</td>\n",
       "      <td>396.90</td>\n",
       "      <td>9.14</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.02729</td>\n",
       "      <td>0.0</td>\n",
       "      <td>7.07</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.469</td>\n",
       "      <td>7.185</td>\n",
       "      <td>61.1</td>\n",
       "      <td>4.9671</td>\n",
       "      <td>2.0</td>\n",
       "      <td>242.0</td>\n",
       "      <td>17.8</td>\n",
       "      <td>392.83</td>\n",
       "      <td>4.03</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.03237</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2.18</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.458</td>\n",
       "      <td>6.998</td>\n",
       "      <td>45.8</td>\n",
       "      <td>6.0622</td>\n",
       "      <td>3.0</td>\n",
       "      <td>222.0</td>\n",
       "      <td>18.7</td>\n",
       "      <td>394.63</td>\n",
       "      <td>2.94</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.06905</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2.18</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.458</td>\n",
       "      <td>7.147</td>\n",
       "      <td>54.2</td>\n",
       "      <td>6.0622</td>\n",
       "      <td>3.0</td>\n",
       "      <td>222.0</td>\n",
       "      <td>18.7</td>\n",
       "      <td>396.90</td>\n",
       "      <td>5.33</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      CRIM    ZN  INDUS  CHAS    NOX     RM   AGE     DIS  RAD    TAX  \\\n",
       "0  0.00632  18.0   2.31   0.0  0.538  6.575  65.2  4.0900  1.0  296.0   \n",
       "1  0.02731   0.0   7.07   0.0  0.469  6.421  78.9  4.9671  2.0  242.0   \n",
       "2  0.02729   0.0   7.07   0.0  0.469  7.185  61.1  4.9671  2.0  242.0   \n",
       "3  0.03237   0.0   2.18   0.0  0.458  6.998  45.8  6.0622  3.0  222.0   \n",
       "4  0.06905   0.0   2.18   0.0  0.458  7.147  54.2  6.0622  3.0  222.0   \n",
       "\n",
       "   PTRATIO       B  LSTAT  \n",
       "0     15.3  396.90   4.98  \n",
       "1     17.8  396.90   9.14  \n",
       "2     17.8  392.83   4.03  \n",
       "3     18.7  394.63   2.94  \n",
       "4     18.7  396.90   5.33  "
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dat.columns = boston.feature_names\n",
    "dat.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
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
       "      <th>CRIM</th>\n",
       "      <th>ZN</th>\n",
       "      <th>INDUS</th>\n",
       "      <th>CHAS</th>\n",
       "      <th>NOX</th>\n",
       "      <th>RM</th>\n",
       "      <th>AGE</th>\n",
       "      <th>DIS</th>\n",
       "      <th>RAD</th>\n",
       "      <th>TAX</th>\n",
       "      <th>PTRATIO</th>\n",
       "      <th>B</th>\n",
       "      <th>LSTAT</th>\n",
       "      <th>Price</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.00632</td>\n",
       "      <td>18.0</td>\n",
       "      <td>2.31</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.538</td>\n",
       "      <td>6.575</td>\n",
       "      <td>65.2</td>\n",
       "      <td>4.0900</td>\n",
       "      <td>1.0</td>\n",
       "      <td>296.0</td>\n",
       "      <td>15.3</td>\n",
       "      <td>396.90</td>\n",
       "      <td>4.98</td>\n",
       "      <td>24.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.02731</td>\n",
       "      <td>0.0</td>\n",
       "      <td>7.07</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.469</td>\n",
       "      <td>6.421</td>\n",
       "      <td>78.9</td>\n",
       "      <td>4.9671</td>\n",
       "      <td>2.0</td>\n",
       "      <td>242.0</td>\n",
       "      <td>17.8</td>\n",
       "      <td>396.90</td>\n",
       "      <td>9.14</td>\n",
       "      <td>21.6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.02729</td>\n",
       "      <td>0.0</td>\n",
       "      <td>7.07</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.469</td>\n",
       "      <td>7.185</td>\n",
       "      <td>61.1</td>\n",
       "      <td>4.9671</td>\n",
       "      <td>2.0</td>\n",
       "      <td>242.0</td>\n",
       "      <td>17.8</td>\n",
       "      <td>392.83</td>\n",
       "      <td>4.03</td>\n",
       "      <td>34.7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.03237</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2.18</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.458</td>\n",
       "      <td>6.998</td>\n",
       "      <td>45.8</td>\n",
       "      <td>6.0622</td>\n",
       "      <td>3.0</td>\n",
       "      <td>222.0</td>\n",
       "      <td>18.7</td>\n",
       "      <td>394.63</td>\n",
       "      <td>2.94</td>\n",
       "      <td>33.4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.06905</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2.18</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.458</td>\n",
       "      <td>7.147</td>\n",
       "      <td>54.2</td>\n",
       "      <td>6.0622</td>\n",
       "      <td>3.0</td>\n",
       "      <td>222.0</td>\n",
       "      <td>18.7</td>\n",
       "      <td>396.90</td>\n",
       "      <td>5.33</td>\n",
       "      <td>36.2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      CRIM    ZN  INDUS  CHAS    NOX     RM   AGE     DIS  RAD    TAX  \\\n",
       "0  0.00632  18.0   2.31   0.0  0.538  6.575  65.2  4.0900  1.0  296.0   \n",
       "1  0.02731   0.0   7.07   0.0  0.469  6.421  78.9  4.9671  2.0  242.0   \n",
       "2  0.02729   0.0   7.07   0.0  0.469  7.185  61.1  4.9671  2.0  242.0   \n",
       "3  0.03237   0.0   2.18   0.0  0.458  6.998  45.8  6.0622  3.0  222.0   \n",
       "4  0.06905   0.0   2.18   0.0  0.458  7.147  54.2  6.0622  3.0  222.0   \n",
       "\n",
       "   PTRATIO       B  LSTAT  Price  \n",
       "0     15.3  396.90   4.98   24.0  \n",
       "1     17.8  396.90   9.14   21.6  \n",
       "2     17.8  392.83   4.03   34.7  \n",
       "3     18.7  394.63   2.94   33.4  \n",
       "4     18.7  396.90   5.33   36.2  "
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dat['Price'] = boston.target\n",
    "dat.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'House Price')"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from matplotlib import pyplot as plt\n",
    "plt.scatter(dat['RM'],dat['Price'], color='r')\n",
    "plt.xlabel('Average Number of Rooms (RM)')\n",
    "plt.ylabel('House Price')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "218    5.951\n",
      "117    6.021\n",
      "195    7.875\n",
      "162    7.802\n",
      "219    6.373\n",
      "440    5.818\n",
      "139    6.151\n",
      "381    6.545\n",
      "319    6.113\n",
      "474    5.427\n",
      "386    4.652\n",
      "62     6.456\n",
      "191    6.739\n",
      "312    6.023\n",
      "155    6.152\n",
      "18     5.456\n",
      "328    5.868\n",
      "198    7.274\n",
      "372    5.875\n",
      "0      6.575\n",
      "60     5.741\n",
      "108    6.474\n",
      "298    6.345\n",
      "138    5.857\n",
      "370    7.016\n",
      "445    6.459\n",
      "428    6.193\n",
      "481    6.750\n",
      "55     7.249\n",
      "488    5.454\n",
      "       ...  \n",
      "468    5.926\n",
      "30     5.713\n",
      "431    6.833\n",
      "65     6.290\n",
      "203    7.853\n",
      "205    5.891\n",
      "44     6.069\n",
      "27     6.047\n",
      "80     6.727\n",
      "437    6.152\n",
      "113    6.092\n",
      "399    5.852\n",
      "204    8.034\n",
      "7      6.172\n",
      "208    6.064\n",
      "158    6.066\n",
      "112    5.913\n",
      "411    6.657\n",
      "446    6.341\n",
      "231    7.412\n",
      "228    7.686\n",
      "8      5.631\n",
      "73     6.245\n",
      "400    5.987\n",
      "118    5.872\n",
      "486    6.114\n",
      "189    7.185\n",
      "495    5.670\n",
      "206    6.326\n",
      "355    5.936\n",
      "Name: RM, Length: 379, dtype: float64\n",
      "<class 'pandas.core.series.Series'>\n"
     ]
    }
   ],
   "source": [
    "import sklearn.model_selection \n",
    "\n",
    "X = dat['RM']\n",
    "Y = dat['Price']\n",
    "\n",
    "X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.25, random_state = 5)\n",
    "X_train.shape\n",
    "print(X_train)\n",
    "print(type(X_train))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Simple Linear Regression(Part-1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train = X_train.values.reshape(-1,1)\n",
    "Y_train = Y_train.values.reshape(-1,1)\n",
    "X_test = X_test.values.reshape(-1,1)\n",
    "Y_test = Y_test.values.reshape(-1,1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.linear_model import LinearRegression\n",
    "\n",
    "lm = LinearRegression()\n",
    "lm.fit(X_train, Y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[-34.41739096]\n",
      "[[9.06976014]]\n"
     ]
    }
   ],
   "source": [
    "#the intercept:\n",
    "print(lm.intercept_)\n",
    "#the slope:\n",
    "print(lm.coef_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "Y_pred = lm.predict(X_test)"
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
       "[<matplotlib.lines.Line2D at 0x1d39467d358>]"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    },
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
    "plt.scatter(X_test, Y_test)\n",
    "plt.plot(X_test, Y_pred, color='red', linewidth=2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1152x720 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "Output = pd.DataFrame({'Actual': Y_test.flatten(), 'Predicted': Y_pred.flatten()})\n",
    "df1 = Output.head(10)\n",
    "df1.plot(kind='bar',figsize=(16,10))\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'me': 0.19893030499139885,\n",
       " 'mae': 3.9352380033262495,\n",
       " 'mse': 38.39960082194633,\n",
       " 'rmse': 6.1967411453074535}"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Accuracy metrics\n",
    "import numpy as np\n",
    "def forecast_accuracy(forecast, actual):\n",
    "    me = np.mean(forecast - actual)             # ME\n",
    "    mae = np.mean(np.abs(forecast - actual))    # MAE\n",
    "    mse = np.mean ((forecast-actual)**2)      #MSE\n",
    "    rmse = np.mean((forecast - actual)**2)**.5  # RMSE                   \n",
    "    return({'me':me, 'mae': mae, 'mse':mse, 'rmse':rmse})\n",
    "\n",
    "forecast_accuracy(Y_pred, Y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "R2 (on testing data) : 0.53\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import r2_score\n",
    "r2 = r2_score(Y_test, Y_pred)\n",
    "print(\"R2 (on testing data) : {:.2}\".format(r2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "R2 (on testing data) : 0.53\n"
     ]
    }
   ],
   "source": [
    "r2= lm.score(X_test, Y_test)\n",
    "print(\"R2 (on testing data) : {:.2}\".format(r2))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Simple Linear Regression (Part-2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "import statsmodels.api as sm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<table class=\"simpletable\">\n",
       "<caption>OLS Regression Results</caption>\n",
       "<tr>\n",
       "  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared (uncentered):</th>      <td>   0.901</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared (uncentered):</th> <td>   0.900</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>          <td>   3423.</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Date:</th>             <td>Sat, 13 Jun 2020</td> <th>  Prob (F-statistic):</th>          <td>1.52e-191</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Time:</th>                 <td>11:01:55</td>     <th>  Log-Likelihood:    </th>          <td> -1312.3</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>No. Observations:</th>      <td>   379</td>      <th>  AIC:               </th>          <td>   2627.</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Df Residuals:</th>          <td>   378</td>      <th>  BIC:               </th>          <td>   2631.</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>              <td> </td>    \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>              <td> </td>    \n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "   <td></td>     <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>x1</th> <td>    3.6677</td> <td>    0.063</td> <td>   58.506</td> <td> 0.000</td> <td>    3.544</td> <td>    3.791</td>\n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "  <th>Omnibus:</th>       <td>51.363</td> <th>  Durbin-Watson:     </th> <td>   1.970</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  81.511</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Skew:</th>          <td> 0.839</td> <th>  Prob(JB):          </th> <td>2.00e-18</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Kurtosis:</th>      <td> 4.532</td> <th>  Cond. No.          </th> <td>    1.00</td>\n",
       "</tr>\n",
       "</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified."
      ],
      "text/plain": [
       "<class 'statsmodels.iolib.summary.Summary'>\n",
       "\"\"\"\n",
       "                                 OLS Regression Results                                \n",
       "=======================================================================================\n",
       "Dep. Variable:                      y   R-squared (uncentered):                   0.901\n",
       "Model:                            OLS   Adj. R-squared (uncentered):              0.900\n",
       "Method:                 Least Squares   F-statistic:                              3423.\n",
       "Date:                Sat, 13 Jun 2020   Prob (F-statistic):                   1.52e-191\n",
       "Time:                        11:01:55   Log-Likelihood:                         -1312.3\n",
       "No. Observations:                 379   AIC:                                      2627.\n",
       "Df Residuals:                     378   BIC:                                      2631.\n",
       "Df Model:                           1                                                  \n",
       "Covariance Type:            nonrobust                                                  \n",
       "==============================================================================\n",
       "                 coef    std err          t      P>|t|      [0.025      0.975]\n",
       "------------------------------------------------------------------------------\n",
       "x1             3.6677      0.063     58.506      0.000       3.544       3.791\n",
       "==============================================================================\n",
       "Omnibus:                       51.363   Durbin-Watson:                   1.970\n",
       "Prob(Omnibus):                  0.000   Jarque-Bera (JB):               81.511\n",
       "Skew:                           0.839   Prob(JB):                     2.00e-18\n",
       "Kurtosis:                       4.532   Cond. No.                         1.00\n",
       "==============================================================================\n",
       "\n",
       "Warnings:\n",
       "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.\n",
       "\"\"\""
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "## Without a constant\n",
    "model = sm.OLS(Y_train, X_train).fit()\n",
    "\n",
    "# Print out the statistics\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA54AAAI7CAYAAACEKUQOAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjAsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+17YcXAAAgAElEQVR4nO3df5DldX3n+9dbhs0IUkSg40VQh8vVlQg6aF+CIonGaDB4SbiYimgiSZmglbUilS2z6G5KrIpbbHA1prwxNbsgeis7XIOCuCQu/mJJvArOwEjAAREd44jCCIrD8kOBz/7RB3bAwemZ6Xc3febxqOqa09/zPd/z7m9Ndfezv9/zPTXGCAAAAHR5wlIPAAAAwHQTngAAALQSngAAALQSngAAALQSngAAALQSngAAALRasZhPdtBBB41Vq1Yt5lMCAACwSNavX/+9McbMo5cvaniuWrUq69atW8ynBAAAYJFU1Te3t9yptgAAALQSngAAALQSngAAALRa1Nd4AgAALJUf//jH2bx5c+69996lHmXZW7lyZQ499NDsvffe81pfeAIAAHuEzZs3Z7/99suqVatSVUs9zrI1xsjtt9+ezZs357DDDpvXY5xqCwAA7BHuvffeHHjggaJzN1VVDjzwwJ06ciw8AQCAPYboXBg7ux+FJwAAwCK66KKLUlW54YYbfup6559/fm655ZZdfp7LL788r3rVq3b58QvJazwBAIA90qozL13Q7W06+8R5rbd27dq8+MUvzgUXXJCzzjrrMdc7//zzc+SRR+apT33qAk24dBzxBAAAWCR33XVXPv/5z+fcc8/NBRdc8PDyP//zP89RRx2V5z3veTnzzDNz4YUXZt26dXnd616X1atX55577smqVavyve99L0mybt26vOQlL0mSXHXVVXnRi16Uo48+Oi960Yty4403LsWX9lM54gkAALBILr744pxwwgl51rOelQMOOCBXX311br311lx88cW58sors88+++SOO+7IAQcckPe///1597vfndnZ2Z+6zWc/+9m54oorsmLFinz605/O29/+9nz0ox9dpK9ofoQnAADAIlm7dm3OOOOMJMlrXvOarF27Ng8++GB+7/d+L/vss0+S5IADDtipbd5555057bTTctNNN6Wq8uMf/3jB595dwhMAAGAR3H777fnsZz+b6667LlWVBx54IFWVU045ZV5XiV2xYkUefPDBJHnEW5n86Z/+aV760pfmoosuyqZNmx4+BffxxGs8AQAAFsGFF16Y17/+9fnmN7+ZTZs25Vvf+lYOO+ywHHDAATnvvPNy9913J0nuuOOOJMl+++2XrVu3Pvz4VatWZf369UnyiFNp77zzzhxyyCFJ5i5I9HgkPAEAABbB2rVrc/LJJz9i2SmnnJJbbrklJ510UmZnZ7N69eq8+93vTpL87u/+bt70pjc9fHGhd7zjHXnLW96S448/PnvttdfD2/iTP/mTvO1tb8txxx2XBx54YFG/pvmqMcaiPdns7OxYt27doj0fAADAQzZu3JgjjjhiqceYGtvbn1W1fozxE1dDcsQTAACAVsITAACAVsITAACAVsITAACAVsITAACAVsITAACAVsITAABgkey1115ZvXp1jjzyyPzmb/5m7r777l3e1uWXX55XvepVSZJLLrkkZ5999mOu+4Mf/CB/9Vd/tdPPcdZZZz38vqK7Y8VubwEAAGA5Omv/Bd7enTtc5YlPfGI2bNiQJHnd616Xv/7rv84f//EfP3z/GCNjjDzhCTt3jPCkk07KSSed9Jj3PxSef/iHf7hT210oUxGeq868tG3bm84+sW3bAADAnuv444/Ptddem02bNuWVr3xlXvrSl+YLX/hCLr744tx44415xzvekfvuuy+HH354PvjBD+ZJT3pSPvnJT+aMM87IQQcdlOc///kPb+v888/PunXr8v73vz+33npr3vSmN+XrX/96kuQDH/hA/vIv/zI333xzVq9enZe//OU555xzcs455+QjH/lI7rvvvpx88sl55zvfmSR517velQ9/+MN52tOelpmZmbzgBS/Y7a/VqbYAAACL7P7778/f//3f56ijjkqS3HjjjXn961+fa665Jvvuu2/+7M/+LJ/+9Kdz9dVXZ3Z2Nu95z3ty77335g/+4A/yiU98Iv/wD/+Q7373u9vd9h/90R/ll37pl/LlL385V199dZ7znOfk7LPPzuGHH54NGzbknHPOyWWXXZabbropV111VTZs2JD169fniiuuyPr163PBBRfkmmuuycc+9rF86UtfWpCvdyqOeAIAACwH99xzT1avXp1k7ojnG97whtxyyy15xjOekWOPPTZJ8sUvfjFf+cpXctxxxyVJfvSjH+WFL3xhbrjhhhx22GF55jOfmST57d/+7axZs+YnnuOzn/1sPvzhDyeZe03p/vvvn+9///uPWOeyyy7LZZddlqOPPjpJctddd+Wmm27K1q1bc/LJJ2efffZJkp96+u7OEJ4AAACLZNvXeG5r3333ffj2GCMvf/nLs3bt2kess2HDhlTVgswxxsjb3va2vPGNb3zE8r/4i79YsOfYllNtAQAAHkeOPfbYfP7zn8/Xvva1JMndd9+dr371q3n2s5+db3zjG7n55puT5CfC9CEve9nL8oEPfCBJ8sADD+SHP/xh9ttvv2zduvXhdX71V3815513Xu66664kybe//e3cdttt+cVf/MVcdNFFueeee7J169Z84hOfWJCvSXgCAAA8jszMzOT888/Pqaeemuc+97k59thjc8MNN2TlypVZs2ZNTjzxxLz4xS/OM57xjO0+/n3ve18+97nP5aijjsoLXvCCXH/99TnwwANz3HHH5cgjj8xb3/rWvOIVr8hrX/vavPCFL8xRRx2VV7/61dm6dWue//zn57d+67eyevXqnHLKKTn++OMX5GuqMcaCbGg+Zmdnx7p16xZ8u65qCwAA7MjGjRtzxBFHLPUYU2N7+7Oq1o8xZh+9riOeAAAAtBKeAAAAtBKeAAAAtBKeAADAHmMxr3EzzXZ2PwpPAABgj7By5crcfvvt4nM3jTFy++23Z+XKlfN+zIrGeQAAAB43Dj300GzevDlbtmxZ6lGWvZUrV+bQQw+d9/rCEwAA2CPsvffeOeyww5Z6jD2SU20BAABoJTwBAABoJTwBAABoJTwBAABoJTwBAABoJTwBAABoJTwBAABoJTwBAABoJTwBAABoJTwBAABoJTwBAABoJTwBAABoJTwBAABoJTwBAABoJTwBAABoJTwBAABoJTwBAABoJTwBAABotcPwrKqVVXVVVX25qq6vqndOlp9fVd+oqg2Tj9X94wIAALDcrJjHOvcl+eUxxl1VtXeSf6yqv5/c99YxxoV94wGw3Kw689K2bW86+8S2bQMAfXYYnmOMkeSuyad7Tz5G51AAAABMj3m9xrOq9qqqDUluS/KpMcaVk7veVVXXVtV7q+pn2qYEAABg2ZpXeI4xHhhjrE5yaJJjqurIJG9L8uwk/2eSA5L8m+09tqpOr6p1VbVuy5YtCzQ2AAAAy8VOXdV2jPGDJJcnOWGM8Z0x574kH0xyzGM8Zs0YY3aMMTszM7PbAwMAALC8zOeqtjNV9bOT209M8itJbqiqgyfLKslvJLmuc1AAAACWp/lc1fbgJB+qqr0yF6ofGWP816r6bFXNJKkkG5K8qXFOAAAAlqn5XNX22iRHb2f5L7dMBAAAwFTZqdd4AgAAwM4SngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALTaYXhW1cqquqqqvlxV11fVOyfLD6uqK6vqpqr6/6rqX/SPCwAAwHIznyOe9yX55THG85KsTnJCVR2b5D8kee8Y45lJvp/kDX1jAgAAsFztMDzHnLsmn+49+RhJfjnJhZPlH0ryGy0TAgAAsKzN6zWeVbVXVW1IcluSTyW5OckPxhj3T1bZnOSQx3js6VW1rqrWbdmyZSFmBgAAYBmZV3iOMR4YY6xOcmiSY5Icsb3VHuOxa8YYs2OM2ZmZmV2fFAAAgGVpp65qO8b4QZLLkxyb5GerasXkrkOT3LKwowEAADAN5nNV25mq+tnJ7Scm+ZUkG5N8LsmrJ6udluTjXUMCAACwfK3Y8So5OMmHqmqvzIXqR8YY/7WqvpLkgqr6syTXJDm3cU4AAACWqR2G5xjj2iRHb2f51zP3ek8AAAB4TDv1Gk8AAADYWfM51XbPdtb+Tdu9s2e7AAAAjzOOeAIAANBKeAIAANBKeAIAANBKeAIAANBKeAIAANBKeAIAANDK26kAALAkVp15adu2N519Ytu2gZ3niCcAAACthCcAAACthCcAAACt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jnien+SE7Sx/7xhj9eTj7xZ2LAAAAKbFDsNzjHFFkjsWYRYAAACm0O68xvPNVXXt5FTcJy/YRAAAAEyVXQ3PDyQ5PMnqJN9J8h8fa8WqOr2q1lXVui1btuzi0wEAALBc7VJ4jjFuHWM8MMZ4MMl/SnLMT1l3zRhjdowxOzMzs6tzAgAAsEztUnhW1cHbfHpykusea10AAAD2bCt2tEJVrU3ykiQHVdXmJO9I8pKqWp1kJNmU5I2NMwIAALCM7TA8xxinbmfxuQ2zAAAAMIV256q2AAAAsEPCEwAAgFbCEwAAgFbCEwAAgFbCEwAAgFbCEwAAgFbCEwAAgFbCEwAAgFbCEwAAgFbCEwAAgFbCEwAAgFYrlnoAgMW06sxLW7a7aWXLZgEApoIjngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALQSngAAALRasdQDAAA7b9WZl7Zsd9PKls0CsIdzxBMAAIBWOwzPqjqvqm6rquu2WXZAVX2qqm6a/Pvk3jEBAABYruZzxPP8JCc8atmZST4zxnhmks9MPgcAAICfsMPwHGNckeSORy3+9SQfmtz+UJLfWOC5AAAAmBK7+hrPp4wxvpMkk39/buFGAgAAYJq0X1yoqk6vqnVVtW7Lli3dTwcAAMDjzK6G561VdXCSTP697bFWHGOsGWPMjjFmZ2ZmdvHpAAAAWK52NTwvSXLa5PZpST6+MOMAAAAwbebzdiprk3whyb+sqs1V9YYkZyd5eVXdlOTlk88BAADgJ6zY0QpjjFMf466XLfAsAAAATKH2iwsBAACwZxOeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtBKeAAAAtFqxOw+uqk1JtiZ5IMn9Y4zZhRgKAACA6bFb4Tnx0jHG9xZgOwAAAEwhp9oCAADQanfDcyS5rKrWV9XpCzEQAAAA02V3T7U9boxxS1X9XJJPVdUNY4wrtl1hEqSnJ8nTn/703Xw6AAAAlpvdOuI5xrhl8u9tSS5Kcsx21lkzxpgdY8zOzMzsztMBAACwDO1yeFbVvlW130O3k7wiyXULNRgAAADTYXdOtX1Kkouq6qHt/JcxxicXZCoAAACmxi6H5xjj60met4CzAAAAMIW8nQoAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACthCcAAACtdis8q+qEqrqxqr5WVWcu1FAAAABMj10Oz6raK8n/k+SVSX4+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\n",
      "text/plain": [
       "<Figure size 1152x720 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "from matplotlib import pyplot as plt\n",
    "Y_pred = model.predict(X_test)\n",
    "Output = pd.DataFrame({'Actual': Y_test.flatten(), 'Predicted': Y_pred.flatten()})\n",
    "df1 = Output.head(10)\n",
    "df1.plot(kind='bar',figsize=(16,10))\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'me': 0.8273430448827166,\n",
       " 'mae': 7.0650715640117605,\n",
       " 'mse': 90.27993917697985,\n",
       " 'rmse': 9.501575615495561}"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Accuracy metrics\n",
    "import numpy as np\n",
    "def forecast_accuracy(forecast, actual):\n",
    "    me = np.mean(forecast - actual)             # ME\n",
    "    mae = np.mean(np.abs(forecast - actual))    # MAE\n",
    "    mse = np.mean ((forecast-actual)**2)      #MSE\n",
    "    rmse = np.mean((forecast - actual)**2)**.5  # RMSE                   \n",
    "    return({'me':me, 'mae': mae, 'mse':mse, 'rmse':rmse})\n",
    "\n",
    "forecast_accuracy(Y_pred, Y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "R2 (on testing data) : 0.33\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import r2_score\n",
    "r2 = r2_score(Y_test, Y_pred)\n",
    "print(\"R2 (on testing data) : {:.2}\".format(r2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\ASUS\\Anaconda3\\lib\\site-packages\\numpy\\core\\fromnumeric.py:2542: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.\n",
      "  return ptp(axis=axis, out=out, **kwargs)\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<table class=\"simpletable\">\n",
       "<caption>OLS Regression Results</caption>\n",
       "<tr>\n",
       "  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.466</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.465</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   329.4</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Date:</th>             <td>Sat, 13 Jun 2020</td> <th>  Prob (F-statistic):</th> <td>2.35e-53</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Time:</th>                 <td>11:02:30</td>     <th>  Log-Likelihood:    </th> <td> -1260.6</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>No. Observations:</th>      <td>   379</td>      <th>  AIC:               </th> <td>   2525.</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Df Residuals:</th>          <td>   377</td>      <th>  BIC:               </th> <td>   2533.</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   \n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>const</th> <td>  -34.4174</td> <td>    3.165</td> <td>  -10.876</td> <td> 0.000</td> <td>  -40.640</td> <td>  -28.195</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>x1</th>    <td>    9.0698</td> <td>    0.500</td> <td>   18.150</td> <td> 0.000</td> <td>    8.087</td> <td>   10.052</td>\n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "  <th>Omnibus:</th>       <td>47.338</td> <th>  Durbin-Watson:     </th> <td>   2.053</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td> 209.380</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Skew:</th>          <td> 0.408</td> <th>  Prob(JB):          </th> <td>3.42e-46</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Kurtosis:</th>      <td> 6.549</td> <th>  Cond. No.          </th> <td>    59.2</td>\n",
       "</tr>\n",
       "</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified."
      ],
      "text/plain": [
       "<class 'statsmodels.iolib.summary.Summary'>\n",
       "\"\"\"\n",
       "                            OLS Regression Results                            \n",
       "==============================================================================\n",
       "Dep. Variable:                      y   R-squared:                       0.466\n",
       "Model:                            OLS   Adj. R-squared:                  0.465\n",
       "Method:                 Least Squares   F-statistic:                     329.4\n",
       "Date:                Sat, 13 Jun 2020   Prob (F-statistic):           2.35e-53\n",
       "Time:                        11:02:30   Log-Likelihood:                -1260.6\n",
       "No. Observations:                 379   AIC:                             2525.\n",
       "Df Residuals:                     377   BIC:                             2533.\n",
       "Df Model:                           1                                         \n",
       "Covariance Type:            nonrobust                                         \n",
       "==============================================================================\n",
       "                 coef    std err          t      P>|t|      [0.025      0.975]\n",
       "------------------------------------------------------------------------------\n",
       "const        -34.4174      3.165    -10.876      0.000     -40.640     -28.195\n",
       "x1             9.0698      0.500     18.150      0.000       8.087      10.052\n",
       "==============================================================================\n",
       "Omnibus:                       47.338   Durbin-Watson:                   2.053\n",
       "Prob(Omnibus):                  0.000   Jarque-Bera (JB):              209.380\n",
       "Skew:                           0.408   Prob(JB):                     3.42e-46\n",
       "Kurtosis:                       6.549   Cond. No.                         59.2\n",
       "==============================================================================\n",
       "\n",
       "Warnings:\n",
       "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.\n",
       "\"\"\""
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import sklearn.model_selection \n",
    "\n",
    "X = dat['RM']\n",
    "Y = dat['Price']\n",
    "\n",
    "X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model\n",
    "\n",
    "X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.25, random_state = 5)\n",
    "X_train = X_train.values.reshape(-1,2)\n",
    "Y_train = Y_train.values.reshape(-1,1)\n",
    "X_test = X_test.values.reshape(-1,2)\n",
    "Y_test = Y_test.values.reshape(-1,1)\n",
    "\n",
    "\n",
    "# Note the difference in argument order\n",
    "model = sm.OLS(Y_train, X_train).fit() ## sm.OLS(output, input)\n",
    "\n",
    "# Print out the statistics\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1152x720 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "from matplotlib import pyplot as plt\n",
    "X_test = sm.add_constant(X_test)\n",
    "Y_pred = model.predict(X_test)\n",
    "Output = pd.DataFrame({'Actual': Y_test.flatten(), 'Predicted': Y_pred.flatten()})\n",
    "df1 = Output.head(10)\n",
    "df1.plot(kind='bar',figsize=(16,10))\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'me': 0.19893030499147174,\n",
       " 'mae': 8.44620333119356,\n",
       " 'mse': 125.71788209532215,\n",
       " 'rmse': 11.21239858796155}"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Accuracy metrics\n",
    "import numpy as np\n",
    "def forecast_accuracy(forecast, actual):\n",
    "    me = np.mean(forecast - actual)             # ME\n",
    "    mae = np.mean(np.abs(forecast - actual))    # MAE\n",
    "    mse = np.mean ((forecast-actual)**2)      #MSE\n",
    "    rmse = np.mean((forecast - actual)**2)**.5  # RMSE                   \n",
    "    return({'me':me, 'mae': mae, 'mse':mse, 'rmse':rmse})\n",
    "\n",
    "forecast_accuracy(Y_pred, Y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "R2 (on testing data) : 0.53\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import r2_score\n",
    "r2 = r2_score(Y_test, Y_pred)\n",
    "print(\"R2 (on testing data) : {:.2}\".format(r2))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Multivariate Linear Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = boston.data\n",
    "Y = boston.target\n",
    "X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.25, random_state = 5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "Y_train = Y_train.reshape(-1,1)\n",
    "Y_test = Y_test.reshape(-1,1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "lm.fit(X_train,Y_train)\n",
    "y_predicted = lm.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x1d3954a2470>]"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    },
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
    "plt.scatter(Y_test, y_predicted)\n",
    "plt.plot([Y_test.min(),Y_test.max()], [[Y_test.min()], [Y_test.max()]], color ='r', linewidth =2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA54AAAI7CAYAAACEKUQOAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjAsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+17YcXAAAgAElEQVR4nO3df5BldX3n/9dbhs0IUkSg4yKow7K6EkEH7SUokmiMBqNfEr6YimgiSZmglbUilS2z6G4KUhW32GA0ptyYml0Q3coOX4OCuCQu/mJJ/Co4AyMBB0TMGEcURlAclh/K8Nk/+sIOODg9M/3upi+PR9VU3z733HPffWqqu599zj23xhgBAACALk9Y6gEAAACYbsITAACAVsITAACAVsITAACAVsITAACAVsITAACAVisW88kOOuigsWrVqsV8SgAAABbJ+vXrvzPGmHnk8kUNz1WrVmXdunWL+ZQAAAAskqr6+o6WO9UWAACAVsITAACAVsITAACAVov6Gk8AAICl8sMf/jCbN2/Ovffeu9SjLHsrV67MoYcemr333nte6wtPAADgcWHz5s3Zb7/9smrVqlTVUo+zbI0xcvvtt2fz5s057LDD5vUYp9oCAACPC/fee28OPPBA0bmHqioHHnjgLh05Fp4AAMDjhuhcGLu6H4UnAADAIrroootSVbnhhht+7Hrnn39+brnllt1+nssvvzyvfvWrd/vxC8lrPAEAgMelVWdcuqDb23T2q+a13tq1a/PiF784F1xwQc4666xHXe/888/PkUcemac+9akLNOHSccQTAABgkdx111353Oc+l3PPPTcXXHDBQ8v/5E/+JEcddVSe97zn5YwzzsiFF16YdevW5fWvf31Wr16de+65J6tWrcp3vvOdJMm6devykpe8JEly1VVX5UUvelGOPvrovOhFL8qNN964FF/aj+WIJwAAwCK5+OKLc8IJJ+RZz3pWDjjggFx99dW59dZbc/HFF+fKK6/MPvvskzvuuCMHHHBA3ve+9+Vd73pXZmdnf+w2n/3sZ+eKK67IihUr8qlPfSrveMc78pGPfGSRvqL5EZ4AAACLZO3atTn99NOTJK997Wuzdu3aPPDAA/mt3/qt7LPPPkmSAw44YJe2eeedd+bUU0/NTTfdlKrKD3/4wwWfe08JTwAAgEVw++235zOf+Uyuu+66VFW2bduWqsrJJ588r6vErlixIg888ECSPOytTP7wD/8wL33pS3PRRRdl06ZND52C+1jiNZ4AAACL4MILL8wb3vCGfP3rX8+mTZvyjW98I4cddlgOOOCAnHfeebn77ruTJHfccUeSZL/99svWrVsfevyqVauyfv36JHnYqbR33nlnDjnkkCRzFyR6LBKeAAAAi2Dt2rU56aSTHrbs5JNPzi233JITTzwxs7OzWb16dd71rnclSX7zN38zb37zmx+6uNCZZ56Zt771rTn++OOz1157PbSNP/iDP8jb3/72HHfccdm2bduifk3zVWOMRXuy2dnZsW7dukV7PgAAgAdt3LgxRxxxxFKPMTV2tD+rav0Y40euhuSIJwAAAK2EJwAAAK2EJwAAAK2EJwAAAK2EJwAAAK2EJwAAAK2EJwAAwCLZa6+9snr16hx55JH51V/91dx99927va3LL788r371q5Mkl1xySc4+++xHXfd73/te/uIv/mKXn+Oss8566H1F98SKPd4CAADAcnTW/gu8vTt3usoTn/jEbNiwIUny+te/Pn/5l3+Z3//933/o/jFGxhh5whN27RjhiSeemBNPPPFR738wPH/3d393l7a7UKYiPFedcWnbtjetfF3PhufxnxIAAJhexx9/fK699tps2rQpr3zlK/PSl740n//853PxxRfnxhtvzJlnnpn77rsvhx9+eD7wgQ/kSU96Uj7xiU/k9NNPz0EHHZTnP//5D23r/PPPz7p16/K+970vt956a9785jfna1/7WpLk/e9/f/78z/88N998c1avXp2Xv/zlOeecc3LOOefkwx/+cO67776cdNJJ+aM/+qMkyTvf+c586EMfytOe9rTMzMzkBS94wR5/rU61BQAAWGT3339//vZv/zZHHXVUkuTGG2/MG97whlxzzTXZd99988d//Mf51Kc+lauvvjqzs7N597vfnXvvvTe/8zu/k49//OP5u7/7u3z729/e4bZ/7/d+Lz/3cz+XL33pS7n66qvznOc8J2effXYOP/zwbNiwIeecc04uu+yy3HTTTbnqqquyYcOGrF+/PldccUXWr1+fCy64INdcc00++tGP5otf/OKCfL1TccQTAABgObjnnnuyevXqJHNHPN/4xjfmlltuyTOe8Ywce+yxSZIvfOEL+fKXv5zjjjsuSfKDH/wgL3zhC3PDDTfksMMOyzOf+cwkya//+q9nzZo1P/Icn/nMZ/KhD30oydxrSvfff/9897vffdg6l112WS677LIcffTRSZK77rorN910U7Zu3ZqTTjop++yzT5L82NN3d4XwBAAAWCTbv8Zze/vuu+9Dt8cYefnLX561a9c+bJ0NGzakqhZkjjFG3v72t+dNb3rTw5b/2Z/92YI9x/acagsAAPAYcuyxx+Zzn/tcvvrVryZJ7r777nzlK1/Js5/97PzjP/5jbr755iT5kTB90Mte9rK8//3vT5Js27Yt3//+97Pffvtl69atD63zi7/4iznvvPNy1113JUm++c1v5rbbbsvP/uzP5qKLLso999yTrVu35uMf//iCfE3CEwAA4DFkZmYm559/fk455ZQ897nPzbHHHpsbbrghK1euzJo1a/KqV70qL37xi/OMZzxjh49/73vfm89+9rM56qij8oIXvCDXX399DjzwwBx33HE58sgj87a3vS2veMUr8rrXvS4vfOELc9RRR+U1r3lNtm7dmuc///n5tV/7taxevTonn3xyjj/++AX5mmqMsSAbmo/Z2dmxbt26Bd+uq9oCAAA7s3HjxhxxxBFLPcbU2NH+rKr1Y4zZR67riCcAAACthCcAAACthCcAAACthCcAAPC4sZjXuJlmu7ofhScAAPC4sHLlytx+++3icw+NMXL77bdn5cqV837MisZ5AAAAHjMOPfTQbN68OVu2bFnqUZa9lStX5tBDD533+sITAAB4XNh7771z2GGHLfUYj0tOtQUAAKCV8AQAAKCV8AQAAKCV8AQAAKCV8AQAAKCV8AQAAKCV8AQAAKCV8AQAAKCV8AQAAKCV8AQAAKCV8AQAAKCV8AQAAKCV8AQAAKCV8AQAAKCV8AQAAKCV8AQAAKDViqUeAADm7az9m7Z7Z892AYAkjngCAADQTHgCAADQaqen2lbVyiRXJPmJyfoXjjHOrKrzk/xckgfPT/rNMcaGrkEBWB5WnXFp27Y3rWzbNADQaD6v8bwvyc+PMe6qqr2T/H1V/e3kvreNMS7sGw8AAIDlbqfhOcYYSe6afLr35N/oHAoAAIDpMa/XeFbVXlW1IcltST45xrhyctc7q+raqnpPVf3Eozz2tKpaV1XrtmzZskBjAwAAsFzMKzzHGNvGGKuTHJrkmKo6Msnbkzw7yb9OckCSf/coj10zxpgdY8zOzMws0NgAAAAsF7t0VdsxxveSXJ7khDHGt8ac+5J8IMkxDfMBAACwzO00PKtqpqp+cnL7iUl+IckNVXXwZFkl+ZUk13UOCgAAwPI0n6vaHpzkg1W1V+ZC9cNjjP9RVZ+pqpkklWRDkjc3zgkAAMAyNZ+r2l6b5OgdLP/5lokAAACYKrv0Gk8AAADYVcITAACAVsITAACAVsITAACAVsITAACAVsITAACAVsITAACAVsITAACAVsITAACAVsITAACAVsITAACAVsITAACAVsITAACAVsITAACAVsITAACAVsITAACAVsITAACAVsITAACAVsITAACAVsITAACAVsITAACAVsITAACAVsITAACAVsITAACAVsITAACAVsITAACAVsITAACAVsITAACAVsITAACAVsITAACAVsITAACAVsITAACAVsITAACAVsITAACAViuWegDIWfs3bffOnu0CAAC7xBFPAAAAWglPAAAAWglPAAAAWglPAAAAWglPAAAAWglPAAAAWnk7FQAApo+3a4PHFEc8AQAAaCU8AQAAaCU8AQAAaCU8AQAAaCU8AQAAaCU8AQAAaCU8AQAAaCU8AQAAaCU8AQAAaCU8AQAAaCU8AQAAaCU8AQAAaCU8AQAAaCU8AQAAaCU8AQAAaCU8AQAAaCU8AQAAaLXT8KyqlVV1VVV9qaqur6o/miw/rKqurKqbqur/q6p/1j8uAAAAy818jnjel+TnxxjPS7I6yQlVdWyS/5TkPWOMZyb5bpI39o0JAADAcrXT8Bxz7pp8uvfk30jy80kunCz/YJJfaZkQAACAZW1er/Gsqr2qakOS25J8MsnNSb43xrh/ssrmJIf0jAgAAMByNq/wHGNsG2OsTnJokmOSHLGj1Xb02Ko6rarWVdW6LVu27P6kAAAALEu7dFXbMcb3klye5NgkP1lVKyZ3HZrklkd5zJoxxuwYY3ZmZmZPZgUAAGAZms9VbWeq6icnt5+Y5BeSbEzy2SSvmax2apKPdQ0JAADA8rVi56vk4CQfrKq9MheqHx5j/I+q+nKSC6rqj5Nck+TcxjkBAABYpnYanmOMa5McvYPlX8vc6z0BAADgUe3SazwBAABgVwlPAAAAWglPAAAAWglPAAAAWglPAAAAWglPAAAAWglPAAAAWglPAAAAWglPAAAAWq1Y6gEAAHh8WnXG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\n",
      "text/plain": [
       "<Figure size 1152x720 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "Output = pd.DataFrame({'Actual': Y_test.flatten(), 'Predicted': y_predicted.flatten()})\n",
    "df1 = Output.head(10)\n",
    "df1.plot(kind='bar',figsize=(16,10))\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'me': -0.47313826074083787,\n",
       " 'mae': 3.290018352688552,\n",
       " 'mse': 24.2746083116879,\n",
       " 'rmse': 4.9269268628312215}"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Accuracy metrics\n",
    "import numpy as np\n",
    "def forecast_accuracy(forecast, actual):\n",
    "    me = np.mean(forecast - actual)             # ME\n",
    "    mae = np.mean(np.abs(forecast - actual))    # MAE\n",
    "    mse = np.mean ((forecast-actual)**2)      #MSE\n",
    "    rmse = np.mean((forecast - actual)**2)**.5  # RMSE                   \n",
    "    return({'me':me, 'mae': mae, 'mse':mse, 'rmse':rmse})\n",
    "\n",
    "forecast_accuracy(y_predicted, Y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "R2 (on testing data) : 0.71\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import r2_score\n",
    "r2 = r2_score(Y_test, y_predicted)\n",
    "print(\"R2 (on testing data) : {:.2}\".format(r2))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Polynomial Linear Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = dat['RM'].values.reshape(-1,1)\n",
    "Y = dat['Price'].values.reshape(-1,1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import PolynomialFeatures\n",
    "transformer = PolynomialFeatures(degree=2, include_bias=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PolynomialFeatures(degree=2, include_bias=False, interaction_only=False,\n",
       "                   order='C')"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "transformer.fit(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_ = transformer.transform(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 6.575    43.230625]\n",
      " [ 6.421    41.229241]\n",
      " [ 7.185    51.624225]\n",
      " ...\n",
      " [ 6.976    48.664576]\n",
      " [ 6.794    46.158436]\n",
      " [ 6.03     36.3609  ]]\n"
     ]
    }
   ],
   "source": [
    "print(X_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.25, random_state = 5)\n",
    "X_train = X_train.reshape(-1,1)\n",
    "Y_train = Y_train.reshape(-1,1)\n",
    "X_test = X_test.reshape(-1,1)\n",
    "Y_test = Y_test.reshape(-1,1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = LinearRegression().fit(X_train, Y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_predicted = model.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x1d3947e7630>]"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    },
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
    "plt.scatter(Y_test, y_predicted)\n",
    "plt.plot([Y_test.min(),Y_test.max()], [[Y_test.min()], [Y_test.max()]], color ='r', linewidth =2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1152x720 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "Output = pd.DataFrame({'Actual': Y_test.flatten(), 'Predicted': y_predicted.flatten()})\n",
    "df1 = Output.head(10)\n",
    "df1.plot(kind='bar',figsize=(16,10))\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'me': 0.19893030499139885,\n",
       " 'mae': 3.9352380033262495,\n",
       " 'mse': 38.39960082194633,\n",
       " 'rmse': 6.1967411453074535}"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Accuracy metrics\n",
    "import numpy as np\n",
    "def forecast_accuracy(forecast, actual):\n",
    "    me = np.mean(forecast - actual)             # ME\n",
    "    mae = np.mean(np.abs(forecast - actual))    # MAE\n",
    "    mse = np.mean ((forecast-actual)**2)      #MSE\n",
    "    rmse = np.mean((forecast - actual)**2)**.5  # RMSE                   \n",
    "    return({'me':me, 'mae': mae, 'mse':mse, 'rmse':rmse})\n",
    "\n",
    "forecast_accuracy(y_predicted, Y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "R2 (on testing data) : 0.53\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import r2_score\n",
    "r2 = r2_score(Y_test, y_predicted)\n",
    "print(\"R2 (on testing data) : {:.2}\".format(r2))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
