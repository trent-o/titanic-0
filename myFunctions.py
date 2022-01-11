import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler


class MissingData:


    def __init__(self, df, categoricalImputer='', numericImputer='', categoricalNumericFeatures=[], dropColPct = .5):
        self.categoricalImputer = categoricalImputer
        self.numericImputer = numericImputer
        self.df = df
        self.categoricalNumericFeatures = categoricalNumericFeatures
        self.dropColPct = dropColPct

    def fillMissingData(self):
        # get number of categorical and numeric columns missing data
        nCat, nNum = self.missing_values_check(self.df,self.categoricalNumericFeatures)

        # if no columns are missing data, we are done
        if nCat + nNum == 0:
            print("Missing data check complete. No columns have missing data.")
            return
        
        # drop columns missing too much data
        self.df = self.drop_cols_missing_data(self.df,self.dropColPct)

        # handle categorical columns missing data
        if nCat != 0:
            if self.categoricalImputer == 'most_frequent':
                imp = self.impute_mode()
            else:
                raise ValueError("Invalid categorical imputer.")
        
            self.df = self.selective_sklearn_impute(self.df, imp,'categorical',self.categoricalNumericFeatures)
            print(f"imputed {nCat} categorical column(s) using {self.categoricalImputer}.")
        
        # handle numeric columns missing data
        if nNum != 0: 
            if self.numericImputer == 'mean':
                imp = self.impute_mean()
            elif self.numericImputer == 'knn':
                imp = self.impute_KNN()
            elif self.numericImputer == 'reg':
                imp = self.impute_reg()
            else: 
                raise ValueError("Invalid numeric imputer.")
            
            self.df = self.selective_sklearn_impute(self.df,imp,'numeric',self.categoricalNumericFeatures)
            print(f"imputed {nNum} numeric column(s) using {self.numericImputer}.")

        # Check that filling in the missing data worked
        # get number of categorical and numeric columns missing data
        nCat, nNum = self.missing_values_check(self.df,self.categoricalNumericFeatures)

        # if no columns are missing data, we are done
        if nCat + nNum == 0:
            print("Missing data check complete. No columns have missing data.")
        else:
            raise ValueError(f"Missing data check failed. {nCat} categorical columns and {nNum} numeric columns have missing data.")
        return self.df

    def missing_values_report(self):
        df = self.df

        for col in df:
            print(f'{col}\t{round(df[col].isnull().sum() / len(df) * 100,2)}%')

    def missing_values_check(self, df, categoricalNumericFeatures):
        catColsMissingData = 0
        numColsMissingData = 0
        
        for col in df:
            #if there is at least one null value in the column
            if df[col].isnull().sum() > 0:

                #if its a categorical column
                if not pd.api.types.is_numeric_dtype(df[col]) or col in categoricalNumericFeatures:
                    catColsMissingData += 1
                else:
                    numColsMissingData +=1
        
        return catColsMissingData, numColsMissingData

    def drop_cols_missing_data(self, df, percent=.5):
        for col in df:
            if df[col].isna().sum()/len(df) > percent:
                df = df.drop(columns=col)
                print("dropped the " + col + " column. Over " + str(round(percent*100)) + "% of records were Null." )
        return df

    # IMPUTATION 
    def selective_sklearn_impute(self, df, imp, type, categoricalNumericFeatures = []):
        validTypes = ['categorical', 'numeric']
        if type not in validTypes:
            raise ValueError("Invalid type. Expected one of: %s" % validTypes)
        
        if type == 'categorical':
            # Only impute on categorical columns
            dfCategorical = df.select_dtypes(include='object')
            # if there are categorical features that are numerical, include them (ex: 1, 0 for survived)
            if len(categoricalNumericFeatures) != 0:
                for colName in categoricalNumericFeatures:
                    dfCategorical = dfCategorical.join(df[colName])
        
            dfCategorical = pd.DataFrame(imp.fit_transform(dfCategorical), columns=dfCategorical.columns)

            # Replace the old categorical columns with the new imputed ones.
            df.drop(columns=dfCategorical.columns, inplace=True)
            df = df.join(dfCategorical)
        elif type == 'numeric':
            # Only impute on numeric columns
            dfNumeric = df.select_dtypes(include='number')
            # if there are categorical features that are numerical, don't include them (ex: 1, 0 for survived)
            if len(categoricalNumericFeatures) != 0:
                for colName in categoricalNumericFeatures:
                    if colName in dfNumeric.columns:
                        dfNumeric = dfNumeric.drop(columns=colName)
            dfNumeric = pd.DataFrame(imp.fit_transform(dfNumeric), columns=dfNumeric.columns)

            # Replace the old numeric columns with the new imputed ones.
            df.drop(columns=dfNumeric.columns, inplace=True)
            df = df.join(dfNumeric)
        return df

    ## CATEGORICAL IMPUTATION
    def impute_mode(self):

        imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

        return imp

    ## NUMERICAL IMPUTATION
    def impute_mean(self):

        # Change the strategy to mean, median, or mode
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')

        return imp

    def impute_KNN(self):
        # Clustering is biased by unstandardized data; so MinMax scale it
        #df = pd.DataFrame(MinMaxScaler().fit_transform(df), columns = df.columns)

        print("did you standardize the data before impute KNN?")
        imp = KNNImputer(n_neighbors=5, weights="uniform")
        return imp

    def impute_reg(self):
        # Scaling is unnecessary for regression-based imputation
        imp = IterativeImputer(max_iter=10, random_state=12345)
        return imp
