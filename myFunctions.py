from operator import index
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

def get_dummies(df, categoricalNumericFeatures=[], label= ''):
    
    # get dummies for all categorical datatypes or if it is in our list, and if it is not the label
    for col in df:
        if (not pd.api.types.is_numeric_dtype(df[col]) or categoricalNumericFeatures.count(col) > 0) and not col == label:
            df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=True)
    return df

def bin_groups(df, categoricalNumericFeatures=[], percent=.05):
    import pandas as pd

    for col in df:
        # if the column is categorical
        if not pd.api.types.is_numeric_dtype(df[col]) or categoricalNumericFeatures.count(col) > 0:
            for group, count in df[col].value_counts().iteritems():
                # For each categorical value, if the percentage of the column that contains that value is below the threshold, relabel to other
                if count / len(df) < percent:
                    df.loc[df[col] == group, col] = 'Other'
    return df

def drop_cols_too_unique(df, percent = .80):

    for col in df:
        pct = len(df[df[col] == 'Other'][col]) / len(df)
        if pct >=.80:
            df.drop(columns=col, inplace=True)
            print("dropped " + col + ", " + str(pct * 100) + "% unique")
    return df     

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
                
                raise ValueError(f"{self.categoricalImputer} is an invalid categorical imputer.")
        
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
        
            dfCategorical = pd.DataFrame(imp.fit_transform(dfCategorical), columns=dfCategorical.columns,index=dfCategorical.index)

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
            dfNumeric = pd.DataFrame(imp.fit_transform(dfNumeric), columns=dfNumeric.columns,index=dfNumeric.index)

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

class ModelSelection:
    
    def fit_crossvalidate_clf(df, label, k=10, r=5, repeat=True):
        # this is someone else's code. Found in Dr. Keith's machine learning in python book, but I believe he copied it from somewhere else.
        
        import sklearn.linear_model as lm, pandas as pd, sklearn.ensemble as se, numpy as np
        from sklearn.model_selection import KFold, RepeatedKFold, cross_val_score
        from numpy import mean, std
        from sklearn import svm
        from sklearn import gaussian_process
        from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from sklearn import svm
        from sklearn.naive_bayes import CategoricalNB
        # from xgboost import XGBClassifier
        from sklearn import preprocessing
        from sklearn.neural_network import MLPClassifier

        X = df.drop(columns=[label])
        y = df[label]

        if repeat:
            cv = RepeatedKFold(n_splits=k, n_repeats=r, random_state=12345)
        else:
            cv = KFold(n_splits=k, random_state=12345, shuffle=True)
        
        fit = {}    # Use this to store each of the fit metrics
        models = {} # Use this to store each of the models
        
        # Create the model objects
        model_log = lm.LogisticRegression(max_iter=100)
        model_logcv = lm.RidgeClassifier()
        model_sgd = lm.SGDClassifier(max_iter=1000, tol=1e-3)
        model_pa = lm.PassiveAggressiveClassifier(max_iter=1000, random_state=12345, tol=1e-3)
        model_per = lm.Perceptron(fit_intercept=False, max_iter=10, tol=None, shuffle=False)
        model_knn = KNeighborsClassifier(n_neighbors=3)
        model_svm = svm.SVC(decision_function_shape='ovo') # Remove the parameter for two-class model
        model_nb = CategoricalNB()
        model_bag = se.BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
        model_ada = se.AdaBoostClassifier(n_estimators=100, random_state=12345)
        model_ext = se.ExtraTreesClassifier(n_estimators=100, random_state=12345)
        model_rf = se.RandomForestClassifier(n_estimators=10)
        model_hgb = se.HistGradientBoostingClassifier(max_iter=100)
        model_vot = se.VotingClassifier(estimators=[('lr', model_log), ('rf', model_ext), ('gnb', model_hgb)], voting='hard')
        model_gb = se.GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        estimators = [('ridge', lm.RidgeCV()), ('lasso', lm.LassoCV(random_state=12345)), ('knr', KNeighborsRegressor(n_neighbors=20, metric='euclidean'))]
        final_estimator = se.GradientBoostingRegressor(n_estimators=25, subsample=0.5, min_samples_leaf=25, max_features=1, random_state=12345)
        model_st = se.StackingRegressor(estimators=estimators, final_estimator=final_estimator)
        # model_xgb = XGBClassifier()
        model_nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=12345)

        # Fit a crss-validated R squared score and add it to the dict
        fit['Logistic'] = mean(cross_val_score(model_log, X, y, scoring='accuracy', cv=cv, n_jobs=-1))
        fit['Ridge'] = mean(cross_val_score(model_logcv, X, y, scoring='accuracy', cv=cv, n_jobs=-1))
        fit['SGD'] = mean(cross_val_score(model_sgd, X, y, scoring='accuracy', cv=cv, n_jobs=-1))
        fit['PassiveAggressive'] = mean(cross_val_score(model_pa, X, y, scoring='accuracy', cv=cv, n_jobs=-1))
        fit['Perceptron'] = mean(cross_val_score(model_per, X, y, scoring='accuracy', cv=cv, n_jobs=-1))
        fit['KNN'] = mean(cross_val_score(model_knn, X, y, scoring='accuracy', cv=cv, n_jobs=-1))
        fit['SVM'] = mean(cross_val_score(model_svm, X, y, scoring='accuracy', cv=cv, n_jobs=-1))
        fit['NaiveBayes'] = mean(cross_val_score(model_nb, X, y, scoring='accuracy', cv=cv, n_jobs=-1))
        fit['Bagging'] = mean(cross_val_score(model_bag, X, y, scoring='accuracy', cv=cv, n_jobs=-1))
        fit['AdaBoost'] = mean(cross_val_score(model_ada, X, y, scoring='accuracy', cv=cv, n_jobs=-1))
        fit['ExtraTrees'] = mean(cross_val_score(model_ext, X, y, scoring='accuracy', cv=cv, n_jobs=-1))
        fit['RandomForest'] = mean(cross_val_score(model_rf, X, y, scoring='accuracy', cv=cv, n_jobs=-1))
        fit['HistGradient'] = mean(cross_val_score(model_hgb, X, y, scoring='accuracy', cv=cv, n_jobs=-1))
        fit['Voting'] = mean(cross_val_score(model_vot, X, y, scoring='accuracy', cv=cv, n_jobs=-1))
        fit['GradBoost'] = mean(cross_val_score(model_gb, X, y, scoring='accuracy', cv=cv, n_jobs=-1))
        # fit['XGBoost'] = mean(cross_val_score(model_xgb, X, y, scoring='accuracy', cv=cv, n_jobs=-1))
        fit['NeuralN'] = mean(cross_val_score(model_nn, X, y, scoring='accuracy', cv=cv, n_jobs=-1))

        # Add the model to another dict; make sure the keys have the same names as the list above
        models['Logistic'] = model_log
        models['Ridge'] = model_logcv
        models['SGD'] = model_sgd
        models['PassiveAggressive'] = model_pa
        models['Perceptron'] = model_per
        models['KNN'] = model_knn
        models['SVM'] = model_svm
        models['NaiveBayes'] = model_nb
        models['Bagging'] = model_bag
        models['AdaBoost'] = model_ada
        models['ExtraTrees'] = model_ext
        models['RandomForest'] = model_rf
        models['HistGradient'] = model_hgb
        models['Voting'] = model_vot
        models['GradBoost'] = model_gb
        # models['XGBoost'] = model_xgb
        models['NeuralN'] = model_nn

            # Add the fit dictionary to a new DataFrame, sort, extract the top row, use it to retrieve the model object from the models dictionary
        df_fit = pd.DataFrame({'Accuracy':fit})
        df_fit.sort_values(by=['Accuracy'], ascending=False, inplace=True)
        best_model = df_fit.index[0]
        print(f"Best model was {best_model}, with an estimated {round((df_fit.at[df_fit.index[0],'Accuracy'])*100,2)}% correct test set classification rate")

        return models[best_model].fit(X, y)