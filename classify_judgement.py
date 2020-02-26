# please go to https://github.com/teijag/legal_text_classification/blob/master/legal%20area%20classification.ipynb
# for some visualizations, chi2 contingency analysis, model tuning and more.

# data ingestion
import pandas as pd
import numpy as np
import glob

# pre-processing
import spacy
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from imblearn.over_sampling import RandomOverSampler, SMOTE

# modeling
from sklearn.linear_model import LogisticRegression

class ClassifyJudgement():

    def __init__(self):

        self.path = "Fixed Judgements/"
        self.df = pd.read_csv("Interview_Mapping.csv")
        self.label_encoder = LabelEncoder()
        self.df_train, self.df_test = self.prepare_dataset()

    # populate the dataframe
    def populate_df(self):

        # initialize lemmatizer
        nlp = spacy.load("en",disable=(['ner', 'parser','tag','pos','shape']))
        # join judgement contents in a folder to df
        for file in glob.glob(self.path + "*.txt"):  # taking out the file name to match the content to df
            title = file.split(".")[0].split("/")[1]
            with open(file, 'rb') as f:
                contents = f.read()
            if title in self.df["Judgements"].values:  # populating values when filename matches the judgement title
                doc = nlp(str(contents))  # lemmatization
                self.df.loc[self.df["Judgements"] == title, "Content"] = ' '.join([token.lemma_ for token in doc])
            else:
                print("File not found in the mapping")
                continue
        return self.df

    # prepare the train and test set
    def prepare_dataset(self):

        # add an empty column to be populated
        self.df["Content"] = None
        df = self.populate_df()

        # split the dataframe based on whether the row has a label
        df_test = df[df["Area.of.Law"] == "To be Tested"]
        df_train = df[df["Area.of.Law"] != "To be Tested"]

        # in train, convert label column into numbers
        df_train["category"] = self.label_encoder.fit_transform(df_train[["Area.of.Law"]])

        return (df_train, df_test)

    # split train dataset
    def split_set(self):

        le = LabelEncoder()
        self.df_train["category"] = le.fit_transform(self.df_train["Area.of.Law"])
        # train test split
        X = self.df_train["Content"]
        y = self.df_train["category"]

        return train_test_split(X, y)

    # train model
    def train_model(self):

        ### pre-processing
        # get train test split
        X_train, X_test, y_train, y_test = self.split_set()

        # count vectorize train data
        count_vect = CountVectorizer(ngram_range=(1, 2), max_features=1500, max_df = 0.5,stop_words = 'english')
        X_train_counts = count_vect.fit_transform(X_train)

        # tfidf transform count data
        tfidf_transformer = TfidfTransformer(sublinear_tf = True)
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

        # over sample classes with fewer samples than SMOTE requires
        min_classes = pd.Series(y_train).value_counts().index[-15:].tolist()
        min_require = np.repeat(10, 15)
        min_dict = dict(zip(min_classes, min_require))

        ros = RandomOverSampler(random_state=0,sampling_strategy=min_dict)
        X_train_ros, y_train_ros = ros.fit_resample(X_train_tfidf, y_train)

        # over sample all classes with synthetically generated data
        max_class = np.argmax(pd.Series(y_train_ros).value_counts())

        smote = SMOTE(random_state=0, sampling_strategy={max_class: 400})

        X_train_smote, y_train_smote = smote.fit_resample(X_train_ros, y_train_ros)

        smote = SMOTE(random_state=0, sampling_strategy="all")

        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_smote, y_train_smote)

        ### model

        # use parameters from grid search best params - see notebook for more
        clf = LogisticRegression(C = 1, n_jobs=-1,solver = 'saga', multi_class = 'multinomial')

        clf.fit(X_train_resampled, y_train_resampled)

        return (count_vect, tfidf_transformer, clf)

    # make prediction
    def predict_judgement(self):

        # get trained model and transformers
        count_vect, tfidf_transformer, clf = self.train_model()

        # transform test data using existing transformers
        test_X = self.df_test["Content"]
        test_X_counts = count_vect.transform(test_X)
        test_X_tfidf = tfidf_transformer.transform(test_X_counts)

        # predict using the chosen model - see model evaluation in notebook
        pred_y = clf.predict(test_X_tfidf)

        # transform encoded y back to Areas of Law
        label_y = self.label_encoder.inverse_transform(pred_y)

        result_df = pd.DataFrame()
        result_df["Judgements"] = self.df_test["Judgements"]
        result_df["Prediction"] = label_y
        result_df.to_csv("Teija_Gui_results.csv", index=False)


if __name__ == '__main__':
    cj = ClassifyJudgement()
    cj.predict_judgement()



