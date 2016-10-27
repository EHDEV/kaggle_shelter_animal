from __future__ import division
import numpy as np
import pandas as p
import seaborn as sns
# import load_data
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# AnimalID,Name,DateTime,OutcomeType,OutcomeSubtype,AnimalType,SexuponOutcome,AgeuponOutcome,Breed,Color
# A671945,Hambone,2014-02-12 18:22:00,Return_to_owner,,Dog,Neutered Male,1 year,Shetland Sheepdog Mix,Brown/White
# A656520,Emily,2013-10-13 12:44:00,Euthanasia,Suffering,Cat,Spayed Female,1 year,Domestic Shorthair Mix,Cream Tabby



def scale(var):
    return MinMaxScaler(var)

def get_sex(x):
    x = str(x)
    if x.find('Male') >= 0: return 'male'
    if x.find('Female') >= 0: return 'female'
    return 'sex_unknown'


def get_neutered(x):
    x = str(x)
    if x.find('Spayed') >= 0: return 'neutered'
    if x.find('Neutered') >= 0: return 'neutered'
    if x.find('Intact') >= 0: return 'intact'
    return 'neuter_unknown'


def get_mix(x):
    x = str(x)
    if x.find('Mix') >= 0 or len(x.split('/')) > 1: return 1
    return 0

def get_breed0(x):
    x = str(x)
    blist = x.split('/')
    if len(blist) <= 1:
        return x.split('Mix')[0]

    try:
        return blist[0]
    except IndexError:
        return x

def get_breed1(x):
    x = str(x)
    blist = x.split('/')
    # if len(blist) <= 1 and x.find('Mix'):
    #     blist = x.split('Mix')
    #     try:
    #         return blist[0]
    #     except IndexError:
    #         return
    # else:
    #     try:
    #         return blist[1]
    #     except IndexError:
    #         return
    if len(blist) > 1:
        return blist[1]
    else:
        return

def get_breed2(x):
    x = str(x)
    blist = x.split('/')


def calc_age_in_years(x):
    x = str(x)
    if x == 'nan': return 0
    age = int(x.split()[0])
    if x.find('year') > -1: return age
    if x.find('month') > -1: return age / 12.
    if x.find('week') > -1: return age / 52.
    if x.find('day') > -1:
        return age / 365.
    else:
        return 0


def calc_age_in_months(x):
    x = str(x)
    if x == 'nan': return 0
    age = int(x.split()[0])
    if x.find('year') > -1: return age * 12.0
    if x.find('month') > -1: return age
    if x.find('week') > -1: return age * 4.0
    if x.find('day') > -1:
        return age * 30.
    else:
        return 0


def get_color0(x):
    clist = x.split('/')
    return clist[0]

def get_color1(x):
    clist = x.split('/')
    try:
        return clist[1]
    except IndexError:
        return None

def encode_target(target):
    le = LabelEncoder()
    le.fit(target)
    return le.transform(target), list(le.classes_)


def calc_age_category(x):
    if x < 3: return 'young'
    if x < 5: return 'young adult'
    if x < 10: return 'adult'
    return 'old'


def dummiefy(var):
    return p.get_dummies(var)


def load_transform():
    rem_cols = ['Name', 'AnimalID', 'ID', 'DateTime', 'OutcomeSubtype']

    data = p.read_csv('../data/train.csv')
    test_data = p.read_csv('../data/test.csv')
    dframes = [data, test_data]
    pets_list = []
    features = []

    for idx, pets in enumerate(dframes):


    # id_name_date_st = pets[rem_cols]

        for c in rem_cols:
            try:
                del pets[c]
            except KeyError:
                continue

        pets['Mix'] = pets.Breed.apply(get_mix)
        # pets['Breed0'] = pets.Breed.apply(get_breed0)
        # pets['Breed1'] = pets.Breed.apply(get_breed1)
        # pets['Color0'] = pets.Color.apply(get_color0)
        # pets['Color1'] = pets.Color.apply(get_color1)

        # del pets['Breed']
        # del pets['Color']
        # sns.countplot(pets.Mix, palette='Set3')

        pets['Sex'] = pets.SexuponOutcome.apply(get_sex)
        pets['Neutered'] = pets.SexuponOutcome.apply(get_neutered)
        del pets['SexuponOutcome']
        pets['AgeInMonths'] = pets.AgeuponOutcome.apply(calc_age_in_months)
        del pets['AgeuponOutcome']
        pets = p.concat((pets, p.DataFrame(dummiefy(pets.Neutered))), axis=1)
        del pets['Neutered']
        pets = p.concat((pets, p.DataFrame(dummiefy(pets.Breed))), axis=1)
        del pets['Breed']
        # pets = p.concat((pets, p.DataFrame(dummiefy(pets.Breed1))), axis=1)
        # del pets['Breed1']
        pets = p.concat((pets, p.DataFrame(dummiefy(pets.Sex))), axis=1)
        del pets['Sex']
        pets = p.concat((pets, p.DataFrame(dummiefy(pets.Color))), axis=1)
        del pets['Color']
        # pets['AgeCategory'] = pets.AgeInYears.apply(calc_age_category)
        pets[pets.AnimalType == 'Cat'].AgeInMonths.fillna(pets.groupby('AnimalType')['AgeInMonths'].mean().astype('int64')['Cat'], inplace=True)
        pets[pets.AnimalType == 'Dog'].AgeInMonths.fillna(pets.groupby('AnimalType')['AgeInMonths'].mean().astype('int64')['Dog'], inplace=True)
        del pets['AnimalType']

        if not idx:

            y, classes_ = encode_target(pets['OutcomeType'])
            del pets['OutcomeType']

        pets_list += [pets]

        features += list(pets.columns)

    features = set(features)

    pets_pets = []
    for pts in pets_list:
        for feat in features:
            if feat not in pts.columns:
                pts[feat] = p.Series(np.repeat(0, pts.shape[0]))

        pets_pets += [pts]

    X = pets_pets[0].values
    X_submission = pets_pets[1].values

    return X, y, X_submission


def load_transform_test():
    rem_cols = ['Name', 'ID', 'DateTime']

    data = p.read_csv('../data/test.csv')
    pets = data

    id_name_date_st = pets[rem_cols]

    for c in rem_cols:
        del pets[c]

    pets['Mix'] = pets.Breed.apply(get_mix)
    pets['Breed0'] = pets.Breed.apply(get_breed0)
    pets['Breed1'] = pets.Breed.apply(get_breed1)
    pets['Color0'] = pets.Color.apply(get_color0)
    pets['Color1'] = pets.Color.apply(get_color1)

    del pets['Breed']
    del pets['Color']
    # sns.countplot(pets.Mix, palette='Set3')

    pets['Sex'] = pets.SexuponOutcome.apply(get_sex)
    pets['Neutered'] = pets.SexuponOutcome.apply(get_neutered)
    del pets['SexuponOutcome']
    pets['AgeInMonths'] = pets.AgeuponOutcome.apply(calc_age_in_months)
    del pets['AgeuponOutcome']
    pets = p.concat((pets, p.DataFrame(dummiefy(pets.Neutered))), axis=1)
    del pets['Neutered']
    pets = p.concat((pets, p.DataFrame(dummiefy(pets.Breed0))), axis=1)
    del pets['Breed0']
    pets = p.concat((pets, p.DataFrame(dummiefy(pets.Breed1))), axis=1)
    del pets['Breed1']
    pets = p.concat((pets, p.DataFrame(dummiefy(pets.Sex))), axis=1)
    del pets['Sex']
    pets = p.concat((pets, p.DataFrame(dummiefy(pets.Color0))), axis=1)
    pets = p.concat((pets, p.DataFrame(dummiefy(pets.Color1))), axis=1)
    del pets['Color0']
    del pets['Color1']
    # pets['AgeCategory'] = pets.AgeInYears.apply(calc_age_category)
    pets[pets.AnimalType == 'Cat'].AgeInMonths.fillna(pets.groupby('AnimalType')['AgeInMonths'].mean().astype('int64')['Cat'], inplace=True)
    pets[pets.AnimalType == 'Dog'].AgeInMonths.fillna(pets.groupby('AnimalType')['AgeInMonths'].mean().astype('int64')['Dog'], inplace=True)


    del pets['AnimalType']
    features += list(pets.columns)
    X = pets.values

    return X

# Turn this to Object Oriented
# Implement modeling + Ensembling - Blend




def logloss(attempt, actual, epsilon=1.0e-15):
    """Logloss, i.e. the score of the bioresponse competition.
    """
    attempt = np.clip(attempt, epsilon, 1.0 - epsilon)
    return - np.mean(actual * np.log(attempt) + (1.0 - actual) * np.log(1.0 - attempt))


if __name__ == '__main__':

    np.random.seed(0)  # seed to shuffle the train set

    # 10 Fold Cross validation
    n_folds = 10
    verbose = True
    shuffle = False

    X, y, X_submission = load_transform()

    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]

    skf = list(StratifiedKFold(y, n_folds))
    # A list of k arrays of indices that will be use in cross validation
    #   [(array([ 359,  361,  363, ..., 3748, 3749, 3750]),
    # array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
    #         13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,...])]

    clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)]

    print("Creating train and test sets for blending.")

    # Blank matrices to hold the probabilities from the best performing model from the 10 validated for each clf
    # (3751, 5)
    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    # (2501, 5)
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))

    for j, clf in enumerate(clfs):
        print(j, clf)
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            print("Fold", i)
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)[:, 1]
            dataset_blend_train[test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:, 1]
        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)

    print
    print("Blending.")
    clf = LogisticRegression()
    clf.fit(dataset_blend_train, y)
    y_submission = clf.predict_proba(dataset_blend_test)[:, 1]

    print("Linear stretch of predictions to [0,1]")
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

    print("Saving Results.")
    np.savetxt(fname='test.csv', X=y_submission, fmt='%0.9f')
