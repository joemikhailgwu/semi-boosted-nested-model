from sklearn.svm import SVC
from sklearn.naive_bayes import  GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier


from sklearn.metrics import roc_auc_score, cohen_kappa_score, precision_score
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, roc_curve

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler, minmax_scale

from sklearn.neighbors import KNeighborsClassifier
from scipy import stats

from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from imblearn.metrics import sensitivity_score, geometric_mean_score, specificity_score

from datetime import date, datetime

import time
import numpy as np
import pandas as pd
import math
import os

class SBN_MODEL:

    #training/test data
    xtrain = np.zeros(1)
    xtest = np.zeros(1)
    ytrain = np.zeros(1)
    ytest = np.zeros(1)

    #number of classifiers in the ensemble
    num_classes = 0
    classifiers_active = []

    #run parameters
    mode_splittrain = False
    count_subensembles = 0
    count_iterations = 0
    count_sublearners = 0


    #ensemble
    clf = BaggingClassifier(n_estimators=1)
    weights_array = np.zeros(1)
    set_prune = 0
    ensemble_type="adadt"
    weight_metric = "tpr"

    #resampling
    resampling = "none"

    #dataname
    data_set = "blank"
    data_name = "data_none"

    #classifier string array
    clf_names = []

    #test data
    y_actual = 0

    def weights_cv(self, xtrain, ytrain):

        X = xtrain.copy()
        Y = ytrain.copy()

        kf = KFold(n_splits=2, shuffle=True)

        auc_array = np.zeros(shape=(2, self.num_classes))
        class_distro = np.ones(shape=(1, self.num_classes))

        i = 0
        print "->weight calculations: ",
        for train_index, test_index in kf.split(X):
            for n in range(0, self.num_classes):
                #print "->class " + str(n+1)
                class_current = (n + 1)
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]

                ind_current_train = np.where(Y_train == class_current)[0]
                ind_other_train = np.where(Y_train != class_current)[0]

                ind_current_test = np.where(Y_test == class_current)[0]
                ind_other_test = np.where(Y_test != class_current)[0]

                y_train_bin = Y_train.copy()
                y_train_bin[ind_current_train] = 1
                y_train_bin[ind_other_train] = -1

                y_test_bin = Y_test.copy()
                y_test_bin[ind_current_test] = 1
                y_test_bin[ind_other_test] = -1

                ws = self.iterate_weights(X_train,  y_train_bin, X_test, y_test_bin)
                #print ws


                auc_array[i, n] = ws
            i = i + 1
            #print auc_array

        weights_final = np.sum(auc_array, axis=0)

        weights_final = weights_final * class_distro
        weights_final = weights_final[0]

        weights_final = np.multiply(weights_final, 1. / np.sum(weights_final))
        print str(weights_final)
        return weights_final



    def load_file(self, **kwargs):
        if kwargs is not None:
            for key, value in kwargs.iteritems():
                if(key == "xtrain_h"):
                    print "    ->load-h [" + str(value) + "]",
                    self.xtrain = np.array(pd.read_csv("data/" + value, delimiter=','))
                    print " ->done"
                elif (key == "xtrain_nh"):
                    print "    ->load-nh [" + str(value) + "]",
                    self.xtrain = np.array(pd.read_csv("data/" + value, delimiter=',', header=None))
                    print " ->done"
                elif (key == "xtest_nh"):
                    print "    ->load-nh [" + str(value) + "]",
                    self.xtest = np.array(pd.read_csv("data/" + value, delimiter=',', header=None))
                    print " ->done"
                elif (key == "xtest_h"):
                    print "    ->load-h [" + str(value) + "]",
                    self.xtest = np.array(pd.read_csv("data/" + value, delimiter=','))
                    print " ->done"
                elif (key == "ytrain_nh"):
                    print "    ->load-nh [" + str(value) + "]",
                    self.ytrain = np.ravel(np.array(pd.read_csv("data/" + value, delimiter=',', header=None)))
                    print " ->done"
                elif (key == "ytrain_h"):
                    print "    ->load-h [" + str(value) + "]",
                    self.ytrain = np.ravel(np.array(pd.read_csv("data/" + value, delimiter=',')))
                    print " ->done"
                elif (key == "ytest_nh"):
                    print "    ->load-nh [" + str(value) + "]",
                    self.ytest = np.ravel(np.array(pd.read_csv("data/" + value, delimiter=',', header=None)))
                    print " ->done"
                elif (key == "ytest_h"):
                    print "    ->load-h [" + str(value) + "]",
                    self.ytest = np.ravel(np.array(pd.read_csv("data/" + value, delimiter=',')))
                    print " ->done"

    def load_info(self, **kwargs):
        print "->file info ",
        if kwargs is not None:
            for key, value in kwargs.iteritems():
                if(key=="info"):
                    if(value=="xtrain"):
                        print "[xtrain]"
                        print "  ->samples [" + str(len(self.xtrain)) +"]"
                        print "  ->features [" + str(len(self.xtrain[0])) + "]"
                    if (value == "ytrain"):
                        print "[ytrain]"
                        print "  ->samples [" + str(len(self.ytrain)) + "]"
                        print "  ->labels [" + str(np.unique(self.ytrain)) + "]"



    def __init__(self, **kwargs):


        print "->settings"
        if kwargs is not None:
            for key, value in kwargs.iteritems():
                if(key == "xtrain"):
                    print "  ->training file: [" + str(value) +"]"
                    self.load_file(xtrain_h=value)
                elif(key == "ytrain"):
                    print "  ->training labels: [" + str(value)+"]"
                    self.load_file(ytrain_h=value)
                elif(key == "xtest"):
                    print "  ->test file: [" + str(value)+"]"
                    self.load_file(xtest_h=value)
                elif(key == "ytest"):
                    print "  ->test labels: [" + str(value)+"]"
                    self.load_file(ytest_h=value)
                elif(key == "prune"):
                    print "  ->pruning: [" + str(value)+"]"
                    if(value == True):
                        self.set_prune = 1
                    else:
                        self.set_prune = 0
                elif (key == "sampling"):
                    print "  ->sampling: [" + str(value)+"]"
                elif (key == "sublearners"):
                    print "  ->sublearners: [" + str(value)+"]"
                    self.count_sublearners = value
                elif (key == "ensemble_type"):
                    print "  ->ensemble type: [" + str(value) + "]"
                    self.ensemble_type = value
                elif (key == "data_name"):
                    print "  ->data_name: [" + str(value) + "]"
                    self.data_name = value
                elif (key == "weight"):
                    print "  ->weighting: [" + str(value) + "]"
                    self.weight_metric = value
                elif (key == "split_train"):
                    print "  ->split training data: [" + str(value) + "]"
                    self.mode_splittrain = value
                elif (key == "subensembles"):
                    print "  ->subensembles: [" + str(value) + "]"
                    self.count_subensembles = value
                elif (key == "iterations"):
                    print "  ->iterations: [" + str(value) + "]"
                    self.count_iterations = value

        self.initialize()
        self.load_info(info='xtrain')
        self.load_info(info='ytrain')
        print ""

        x=raw_input("|press enter to continue|")
        self.run()


    def initialize(self):
        #initialize classifiers
        self.clf = BaggingClassifier(n_estimators=self.count_sublearners)
        print "->initialize sublearners=[" + str(self.count_sublearners) + "]",
        self.clf.fit(self.xtrain[0:10, :], self.ytrain[0:10])
        print "->done"


    def run(self):
        self.num_classes = len(np.unique(self.ytrain))

        #make copies of original data
        train_copy_orig_x = self.xtrain.copy()
        test_copy_orig_x = self.xtest.copy()
        train_copy_orig_y = self.ytrain.copy()
        test_copy_orig_y = self.ytest.copy()

        tpr_array = np.zeros(self.count_iterations)
        fpr_array = np.zeros(self.count_iterations)

        time_start = time.time()
        for i in range(0, self.count_iterations):
            os.system('cls')
            print "-----------------------------------------------------------------"
            print "--------------------SEMI-BOOSTED NESTED MODEL--------------------"
            print "------JOSEPH MIKHAIL, GEORGE WASHINGTON UNIVERSITY, 2019---------"
            print "-----------------------------------------------------------------"
            print "->iteration: [" + str(i+1) + "]"

            if(self.mode_splittrain):
                xtrain_working, xtest_working, ytrain_working, ytest_working = train_test_split(train_copy_orig_x, train_copy_orig_y, test_size=0.5)
            else:
                xtrain_working = train_copy_orig_x.copy()
                xtest_working = test_copy_orig_x.copy()
                ytrain_working = train_copy_orig_y.copy()
                ytest_working = test_copy_orig_y.copy()

            if(self.weight_metric != 'uw'):
                w = self.weights_cv(xtrain_working, ytrain_working)
            else:
                w = np.ones(shape=(self.num_classes))

            ova_pred_array_combined = np.zeros(shape=(len(xtest_working), self.num_classes))
            for l in range(0, self.count_subensembles):
                print "->subensemble #[" + str(l+1) + "]"
                ova_pred_array_binary = np.zeros(shape=(len(xtest_working), self.num_classes))
                ova_pred_array_weighted = np.ones(shape=(len(xtest_working), self.num_classes))
                for n in range(0, self.num_classes):
                    print "  ->class " + str(n+1),
                    class_current = (n + 1)

                    ind_current_train = np.where(ytrain_working == class_current)[0]
                    ind_other_train = np.where(ytrain_working != class_current)[0]

                    ind_current_test = np.where(ytest_working == class_current)[0]
                    ind_other_test = np.where(ytest_working != class_current)[0]

                    y_train_bin = ytrain_working.copy()
                    y_train_bin[ind_current_train] = 1
                    y_train_bin[ind_other_train] = -1

                    y_test_bin = ytest_working.copy()
                    y_test_bin[ind_current_test] = 1
                    y_test_bin[ind_other_test] = -1

                    metric_arr, ws = self.iterate(xtrain_working, xtest_working, y_train_bin, y_test_bin)
                    ova_pred_array_binary[:, n] = self.y_prediction[:, 0]
                    ova_pred_array_weighted[:, n] = (ova_pred_array_binary[:, n] * w[n])

                ova_pred_array_combined = ova_pred_array_combined + ova_pred_array_weighted

                y_pred = np.add(np.argmax(ova_pred_array_combined, axis=1),1)
                tpr_arr = np.round(sensitivity_score(ytest_working, y_pred,average=None),4)
                fpr_arr = np.round(np.subtract(1,specificity_score(ytest_working, y_pred, average=None)),4)
                print "  ->results - subensembles = [" + str(l+1) + "]"
                print "    ->tpr: " + str(tpr_arr),
                print "    ->avg: ["  + str(np.round(np.average(tpr_arr),4)) +"]"
                print "    ->fpr: "+ str(fpr_arr),
                print "    ->avg: [" + str(np.round(np.average(fpr_arr),4)) + "]"
            tpr_array[i] = np.round(np.average(tpr_arr),4)
            fpr_array[i] = np.round(np.average(fpr_arr),4)
        time_finish = time.time()
        os.system('cls')
        print "-----------------------------------------------------------------"
        print "--------------------SEMI-BOOSTED NESTED MODEL--------------------"
        print "------JOSEPH MIKHAIL, GEORGE WASHINGTON UNIVERSITY, 2019---------"
        print "-----------------------------------------------------------------"
        print "->completed [" +str(i+1) + "] iterations in [" + str(time_finish-time_start) +"] seconds"
        print "->tpr: " + str(tpr_array)
        print "->average tpr: [" + str(np.round(np.average(tpr_array),4)) + "]"
        print "->fpr: " + str(tpr_array)
        print "->average fpr: [" + str(np.round(np.average(fpr_array), 4)) + "]"




    def resample(self, xtrain, ytrain):
        RS = RandomUnderSampler()
        xtrain_rs, ytrain_rs = RS.fit_sample(xtrain, ytrain)
        return xtrain_rs, ytrain_rs

    def create_sublearners(self):
        print "  ->create sublearners: [",
        for each_classifier in range(0, self.count_sublearners):

            if(self.ensemble_type=='adadt'):
                if (each_classifier<=4):
                    self.clf.estimators_[each_classifier] = DecisionTreeClassifier()
                    self.clf_names.append("dt")
                    print " dt ",
                else:
                    print " ada ",
                    self.clf_names.append("ada")
                    A = AdaBoostClassifier(n_estimators=10)
                    self.clf.estimators_[each_classifier] = A

            if (self.ensemble_type == 'ada'):
                print " ada ",
                self.clf_names.append("ada")
                A = AdaBoostClassifier(n_estimators=10)
                self.clf.estimators_[each_classifier] = A

            if (self.ensemble_type == 'dt'):
                self.clf.estimators_[each_classifier] = DecisionTreeClassifier()
                self.clf_names.append("dt")
                print " dt ",
        print "]"


    def fit(self, xtrain, ytrain, prune_size):

        xtrain_rs, xprune_rs, ytrain_rs, yprune_rs = train_test_split(xtrain, ytrain, test_size=prune_size, shuffle=True)

        print "  ->training samples [" + str(len(xtrain_rs)) + " (" + str(round(float(100.0*len(xtrain_rs)/(len(xtrain_rs) + len(xprune_rs))),2)) + "%)]"
        print "  ->pruning samples [" + str(len(xprune_rs)) + " (" + str(round(float(100.0*len(xprune_rs)/(len(xtrain_rs) + len(xprune_rs))),2)) + "%)]"

        pred_matrix_prune = np.zeros(shape=(len(xprune_rs), self.count_sublearners))

        compare_matrix = np.zeros(shape=(len(xprune_rs), self.count_sublearners))

        NT_MATRIX = np.zeros(len(yprune_rs))
        NF_MATRIX = np.zeros(len(yprune_rs))

        print "  ->fit sublearners: [",
        for each_classifier in range(0, self.count_sublearners):

            r = np.random.randint(2, size=1)
            if(r==1):
                xtrain_rs, ytrain_rs = self.resample(xtrain_rs, ytrain_rs)
                print str(self.clf_names[each_classifier]) +"[r]",
            else:
                print str(self.clf_names[each_classifier]),

            self.clf.estimators_[each_classifier].fit(xtrain_rs, ytrain_rs)


            pred_matrix_prune[:, each_classifier] = self.clf.estimators_[each_classifier].predict(xprune_rs)

            compare_matrix[:, each_classifier] = np.equal(yprune_rs, pred_matrix_prune[:, each_classifier])

        print "]"

        yprune_pred = stats.mode(pred_matrix_prune, axis=1)

        #print "->calculate diversity"

        active_indices = np.zeros(self.count_sublearners)
        for x in range(0, self.count_sublearners):
            active_indices[x] = int(x)

        #print "  ->compute NT/NF matrices",
        NT_MATRIX = np.sum(compare_matrix[:, 0:len(active_indices)], axis=1)
        NT_MATRIX = np.multiply(NT_MATRIX, 1.0 / len(active_indices))
        NF_MATRIX = np.subtract(np.ones(len(yprune_rs)), NT_MATRIX)

        HYP_SCORE_MATRIX = np.zeros(shape=(len(yprune_rs), len(active_indices)))

        for c in active_indices:

            # EVENT E_TF
            EVENT_TF_IND = np.where((compare_matrix[:, int(c)] == 1) & (NF_MATRIX >= 0.5))
            # EVENT E_FT
            EVENT_FT_IND = np.where((compare_matrix[:, int(c)] == 0) & (NT_MATRIX >= 0.5))
            # EVENT E_TT
            EVENT_TT_IND = np.where((compare_matrix[:, int(c)] == 1) & (NT_MATRIX >= 0.5))
            # EVENT E_FF
            EVENT_FF_IND = np.where((compare_matrix[:, int(c)] == 0) & (NF_MATRIX >= 0.5))

            if (self.set_prune == 1):
                HYP_SCORE_MATRIX[EVENT_TF_IND, int(c)] = 1

        hyp_ens_sum = np.sum(HYP_SCORE_MATRIX, axis=0)
        hyp_ens_ranked = np.argsort(hyp_ens_sum)
        classifiers_active = np.ones(len(pred_matrix_prune[0]))
        classifiers_active_next = np.ones(len(pred_matrix_prune[0]))

        acc_score_max = 0
        for c in range(0, len(pred_matrix_prune[0])):
            #print "    " + str(classifiers_active)
            if (self.set_prune == 1 and np.sum(classifiers_active) > 1):
                classifiers_active_next = classifiers_active.copy()
                classifiers_active_next[hyp_ens_ranked[c]] = 0
                ind_prune = np.where(classifiers_active_next == 1)[0]
                yprune_pred_next = stats.mode(pred_matrix_prune[:, ind_prune], axis=1)
                acc_score_temp = f1_score(yprune_rs, yprune_pred_next[0], average='macro')

                # DETERMINE WHETHER TO SAVE THE BEST CLASSIFIER
                print "  ->prune: [" + str(classifiers_active)+"]",
                print "\r",
                if (acc_score_temp >= acc_score_max):
                    #print "yes",
                    #print "\r",
                    #print "  ->",
                    #print("%.4f" % acc_score_max),
                    #print "-> ",
                    acc_score_max = acc_score_temp
                    #print("%.4f" % acc_score_temp),

                    classifiers_active = classifiers_active_next.copy()
                #else:
                    #print "no",
        print ""
        self.classifiers_active = classifiers_active
        return f1_score(yprune_rs, yprune_pred[0], average='macro')

    def predict(self, x_test):
        ind = np.where(self.classifiers_active==1)[0]
        self.y_prediction_all = np.zeros(shape=(len(x_test), self.count_sublearners))
        self.y_prediction = np.ravel(np.zeros(shape=(len(x_test), 1)))

        for x in range(0, self.count_sublearners):
            proba = self.clf.estimators_[x].predict_proba(x_test)[:, 1]
            self.y_prediction_all[:, x] = proba[:]

        ni = np.where(self.y_prediction_all == 0)
        self.y_prediction_all[ni] = -1

        nn = np.where((self.y_prediction_all > 0) & (self.y_prediction_all < 0.5))
        self.y_prediction_all[nn] = -1 * (1 - self.y_prediction_all[nn])


        for x in range(0, self.count_sublearners):
            self.y_prediction_all[:, x] = self.clf.estimators_[x].predict(x_test)

        self.y_prediction = stats.mode(self.y_prediction_all[:, ind], axis=1)[0]
        return self.y_prediction

    def iterate(self, xtrain, xtest, ytrain, ytest):
        metric_array = np.zeros(shape=(1, 5))

        self.create_sublearners()
        weight_score = self.fit(xtrain, ytrain, 0.2)
        y_pred = self.predict(xtest)

        a,b,c = self.print_metrics_binary(ytest, y_pred)
        metric_array[0, 0] = a
        metric_array[0, 1] = b
        metric_array[0, 2] = c


        #df = pd.DataFrame(metric_array, columns=['auc', 'acc', 'tpr', 'fpr', 'f1'])
        #df.to_csv(str(self.data_name) + "_" + str(date.today()) + "_" + str(div_measure) + '_' + str(the_count) + '_iterate.csv', header=True, sep=',')
        return metric_array, weight_score

    def iterate_weights(self, xtrain, ytrain, xtest, ytest):
        DT = DecisionTreeClassifier()
        DT.fit(xtrain, ytrain)
        y_pred = DT.predict(xtest)

        if(self.weight_metric == "tpr"):
            return sensitivity_score(ytest, y_pred)
        if (self.weight_metric == "acc"):
            return accuracy_score(ytest, y_pred)


    def print_metrics_binary(self, y_actual, y_pred):

        metric_f1 = f1_score(y_actual, y_pred, average='macro')
        metric_sensitivity = sensitivity_score(y_actual, y_pred, average='macro')
        metric_specificity = specificity_score(y_actual, y_pred, average='macro')
        metric_accuracy = accuracy_score(y_actual, y_pred)
        print "  ->metrics",

        print "  ->acc: ",
        print("[%.4f]" % metric_accuracy),

        print "  ->tpr: ",
        print("[%.4f]" % metric_sensitivity),

        print "  ->fpr: ",
        print("[%.4f]" % (1.0-metric_specificity))

        return metric_accuracy, metric_sensitivity, (1.0000 - metric_specificity)
