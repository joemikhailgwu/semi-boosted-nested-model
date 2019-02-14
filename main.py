#SEMI-BOOSTED NESTED MODEL (SBN MODEL)
#Joseph Mikhail - George Washington University
#joemik@gwu.edu

from SBN import SBN_MODEL
import datetime
import os

def main():

    os.system('cls')
    print "-----------------------------------------------------------------"
    print "--------------------SEMI-BOOSTED NESTED MODEL--------------------"
    print "------JOSEPH MIKHAIL, GEORGE WASHINGTON UNIVERSITY, 2019---------"
    print "-----------------------------------------------------------------"
    print "->started at: [" + str(datetime.datetime.today()) +"]"
    h = SBN_MODEL(xtrain="AWID_X_TRAIN_FS.csv", ytrain="AWID_TRAIN_FULL_Y.csv", xtest="AWID_X_TEST_FS.csv", ytest="AWID_TEST_FULL_Y.csv", prune=False, sampling="RUS"
                  , ensemble_type="adadt", data_name="SBN", weight='tpr', subensembles=5, sublearners=10, iterations=1, split_train=False)
    print "->completed at: [" + str(datetime.datetime.today()) + "]"




if __name__ == "__main__":
    main()

