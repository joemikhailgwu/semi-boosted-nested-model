# semi-boosted-nested-model
Code for "A Semi-Boosted Nested Model with Sensitivity-based Weighted Binarization for Multi-Domain Network Intrusion Detection", ACM Transactions on Intelligent Systems, 2019. Joseph Mikhail, John Fossaceca, Ron Iammartino - George Washington University

Note: Still requires cleanup/documentation

Macro-Average Performance for Tested Datasets using b=5:

    ->NSL-KDD (50%/50% Training Set Split, Pruning=True): 92.31% TPR, 0.07% FPR

    ->AWID 802.11 (Training/Test sets, Pruning=False): 86.34% TPR, 3.48% FPR

------------------------------------------------------------------------------

Other dataset results (not in paper) -- Macro-Average Performance for Tested Datasets using b=20:

    ->UNSW-NB15 (50%/50% Training Set Split, b=20, Pruning=True): 56.35% TPR, 2.21% FPR

        ->Normal Traffic Detection: 90.0% TPR, 1.80% FPR

    ->New Gas Pipeline (Ian Turnipseed/Tommy Morris) (50%/50% Training Set Split, b=5, Pruning = True): 90.29% TPR, 1.52% FPR
    
  
