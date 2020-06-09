{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'null' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-12-a1ea2038d7cb>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0;32mimport\u001b[0m \u001b[0mxgboost\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mxgb\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      2\u001b[0m \u001b[0;31m#from xgboost.sklearn import XGBClassifier\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0msklearn\u001b[0m \u001b[0;32mimport\u001b[0m  \u001b[0mmetrics\u001b[0m   \u001b[0;31m#Additional scklearn functions\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0;32mclass\u001b[0m \u001b[0mXGBoost\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/Insight-Project/model/xgboost.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m    100\u001b[0m   {\n\u001b[1;32m    101\u001b[0m    \u001b[0;34m\"cell_type\"\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0;34m\"code\"\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 102\u001b[0;31m    \u001b[0;34m\"execution_count\"\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mnull\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    103\u001b[0m    \u001b[0;34m\"metadata\"\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0;34m{\u001b[0m\u001b[0;34m}\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    104\u001b[0m    \u001b[0;34m\"outputs\"\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mNameError\u001b[0m: name 'null' is not defined"
     ]
    }
   ],
   "source": [
    "import xgboost as xgb\n",
    "#from xgboost.sklearn import XGBClassifier\n",
    "from sklearn import  metrics   #Additional scklearn functions\n",
    "\n",
    "class XGBoost:\n",
    "    def __init__(self):\n",
    "        #self.xgb = xgb.XGBClassifier(n_estimators=500, max_depth=5, base_score=0.5,objective='binary:logistic', random_state=42)\n",
    "        self.xgb = XGBClassifier(learning_rate =0.1, n_estimators=1000, max_depth=5, min_child_weight=1, gamma=0, subsample=0.8, \n",
    "                                 colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27)\n",
    "        self.title = 'XGBoost'\n",
    "\n",
    "    def fit(self, X_train, y_train):\n",
    "        self.xgb.fit(X_train, y_train)\n",
    "    \n",
    "    def modelfit(alg, x_train, y_train,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):\n",
    "        start_time = time.time()\n",
    "    \n",
    "        if useTrainCV:\n",
    "            xgb_param = alg.get_xgb_params()\n",
    "            xgtrain = xgb.DMatrix(x_train, label=y_train)\n",
    "            cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,\n",
    "                metrics='auc', early_stopping_rounds=early_stopping_rounds)\n",
    "            alg.set_params(n_estimators=cvresult.shape[0])\n",
    "            print(\"cvresult.shape[0] :\",cvresult.shape[0] )\n",
    "            #clf.best_ntree_limit\n",
    "    \n",
    "        #Fit the algorithm on the data\n",
    "        alg.fit(x_train, y_train,eval_metric='auc')\n",
    "        #print(\"best ntree limit: \", alg.best_ntree_limit)\n",
    "        print ('Model trained in seconds ',format(time.time() - start_time))   \n",
    "        #Predict training set:\n",
    "        dtrain_predictions = alg.predict(x_train)\n",
    "        dtrain_predprob = alg.predict_proba(x_train)[:,1]\n",
    "        \n",
    "        #Print model report:\n",
    "        print (\"Accuracy : \" , metrics.accuracy_score(y_train, dtrain_predictions))\n",
    "        print (\"AUC Score (Train):\", metrics.roc_auc_score(y_train, dtrain_predprob))\n",
    "    \n",
    "\n",
    "        feat_imp = pd.Series(alg.get_booster().get_score(importance_type='weight')).sort_values(ascending=False)\n",
    "        feat_imp.plot(kind='bar', title='Feature Importances')\n",
    "        plt.ylabel('Feature Importance Score')\n",
    "        return alg\n",
    "\n",
    "    def predict(self, X_test):\n",
    "        y_pred = self.xgb.predict(X_test)\n",
    "        return y_pred\n",
    "\n",
    "    def get_model(self):\n",
    "        return self.xgb\n",
    "\n",
    "    def get_title(self):\n",
    "        return self.title"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: xgboost in /opt/conda/lib/python3.7/site-packages (1.1.0)\n",
      "Requirement already satisfied: scipy in /opt/conda/lib/python3.7/site-packages (from xgboost) (1.4.1)\n",
      "Requirement already satisfied: numpy in /opt/conda/lib/python3.7/site-packages (from xgboost) (1.18.4)\n"
     ]
    }
   ],
   "source": [
    "!pip install xgboost"
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
      "Python 3.7.6\n"
     ]
    }
   ],
   "source": [
    "!python3 --version"
   ]
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
  "environment": {
   "name": "tf2-2-2-gpu.2-2.m48",
   "type": "gcloud",
   "uri": "gcr.io/deeplearning-platform-release/tf2-2-2-gpu.2-2:m48"
  },
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
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
