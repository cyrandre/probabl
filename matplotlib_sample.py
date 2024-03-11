from sklearn.metrics import (
    ConfusionMatrixDisplay, confusion_matrix,
    RocCurveDisplay,roc_curve,
    PrecisionRecallDisplay,precision_recall_curve,
)

from itertools import product
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import numpy as np

# Interactive Display of ROC curve, PR curve and confusion matrix 
# Matplotlib version
class RocPrCmDisplay():

    def __init__(self,estimator,x_test,y_test,pos_label=None):

        # Keep score and limits
        self.y_score = estimator.decision_function(x_test)
        self.min_score = np.min(self.y_score)
        self.max_score = np.max(self.y_score)
        
        # Set default pos label
        if pos_label is None:
            pos_label = 1
        
        # Create ROC curve
        self.fpr,self.tpr,self.roc_th = roc_curve(
            y_test,self.y_score,pos_label=pos_label
        )
        
        # Create PR curve
        self.prec,self.recall,self.pr_th = precision_recall_curve(
            y_test,self.y_score,pos_label=pos_label
        )

        # Convert Y test to boolean array
        self.y_test = y_test == pos_label

    def plot(self,threshold = 0.5):
        # Adjust thresh to the score limits
        th = self._thresh(threshold)

        # Confusion matrix
        cm = self._confusion_matrix(th)
        #Confusion matrix display
        cm_display = ConfusionMatrixDisplay(cm)

        # ROC curve display
        roc_display = RocCurveDisplay(fpr=self.fpr,tpr=self.tpr)

        # PR curve display
        pr_display = PrecisionRecallDisplay(
            precision=self.prec,recall=self.recall
        )

        # Figure
        self.fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(12,4))
        self.cm_plot = cm_display.plot(ax = ax1,colorbar=True)

        roc_pt = self._roc(th)        
        roc_display.plot(ax=ax2)
        self.roc_pt, = ax2.plot(roc_pt[0],roc_pt[1],'o')

        pr_pt = self._pr(th)
        pr_display.plot(ax=ax3)
        self.pr_pt, = ax3.plot(pr_pt[0],pr_pt[1],'o')

        # Matplotlib slider 
        self.fig.subplots_adjust(bottom=0.25)
        axfreq = self.fig.add_axes([0.2, 0.1, 0.65, 0.03])
        self.slider = Slider(
            ax=axfreq,
            label='Threshold',
            valmin=0,
            valmax=1,
            valinit=0.5,
        )
        self.slider.on_changed(self.update)
        return self
    
    def _thresh(self,th):
        # Adjust threshold according to y_score limits
        return self.min_score * (1-th) + self.max_score * th

    def _confusion_matrix(self,th):
        # Compute the confusion matrix
        if th <= self.min_score:
            y_pred = np.ones(self.y_test.size,dtype=bool)
        else:
            y_pred = self.y_score > th
        return confusion_matrix(self.y_test,y_pred)
        
    def _roc(self,th):
        # Search the closest point to the threshold on the ROC curve
        roc_index = np.abs(self.roc_th-th).argmin()
        roc_pt = self.fpr[roc_index], self.tpr[roc_index],
        return roc_pt
    
    def _pr(self,th):
        # Search the closest point to the threshold on the PR curve
        pr_index = np.abs(self.pr_th-th).argmin()
        pr_pt = self.recall[pr_index], self.prec[pr_index], 
        return pr_pt
    
    def update(self,threshold):
        # Update plots
        th = self._thresh(threshold)
        # Update threshold position in ROC curve
        self.roc_pt.set_data(self._roc(th))     
        # Update threshold position in PR curve   
        self.pr_pt.set_data(self._pr(th))
        # Update confusion matrix
        cm = self._confusion_matrix(th)
        self.cm_plot.im_.set_data(cm)
        # Update text and text color of text in the confusion matrix
        thresh = (cm.max() + cm.min()) / 2.0
        cmap_min, cmap_max = self.cm_plot.im_.cmap(0), self.cm_plot.im_.cmap(1.0)
        for i, j in product(range(2), range(2)):
            color = cmap_max if cm[i, j] < thresh else cmap_min

            text_cm = format(cm[i, j], ".2g")
            if cm.dtype.kind != "f":
                text_d = format(cm[i, j], "d")
                if len(text_d) < len(text_cm):
                    text_cm = text_d
            self.cm_plot.text_[i, j].set_text(text_cm)
            self.cm_plot.text_[i, j].set_color(color)

if __name__ == '__main__':
    # Test display with a matplotlib slider  
    from sklearn.datasets import fetch_openml
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    
    # Load a dataset
    X,y = fetch_openml(data_id=1464, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0,stratify=y)
    
    # Train the model
    clf = make_pipeline(StandardScaler(), LogisticRegression(random_state=0))
    clf.fit(X_train, y_train)
    
    # Create display
    disp = RocPrCmDisplay(estimator=clf,
                x_test=X_test,
                y_test=y_test,
                pos_label= clf.classes_[1])
    
    disp.plot()
    plt.show()
