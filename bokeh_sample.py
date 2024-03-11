from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
import numpy as np
from bokeh.layouts import row,column
from bokeh.plotting import figure, show
from bokeh.palettes import Viridis256
from bokeh.models import CustomJS,Slider,ColumnDataSource,LinearColorMapper

# Interactive Display of ROC curve, PR curve and confusion matrix 
# Bokey version
class RocPrCmDisplay():

  def __init__(self,
               estimator,
               x_test,
               y_test,
               pos_label=None,
               threshold_step=0.01):

    # Compute the decision function
    y_score = estimator.decision_function(x_test)
    min_score = np.min(y_score)
    max_score = np.max(y_score)
    
    # Set default pos label
    if pos_label is None:
        pos_label = 1

    # Create ROC curve
    self.fpr,self.tpr,roc_th = roc_curve(
        y_test,y_score,pos_label=pos_label
    )
    
    # Create PR curve
    self.prec,self.recall,pr_th = precision_recall_curve(
        y_test,y_score,pos_label=pos_label
    )

    # Convert Y test too boolean array
    y_test = y_test == pos_label

    # Save the confusion matrix and threshold 
    # position on ROC and PR curves
    # for all thresholds
    size = int(np.ceil(1./threshold_step))+1
    self.cm = np.empty((size,4))
    self.pr_ind = np.empty(size,dtype=int)
    self.roc_ind = np.empty(size,dtype=int)

    for i in range(size):
      t = min(i * threshold_step,1)
      th = (1-t) * min_score + t * max_score
      if th <= min_score:
            y_pred = np.ones(y_test.size,dtype=bool)
      else:
            y_pred = y_score > th

      self.cm[i] = confusion_matrix(y_test,y_pred).flatten()
      self.roc_ind[i] = np.abs(roc_th-th).argmin()
      self.pr_ind[i] = np.abs(pr_th-th).argmin()

  def plot(self,threshold = 0.5,labels = ['0','1']):
    
    # Index 
    end = self.roc_ind.size - 1
    ithresh = min(int(end * threshold),end)

    roc_ind = self.roc_ind[ithresh]
    pr_ind = self.pr_ind[ithresh]

    xgrid,ygrid = np.meshgrid(
        range(2),range(2))
    
    xgrid = xgrid.flatten() + 0.5
    ygrid = np.flip(ygrid.flatten() + 0.5)

    # Non static data source
    cm_source = ColumnDataSource(data=dict(
       xcm=xgrid,
       ycm=ygrid,
       cm=self.cm[ithresh]))

    rocpr_source = ColumnDataSource(data=dict(
       xroc=(self.fpr[roc_ind],),
       yroc=(self.tpr[roc_ind],),
       xpr=(self.recall[pr_ind],),
       ypr=(self.prec[pr_ind],)))
    
    
    # Figure for the confusion matrix
    s1 = figure(
       width=350, 
       height=350, 
       title="Confusion Matrix",
       x_axis_label="Predicted Label",
       y_axis_label="True Label",
       toolbar_location=None,
       x_range = labels,
       y_range=list(reversed(labels)),       
       background_fill_color="#fafafa")
    
    colors = list(Viridis256)
    mapper = LinearColorMapper(palette=colors, low=0.0, high=end)
    text_mapper = LinearColorMapper(
        palette=[colors[-1],colors[0]],
        low=0.,high=end)
    
    s1.rect(
      source=cm_source,
      x='xcm',
      y='ycm',
      width=1, 
      height=1, 
      line_width = 0,
      fill_color={'field':'cm','transform': mapper},
    )

    s1.text(
        source=cm_source,
        x='xcm',
        y='ycm',
        text='cm',
        text_font_size='10pt',
        text_color = {'field':'cm','transform': text_mapper},
        x_offset=-30,
        y_offset=10
        )
    
    # Display the ROC Curve
    s2 = figure(
       width=350, 
       height=350, 
       title="ROC Curve",
       x_axis_label="False Positive Rate",
       y_axis_label="True Positive Rate",
       toolbar_location=None,
       background_fill_color="#fafafa")
    s2.line(self.fpr, self.tpr,line_width=2,color="#53777a")
    s2.circle('xroc','yroc',size=12,source=rocpr_source)

    # Display the PR Curve
    s3 = figure(
       width=350, 
       height=350, 
       title="PR Curve",
       x_axis_label="Recall",
       y_axis_label="Precision",
       toolbar_location=None,
       background_fill_color="#fafafa")
    s3.line(self.recall, self.prec,line_width=2, color="#53777a")
    s3.circle('xpr','ypr',size=12,source=rocpr_source)
    
    # Create the slider
    slider = Slider(start=0,end=end,value=ithresh)

    callback = CustomJS(args = dict(
      cm_source= cm_source,
      rocpr_source = rocpr_source,
      roc_ind = self.roc_ind,
      pr_ind = self.pr_ind,
      fpr = self.fpr,
      tpr = self.tpr,
      recall = self.recall,
      prec = self.prec,
      cm_val = self.cm,
      slider=slider),
      code="""

    const t = slider.value
    const i0 = roc_ind[t]
    const i1 = pr_ind[t]

    const xroc = Array(1).fill(fpr[i0])
    const yroc = Array(1).fill(tpr[i0])
    const xpr = Array(1).fill(recall[i1])
    const ypr = Array(1).fill(prec[i1])
    rocpr_source.data={xroc,yroc,xpr,ypr}    

    const xcm = cm_source.data['xcm']
    const ycm = cm_source.data['ycm']
    const cm = cm_val.slice(t*4,t*4+4)
    cm_source.data={xcm,ycm,cm}
""")
    slider.js_on_change('value',callback)
    # put the results in a row and show
    show(column(row(s1, s2, s3),slider))

if __name__ == '__main__':
    # Test display with a matplotlib slider  
    from sklearn.datasets import fetch_openml
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    
    # Load a dataset
    X,y = fetch_openml(data_id=1464, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,random_state=0)
    
    # Train the model
    clf = make_pipeline(StandardScaler(), LogisticRegression(random_state=0))
    clf.fit(X_train, y_train)
    
    # Create display
    disp = RocPrCmDisplay(estimator=clf,
                x_test=X_test,
                y_test=y_test,
                pos_label= clf.classes_[1])
    
    disp.plot()
