/*---------------------------------------------------------------------
  SAS Program: generate_figures.sas
  Description: Generates figures using CSV files exported from the 
               Python pipeline. Uses relative paths (files are assumed 
               to be in the same folder as this script).
---------------------------------------------------------------------*/

/* Set the ODS destination to HTML (or PDF) using relative paths */
ods html path="." file="Figures.html" style=HTMLBlue;
ods graphics on / reset=all imagename="Figure";

/*---------------------------------------------------------
  1. Feature Distribution Plots
     (CSV with columns: feature, value, plot_type)
---------------------------------------------------------*/
proc import datafile="feature_distribution.csv"
    out=feat_dist
    dbms=csv
    replace;
    guessingrows=MAX;
run;

proc sgpanel data=feat_dist;
   panelby feature / layout=panel columns=2;
   styleattrs datacolors=(skyblue lightgreen);
   histogram value / scale=count;
   density value / type=kernel lineattrs=(color=red);
   colaxis label="Value";
   rowaxis label="Count";
   title "Histogram and Density for #byval(feature)";
run;

/*---------------------------------------------------------
  2. Class Distribution Bar Charts
     (CSV with columns: attack_cat, Count, dataset)
---------------------------------------------------------*/
proc import datafile="class_distribution.csv"
    out=class_dist
    dbms=csv
    replace;
    guessingrows=MAX;
run;

proc sgplot data=class_dist(where=(dataset="original"));
   vbar attack_cat / response=Count datalabel;
   title "Original Class Distribution";
   xaxis label="Attack Category";
   yaxis label="Count";
run;

proc sgplot data=class_dist(where=(dataset="balanced"));
   vbar attack_cat / response=Count datalabel;
   title "Balanced Class Distribution after SMOTE";
   xaxis label="Attack Category";
   yaxis label="Count";
run;

/*---------------------------------------------------------
  3. Hyperparameter Tuning Heatmap
     (CSV with columns: n_estimators, min_samples_split, mean_test_score)
---------------------------------------------------------*/
proc import datafile="hyperparameter_tuning.csv"
    out=tuning
    dbms=csv
    replace;
    guessingrows=MAX;
run;

proc sgplot data=tuning;
   heatmapparm x=n_estimators 
               y=min_samples_split 
               colorresponse=mean_test_score / 
               colormodel=(blue green yellow red) discretex discretey;
   title "Hyperparameter Tuning (Weighted F1 Score)";
   xaxis label="n_estimators";
   yaxis label="min_samples_split";
run;

/*---------------------------------------------------------
  4. Classifier Performance Comparison
     (CSV with columns: Model, Metric, Score)
---------------------------------------------------------*/
proc import datafile="classifier_performance.csv"
    out=perf
    dbms=csv
    replace;
    guessingrows=MAX;
run;

proc sgplot data=perf;
   vbar Model / response=Score group=Metric groupdisplay=cluster datalabel;
   title "Classifier Performance Comparison";
   xaxis label="Model";
   yaxis label="Score" min=0 max=1;
run;

/*---------------------------------------------------------
  5. Confusion Matrix Heatmap
     (CSV with columns: TrueLabel, PredLabel, Count)
---------------------------------------------------------*/
proc import datafile="confusion_matrix.csv"
    out=conf_mat
    dbms=csv
    replace;
    guessingrows=MAX;
run;

proc sgplot data=conf_mat;
   heatmapparm x=PredLabel 
               y=TrueLabel 
               colorresponse=Count / 
               colormodel=(lightblue blue darkblue) discretex discretey;
   title "Confusion Matrix Heatmap";
   xaxis label="Predicted Label";
   yaxis label="True Label";
run;

/*---------------------------------------------------------
  6. Real-Time Monitoring & Prevention Flow Diagram
---------------------------------------------------------*/
data flow;
   length Stage $50;
   input Stage $50.;
   StageNum = _N_;
   datalines;
"Live Packet Capture"
"Feature Extraction"
"Data Scaling"
"Model Prediction"
"Threshold Check"
"Automated IP Blocking"
;
run;

proc sgplot data=flow noautolegend;
   scatter x=StageNum y=1 / markerattrs=(symbol=circlefilled size=12 color=lightblue);
   series x=StageNum y=1 / lineattrs=(color=gray thickness=2);
   xaxis type=discrete label="Flow Stage" discreteorder=data;
   yaxis display=none;
   inset Stage / position=top;
   title "Real-Time Monitoring & Prevention Flow";
run;

ods html close;
ods graphics off;
