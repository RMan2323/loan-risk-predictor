# Team 21
## Work Split:
Raghuveer: Preprocessing, Logistic Regression, XGBoost

Vijwal: KNN (+Mahalanobis variant), Linear and Polynomial Regression

Aariz: SVM, K-Medoid, Random Forest

Abhinandan: Neural Network for classification and regression

## CSV Files:
|Name|Description|
|---|---|
|21.csv|Given file|
|Str_To_Int.csv/Int|Converted string classes to integers|
|One-Hot.csv/OH|Converted string classes to different one-hot columns|
|Orig|No scaling|
|Scaled_All|All features standardized|
|Scaled_Cont|Only continuous features standardized|
|MM|Min-Max normalized|

#### String replacements:
| Int | EmploymentStatus |
| --- | --- |
|0|Employed|
|1|Self-Employed|
|2|Unemployed|

| Int | EducationLevel |
| --- | --- |
|0|Master|
|1|Associate|
|2|Bachelor|
|3|High School|
|4|Doctorate|

| Int | MaritalStatus |
| --- | --- |
|0|Married|
|1|Single|
|2|Divorced|
|3|Widowed|

| Int | HomeOwnershipStatus |
| --- | --- |
|0|Own|
|1|Mortgage|
|2|Rent|
|3|Other|

| Int | LoanPurpose |
| --- | --- |
|0|Home|
|1|Debt Consolidation|
|2|Education|
|3|Other|
|4|Auto|
